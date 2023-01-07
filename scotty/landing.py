import cvxpy as cp
from dataclasses import dataclass, replace
from functools import cache
import math
import numpy as np
import numpy.typing as npt
import scipy.optimize
import sys
from typing import Tuple

from . import sim
from .util import VarDict, Slice, OptiVars


class State(VarDict[cp.Expression]):
    r = Slice[0:3]
    v = Slice[3:6]
    lnm = Slice[6]

    @property
    def m(self):
        return cp.exp(self.lnm)


class Control(VarDict[cp.Expression]):
    acc = Slice[0:3]
    sigma = Slice[3]


class Params(VarDict[cp.Expression]):
    # TODO add thrust pointing and glide cone parameters, for now we'll hardcode acc[0] >= 0
    q = Slice[0:3]
    w = Slice[3:6]
    g = Slice[6]
    min_thr = Slice[7]
    max_thr = Slice[8]
    alpha = Slice[9]
    mw = Slice[10]
    md = Slice[11]
    logmw = Slice[12]
    logmd = Slice[13]
    r0 = Slice[14:17]
    v0 = Slice[17:20]
    z0 = Slice[20:40]
    d1 = Slice[40:60]
    d2 = Slice[60:80]


@dataclass(frozen=True)
class ParamSet:
    q: npt.NDArray
    w: npt.NDArray
    g: float
    min_thr: float
    max_thr: float
    ve: float
    mw: float
    md: float
    r0: npt.NDArray
    v0: npt.NDArray


def skew_sym(x):
    return cp.bmat([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])


def landing_dynamics(x, u, p):
    """Returns dynamics as defined in http://www.larsblackmore.com/iee_tcst13.pdf.

    NOTE: coordinate system is surface fixed with first coordinate representing height."""

    _x = State(x)  # type: ignore
    _u = Control(u)  # type: ignore
    _p = Params(p)  # type: ignore

    i, j, k = cp.diag([1, 1, 1])

    wx = skew_sym(_p.w)

    rdot = _x.v
    # vdot = -wx @ wx @ _x.r - 2 * wx @ _x.v - _p.g * i + _u.acc
    vdot = -_p.g * i + _u.acc
    lnmdot = -_p.alpha * _u.sigma

    return cp.hstack([rdot, vdot, lnmdot]).T


def base_landing_problem():
    vars = sim.vars(20, 3, State, Control, Params)
    constraints = list(sim.constraints(landing_dynamics, vars))

    T = vars.T
    x = vars.x
    u = vars.u
    p = vars.p

    # (5) glide slope constraints. For now, hardcode to a maximum glide slope of 45deg.
    glide_slope = np.pi / 4
    constraints += [cp.norm2(x.r[i, 1:] - x.r[-1, 1:]) - np.tan(glide_slope)
                    * (x.r[i, 0] - x.r[-1, 0]) <= 0 for i in range(x.var.shape[0])]

    # (7) mass constraints
    constraints += [x.lnm[0] == p.logmw]
    constraints += [p.logmd <= x.lnm[-1], x.lnm[-1] <= p.logmw]

    # (8) and (9) boundary constraints on r, v
    constraints += [x.r[0, :] == p.r0]
    constraints += [x.v[0, :] == p.v0]
    constraints += [x.r[-1, 0] == p.q[0]]
    constraints += [x.v[-1, :] == 0]

    # (18) thrust constraints
    for i in range(u.var.shape[0]):
        tot_acc = cp.norm2(u.acc[i, :])
        constraints += [tot_acc <= u.sigma[i]]

        zp = x.lnm[i * 4] - p.z0[i]
        constraints += [
            1 - zp + 0.5 * cp.square(zp) <= p.d1[i] * u.sigma[i],
            # 0 <= 1 * u.sigma[i],
            p.d2[i] * u.sigma[i] <= 1 - zp
        ]

    # (19) thrust pointing
    constraints += [u.acc[:, 0] >= 0]

    return vars, constraints


@cache
def min_err_landing_problem():
    vars, constraints = base_landing_problem()
    x = vars.x
    u = vars.u
    p = vars.p

    err = x.r[-1, 1:] - p.q[1:].T
    goal = cp.Minimize(cp.norm2(err))

    return vars, cp.Problem(goal, constraints)


@cache
def min_fuel_landing_problem():
    vars, constraints = base_landing_problem()
    x = vars.x
    u = vars.u
    p = vars.p

    constraints += [x.r[-1, 1:] == p.q[1:].T]

    # While we could directly state to maximize the final mass, this equivalent condition greatly improves solver perf.
    goal = cp.Minimize(cp.norm1(u.sigma))
    return vars, cp.Problem(goal, constraints)


def set_params(vars, tf: float, param_set: ParamSet):
    t = np.array([i * tf / 20 for i in range(20)])
    m0 = param_set.mw - param_set.max_thr * t / param_set.ve
    z0 = np.log(m0)
    d1 = m0 / \
        (param_set.min_thr if param_set.min_thr > 0 else 0.001)
    d2 = m0 / param_set.max_thr

    vars.T.value = tf
    vars.Tinv.value = 1 / tf
    vars.p.var.value = np.array([
        *param_set.q,
        *param_set.w,
        param_set.g,
        param_set.min_thr,
        param_set.max_thr,
        1 / param_set.ve,
        param_set.mw,
        param_set.md,
        np.log(param_set.mw),
        np.log(param_set.md),
        *param_set.r0,
        *param_set.v0,
        *z0,
        *d1,
        *d2
    ])


def get_tl(param_set: ParamSet) -> float:
    h0 = param_set.r0[0]
    v0 = param_set.v0[0]
    g = param_set.g
    mw = param_set.mw
    ve = param_set.ve
    max_thr = param_set.max_thr
    k = max_thr / ve

    def hf(ts: float) -> Tuple[float, float]:
        hts = -0.5 * g * ts**2 + ts * v0 + h0
        vts = -g * ts + v0

        def tburn(tau: float):
            return tau - ts

        def dv(tau: float):
            return math.log(mw / (mw - k * tburn(tau)))

        def hf(tau: float):
            tburn = tau - ts
            return hts + tburn * vts - 0.5 * g * tburn**2 + ve * (tburn + (tburn - mw/k) * dv(tau))

        def vf(tau: float):
            tburn = tau - ts
            return vts - g * tburn + ve * dv(tau)

        def vfdot(tau: float):
            tburn = tau - ts
            return -g + k * ve / (mw - k * tburn)

        tau = scipy.optimize.newton(vf, x0=0, fprime=vfdot)
        return tau, hf(tau)  # type: ignore

    guess = (v0 + math.sqrt(v0**2 + 2 * h0 * g)) / g
    ts = scipy.optimize.newton(lambda ts: hf(ts)[1], x0=guess)

    return hf(ts)[0]  # type: ignore


def solve_problem(vars, problem, tf: float, param_set: ParamSet):
    x = vars.x
    u = vars.u
    p = vars.p

    set_params(vars, tf, param_set)
    problem.solve(solver='ECOS', warm_start=True, enforce_dpp=True)

    if problem.status == 'infeasible' or problem.status == 'infeasible_inaccurate':
        raise Exception(problem.status)

    return OptiVars(tf, 1 / tf, State(x.var.value, axis=1), Control(u.var.value, axis=1), Params(vars.p.var.value))


class AcceptableSolution(Exception):
    def __init__(self, t: float):
        self.time = t


def solve_min_err_and_time(param_set: ParamSet) -> Tuple[float, float]:
    vars, problem = min_err_landing_problem()
    tl = get_tl(param_set)
    th = min(3 * tl, param_set.mw * param_set.ve / param_set.max_thr - 1)

    sols = {}

    def to_opt(t: float):
        try:
            sol = solve_problem(vars, problem, t, param_set)
        except:
            return math.inf

        rf = sol.x.r[-1, :]
        sols[t] = rf

        rf = cp.norm2(rf).value
        if rf < 1e-2:
            raise AcceptableSolution(t)

        return rf

    try:
        time = scipy.optimize.brent(to_opt, brack=(tl, th))
        return time, sols[time]  # type: ignore
    except AcceptableSolution as e:
        return e.time, sols[e.time]  # type: ignore


def solve_min_fuel_and_time(param_set: ParamSet):
    vars, problem = min_fuel_landing_problem()
    t_min_err, rf_min_err = solve_min_err_and_time(param_set)
    param_set = replace(param_set, q=rf_min_err)

    # In the paper they bound the search from above by 3 * tl.
    # This is really not necessary for us in practice, and hopping around such an extreme range hurts our solve time.
    # Instead, we use scipy's nifty feature where you don't explicitly provide an upper bound. Maybe it doesn't provide
    # the same guarantees on convergence though? Hard to say.
    # th = 3 * tl
    # tl = get_tl(param_set)
    tl = t_min_err

    def to_opt(t: float):
        try:
            sol = solve_problem(vars, problem, t, param_set)
        except:
            return math.inf

        return -sol.x.lnm[-1]

    t = scipy.optimize.brent(to_opt, brack=(tl, tl + 1), tol=0.01)
    return solve_problem(vars, problem, t, param_set)  # type: ignore
