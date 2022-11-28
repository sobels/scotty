import casadi as ca
from dataclasses import dataclass, replace
from functools import cache
import math
import numpy as np
import numpy.typing as npt
import scipy.optimize
from typing import Optional, Tuple

from sim import collocate
from util import VarDict, Slice, OptiProblem, OptiSol


class State(VarDict):
    r = Slice[0:3]
    v = Slice[3:6]
    lnm = Slice[6]

    @property
    def m(self):
        return ca.exp(self.lnm)


class Control(VarDict):
    acc = Slice[0:3]
    sigma = Slice[3]


class Params(VarDict):
    # TODO add thrust pointing and glide cone parameters, for now we'll hardcode acc[0] >= 0
    q = Slice[0:3]
    w = Slice[3:6]
    g = Slice[6]
    min_thr = Slice[7]
    max_thr = Slice[8]
    ve = Slice[9]
    mw = Slice[10]
    md = Slice[11]
    r0 = Slice[12:15]
    v0 = Slice[15:18]


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


def skew_sym(x: ca.MX | ca.DM):
    return ca.vertcat(
        ca.horzcat(0, -x[2], x[1]),
        ca.horzcat(x[2], 0, -x[0]),
        ca.horzcat(-x[1], x[0], 0)
    )


def landing_dynamics():
    """Returns dynamics as defined in http://www.larsblackmore.com/iee_tcst13.pdf.

    NOTE: coordinate system is surface fixed with first coordinate representing height."""

    x = State(ca.MX.sym('x', 7))  # type: ignore
    u = Control(ca.MX.sym('u', 4))  # type: ignore
    p = Params(ca.MX.sym('p', 18))  # type: ignore

    i, j, k = ca.horzsplit(ca.MX_eye(3))

    wx = skew_sym(p.w)

    rdot = x.v
    vdot = -wx @ wx @ x.r - 2 * wx @ x.v - p.g * i + u.acc
    lnmdot = -u.sigma / p.ve

    return ca.Function('xdot', [x.var, u.var, p.var], [ca.vertcat(rdot, vdot, lnmdot)]).expand()


def base_landing_problem():
    opti, T, x, u, p = collocate(landing_dynamics(), 20, 3)

    x = State(x, axis=1)
    u = Control(u, axis=1)
    p = Params(p)

    problem = OptiProblem(opti, T, x, u, p)

    # (5) glide cone constraints (TODO implement)

    # (7) mass constraints
    opti.subject_to(x.lnm[0] == ca.log(p.mw))
    opti.subject_to(opti.bounded(ca.log(p.md), x.lnm[-1], ca.log(p.mw)))

    # (8) and (9) boundary constraints on r, v
    opti.subject_to(x.r[0, :] == p.r0.T)
    opti.subject_to(x.v[0, :] == p.v0.T)
    opti.subject_to(x.r[-1, 0] == p.q[0])
    opti.subject_to(x.v[-1, :] == 0)

    # (18) thrust constraints
    for i in range(u.var.shape[0]):
        tot_acc = ca.norm_2(u.acc[i, :])
        opti.subject_to(tot_acc <= u.sigma[i])

        opti.subject_to(opti.bounded(
            p.min_thr * ca.exp(-x.lnm[i * 4]),
            u.sigma[i],
            p.max_thr * ca.exp(-x.lnm[i * 4])
        ))
        # zp = lnm[i * 4] - ca.log(mw - max_thr * i * T / 40 / ve)
        # opti.subject_to(opti.bounded(
        #     min_thr / mw * (1 - zp + 0.5 * zp**2),
        #     sigma[i],
        #     max_thr / mw * (1 - zp)
        # ))

    # (19) thrust pointing
    opti.subject_to(u.acc[:, 0] >= 0)

    return problem


@cache
def min_err_landing_problem():
    result = base_landing_problem()
    x = result.x
    p = result.p

    err = x.r[-1, 1:] - p.q[1:].T
    result.opti.minimize(err @ err.T)

    return result


@cache
def min_fuel_landing_problem():
    result = base_landing_problem()
    x = result.x
    u = result.u
    p = result.p

    result.opti.subject_to(x.r[-1, 1:] == p.q[1:].T)

    # While we could directly state to maximize the final mass, this equivalent condition greatly improves solver perf.
    result.opti.minimize(ca.sum1(u.sigma))

    return result


def set_params(problem: OptiProblem[State, Control, Params], tf: float, param_set: ParamSet):
    opti = problem.opti
    p = problem.p

    opti.set_value(problem.T, tf)

    opti.set_value(p.q, param_set.q)
    opti.set_value(p.w, param_set.w)
    opti.set_value(p.g, param_set.g)
    opti.set_value(p.min_thr, param_set.min_thr)
    opti.set_value(p.max_thr, param_set.max_thr)
    opti.set_value(p.ve, param_set.ve)
    opti.set_value(p.mw, param_set.mw)
    opti.set_value(p.md, param_set.md)
    opti.set_value(p.r0, param_set.r0)
    opti.set_value(p.v0, param_set.v0)


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


def solve_problem(problem: OptiProblem[State, Control, Params], tf: float, param_set: ParamSet, warm_sol: Optional[OptiSol] = None, warm=False):
    opti = problem.opti
    x = problem.x
    u = problem.u

    set_params(problem, tf, param_set)
    if warm_sol is None:
        # TODO provide a better initial guess using the closed-form solution for vertical-only coordinates.
        opti.set_initial(x.lnm, ca.log(2000))
        opti.set_initial(u.acc, 1)
    else:
        warm_sol.load(opti)

    opti.solver('ipopt', {
        'print_time': False,
        'ipopt.sb': 'yes',
        'ipopt.print_level': 0,
        'ipopt.warm_start_init_point': 'yes' if warm else 'no',
        'ipopt.warm_start_bound_frac': 1e-16,
        'ipopt.warm_start_bound_push': 1e-16,
        'ipopt.warm_start_mult_bound_push': 1e-16,
        'ipopt.warm_start_slack_bound_frac': 1e-16,
        'ipopt.warm_start_slack_bound_push': 1e-16
    })

    return OptiSol.save(opti.solve(), x, u)


class AcceptableSolution(Exception):
    def __init__(self, t: float):
        self.time = t


def solve_min_err_and_time(param_set: ParamSet) -> Tuple[float, float]:
    problem = min_err_landing_problem()
    tl = get_tl(param_set)
    th = 3 * tl

    sols = {}
    warm_sol = None

    def to_opt(t: float):
        nonlocal warm_sol
        try:
            warm_sol = solve_problem(problem, t, param_set, warm_sol)
        except:
            return math.inf

        rf = warm_sol.x.r[-1, :]
        sols[t] = rf

        rf = ca.norm_2(rf)
        if rf < 1e-2:
            raise AcceptableSolution(t)

        return rf

    try:
        time = scipy.optimize.golden(to_opt, brack=(tl, th))
        return time, sols[time]  # type: ignore
    except AcceptableSolution as e:
        return e.time, sols[e.time]  # type: ignore


def solve_min_fuel_and_time(param_set: ParamSet):
    problem = min_fuel_landing_problem()
    t_min_err, rf_min_err = solve_min_err_and_time(param_set)
    param_set = replace(param_set, q=rf_min_err)

    # In the paper they bound the search from above by 3 * tl.
    # This is really not necessary for us in practice, and hopping around such an extreme range hurts our solve time.
    # Instead, we use scipy's nifty feature where you don't explicitly provide an upper bound. Maybe it doesn't provide
    # the same guarantees on convergence though? Hard to say.
    # th = 3 * tl
    tl = get_tl(param_set)

    warm_sol = None

    def to_opt(t: float):
        nonlocal warm_sol
        try:
            warm_sol = solve_problem(problem, t, param_set, warm_sol)
        except:
            return math.inf

        return -warm_sol.x.lnm[-1]

    t = scipy.optimize.golden(to_opt, brack=(tl, tl + 1), tol=0.01)
    return solve_problem(problem, t, param_set, warm_sol)  # type: ignore


test_prob = ParamSet(
    q=np.zeros(3),
    w=np.array([2.53e-5, 0, 6.62e-5]),
    g=3.71,
    min_thr=0.2 * 24000,
    max_thr=0.8 * 24000,
    ve=2000,
    mw=2000,
    md=300,
    r0=np.array([2400, 450, -330]),
    v0=np.array([-10, -40, 10])
)

test_prob_2 = ParamSet(
    q=np.zeros(3),
    w=np.array([2.53e-5, 0, 6.62e-5]),
    g=3.71,
    min_thr=0.0 * 24000,
    max_thr=1.0 * 24000,
    ve=2000,
    mw=2000,
    md=300,
    r0=np.array([2400, 450, -330]),
    v0=np.array([-10, -40, 10])
)
