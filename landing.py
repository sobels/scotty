import casadi as ca
import math
import scipy.optimize
from typing import Tuple

from sim import collocate
from util import VarDict, Slice


class State(VarDict):
    r = Slice[0:3]
    v = Slice[3:6]
    lnm = Slice[6]


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


def skew_sym(x: ca.MX):
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

    return ca.Function('xdot', [x.var, u.var, p.var], [ca.vertcat(rdot, vdot, lnmdot)])


def base_landing_problem():
    opti, T, x, u, p = collocate(landing_dynamics(), 20, 3)

    x = State(x, axis=1)
    u = Control(u, axis=1)
    p = Params(p)

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

    return opti, T, x, u, p


def min_err_landing_problem():
    opti, T, x, u, p = base_landing_problem()

    err = x.r[-1, 1:] - p.q[1:].T
    opti.minimize(err @ err.T)

    return opti, T, x, u, p


def min_fuel_landing_problem():
    opti, T, x, u, p = base_landing_problem()

    opti.subject_to(x.r[-1, 1:] == p.q[1:].T)
    opti.minimize(-x.lnm[-1])

    return opti, T, x, u, p


def set_params(opti: ca.Opti, T: ca.MX, p: Params, tf: float, rf=None):
    opti.set_value(T, tf)

    opti.set_value(p.q, ca.vertcat(0, 0, 0))
    if rf is not None:
        opti.set_value(p.q[1:], rf[1:])

    opti.set_value(p.w, ca.vertcat(2.53e-5, 0, 6.62e-5))
    opti.set_value(p.g, 3.71)

    opti.set_value(p.min_thr, 0.2 * 24000)
    opti.set_value(p.max_thr, 0.8 * 24000)

    opti.set_value(p.ve, 1 / 5e-4)
    opti.set_value(p.mw, 2000)
    opti.set_value(p.md, 300)

    opti.set_value(p.r0, ca.vertcat(2400, 450, -330))
    opti.set_value(p.v0, ca.vertcat(-10, -40, 10))


def get_tl(h0: float, v0: float, g: float, mw: float, ve: float, max_thr: float) -> float:
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
        # tau = scipy.optimize.newton(vf, x0=0)
        return tau, hf(tau)  # type: ignore

    guess = (v0 + math.sqrt(v0**2 + 2 * h0 * g)) / g
    ts = scipy.optimize.newton(lambda ts: hf(ts)[1], x0=guess)

    return hf(ts)[0]  # type: ignore


min_err_state = min_err_landing_problem()


def solve_min_err(tf: float, min_err_x=None, min_err_dual=None):
    opti, T, x, u, p = min_err_state

    set_params(opti, T, p, tf)
    if min_err_x is not None:
        opti.set_initial(opti.x, min_err_x)
        if min_err_dual is not None:
            opti.set_initial(opti.lam_g, min_err_dual)
    else:
        opti.set_initial(x.lnm, ca.log(2000))
        opti.set_initial(u.acc, 1)

    opti.solver('ipopt', {'ipopt.sb': 'yes',
                'ipopt.print_level': 0, 'ipopt.suppress_all_output': 'yes'})
    opti.solve()

    rf = x.r[-1, :]
    return opti.value(opti.f), opti.value(rf), opti.value(opti.x), opti.value(opti.lam_g)


class AcceptableSolution(Exception):
    def __init__(self, t: float):
        self.time = t


def solve_min_err_and_time():
    tl = get_tl(2400, -10, 3.71, 2000, 1/5e-4, 0.8 * 24000)
    th = 3 * tl
    cached_err = None
    cached_rf = None
    cached_x = None
    cached_lam_g = None

    def to_opt(t: float):
        nonlocal cached_err
        nonlocal cached_rf
        nonlocal cached_x
        nonlocal cached_lam_g
        try:
            result, rf, cached_x, cached_lam_g = solve_min_err(
                t, cached_x, cached_lam_g)
        except:
            result = math.inf
            rf = None

        print(t, result)
        if cached_err is None or result < cached_err:
            cached_err = result
            cached_rf = rf

        if result < 1e-2:
            raise AcceptableSolution(result)

        return result

    try:
        return scipy.optimize.golden(to_opt, brack=(tl, th)), cached_rf
    except AcceptableSolution as e:
        return e.time, cached_rf


min_fuel_state = min_fuel_landing_problem()


def solve_min_fuel(tf: float, rf, min_fuel_x=None, min_fuel_dual=None):
    opti, T, x, u, p = min_fuel_state

    set_params(opti, T, p, tf, rf)

    if min_fuel_x is not None:
        opti.set_initial(opti.x, min_fuel_x)
        if min_fuel_dual is not None:
            opti.set_initial(opti.lam_g, min_fuel_dual)
    else:
        opti.set_initial(x.lnm, ca.log(2000))
        opti.set_initial(u.acc, 1)

    opti.solver('ipopt', {'ipopt.sb': 'yes',
                'ipopt.print_level': 0, 'ipopt.suppress_all_output': 'yes'})
    opti.solve()

    return opti.value(opti.f), opti.value(opti.x), opti.value(opti.lam_g)


def solve_min_fuel_and_time():
    t_min_err, rf_min_err = solve_min_err_and_time()

    tl = get_tl(2400, -10, 3.71, 2000, 1/5e-4, 0.8 * 24000)
    # In the paper they bound the search from above by 3 * tl.
    # This is really not necessary for us in practice, and hopping around such an extreme range hurts our solve time.
    # Instead, we use scipy's nifty feature where you don't explicitly provide an upper bound. Maybe it doesn't provide
    # the same guarantees on convergence though? Hard to say.
    # th = 3 * tl

    cached_x = None
    cached_lam_g = None

    def to_opt(t: float):
        nonlocal cached_x
        nonlocal cached_lam_g
        try:
            result, cached_x, cached_lam_g = solve_min_fuel(
                t, rf_min_err, cached_x)
        except:
            result = math.inf

        print(t, result)
        return result

    scipy.optimize.golden(to_opt, brack=(tl, tl + 1), tol=0.01)
