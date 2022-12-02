import casadi as ca
from dataclasses import dataclass, replace
from functools import cache
import numpy as np
import numpy.typing as npt

from .sim import collocate
from .util import VarDict, Slice, OptiProblem, OptiSol


class State(VarDict):
    r = Slice[0:3]
    v = Slice[3:6]
    lnm = Slice[6]


class Control(VarDict):
    acc = Slice[0:3]
    sigma = Slice[3]


class Params(VarDict):
    mu = Slice[0]
    min_thr = Slice[1]
    max_thr = Slice[2]
    ve = Slice[3]
    mw = Slice[4]
    md = Slice[5]
    r0 = Slice[6:9]
    v0 = Slice[10:13]
    rf = Slice[13]
    vf = Slice[14]


@dataclass(frozen=True)
class ParamSet:
    mu: float
    min_thr: float
    max_thr: float
    ve: float
    mw: float
    md: float
    r0: npt.NDArray
    v0: npt.NDArray
    rf: float
    vf: float


def dynamics():
    """Returns dynamics for a rocket thrusting in an inertial planet-centered frame"""

    x = State(ca.MX.sym('x', 7))  # type: ignore
    u = Control(ca.MX.sym('u', 3))  # type: ignore
    p = Params(ca.MX.sym('p', 13))  # type: ignore

    rdot = x.v
    vdot = u.acc - p.mu * x.r / ca.norm_2(x.r)**3
    lnmdot = -u.sigma / p.ve

    return ca.Function('xdot', [x.var, u.var, p.var], [ca.vertcat(rdot, vdot, lnmdot)])


def base_problem():
    opti, T, x, u, p = collocate(dynamics(), 20, 3)

    x = State(x, axis=1)
    u = Control(u, axis=1)
    p = Params(p)

    problem = OptiProblem(opti, T, x, u, p)

    opti.subject_to(x.lnm[0] == ca.log(p.mw))
    opti.subject_to(opti.bounded(ca.log(p.md), x.lnm[-1], ca.log(p.mw)))

    opti.subject_to(x.r[0, :] == p.r0.T)
    opti.subject_to(x.v[0, :] == p.v0.T)

    for i in range(u.var.shape[0]):
        tot_acc = ca.norm_2(u.acc[i, :])
        opti.subject_to(tot_acc <= u.sigma[i])

        opti.subject_to(opti.bounded(
            p.min_thr * ca.exp(-x.lnm[i * 4]),
            u.sigma[i],
            p.max_thr * ca.exp(-x.lnm[i * 4])
        ))

    return problem


@cache
def min_fuel_problem():
    result = base_problem()
    x = result.x
    u = result.u
    p = result.p

    result.opti.subject_to(ca.norm_2(x.r[-1, :]) == p.rf)
    result.opti.subject_to(ca.norm_2(x.v[-1, :]) == p.vf)
    result.opti.subject_to(x.r[-1, :].T @ x.v[-1, :] == 0)

    result.opti.minimize(-x.lnm[-1])

    return result


def set_params(problem: OptiProblem[State, Control, Params], tf: float, param_set: ParamSet):
    opti = problem.opti
    p = problem.p

    opti.set_value(problem.T, tf)

    opti.set_value(p.mu, param_set.mu)
    opti.set_value(p.min_thr, param_set.min_thr)
    opti.set_value(p.max_thr, param_set.max_thr)
    opti.set_value(p.ve, param_set.ve)
    opti.set_value(p.mw, param_set.mw)
    opti.set_value(p.md, param_set.md)
    opti.set_value(p.r0, param_set.r0)
    opti.set_value(p.v0, param_set.v0)
    opti.set_value(p.rf, param_set.rf)
    opti.set_value(p.vf, param_set.vf)


test_prob = ParamSet(
    mu=3.986004418e14,
    min_thr=0,
    max_thr=0.4 * 2000 * 9.8,
    ve=9.8 * 320,
    mw=2000,
    md=1,
    r0=np.zeros(3),
    v0=np.zeros(3),
    rf=0,
    vf=0
)
