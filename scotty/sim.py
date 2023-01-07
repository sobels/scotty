import casadi as ca
import cvxpy as cp
from typing import Any, TypeVar

from .util import OptiVars, VarDict, T


X = TypeVar('X', bound=VarDict[cp.Expression])
U = TypeVar('U', bound=VarDict[cp.Expression])
P = TypeVar('P', bound=VarDict[cp.Expression])


def vars(N: int, d: int, State: type[X], Control: type[U], Params: type[P]):
    return OptiVars(
        T=cp.Parameter(),
        Tinv=cp.Parameter(),
        x=State(cp.Variable((N * (d + 1) + 1, State.width())), axis=1),
        u=Control(cp.Variable((N, Control.width())), axis=1),
        p=Params(cp.Parameter((Params.width(),)))
    )


def constraints(dynamics: Any, vars: OptiVars[VarDict[T], VarDict[T], VarDict[T]]):
    """Create an Opti instance and initially constrain it to the given dynamics using a direct collocation method with
    N d-degree polynomials. `dynamics` must take state, control, and parameter vectors and compute x'. Returns the Opti
    instance and the created variables:
    - T (final time)
    - x (state)
    - u (control)
    - p (parameters)"""

    x = vars.x.var
    u = vars.u.var
    p = vars.p.var

    N = u.shape[0]

    assert (x.shape[0] - 1) % N == 0
    d = (x.shape[0] - 1) // N - 1

    h = (1 / N) * vars.T
    hinv = N * vars.Tinv

    tau = ca.collocation_points(d, 'legendre')
    C, D, B = ca.collocation_coeff(tau)
    C = C.toarray()
    D = D.toarray()
    B = B.toarray()

    for k in range(N):
        lb = (d + 1) * k
        ub = lb + d + 1
        xc = x[lb:ub, :]

        # Derivative constraint: the polynomial must have a derivative equal to the given dynamics
        xp = C.T @ xc
        fj = cp.vstack(
            (dynamics(xc[j + 1, :].T, u[k, :].T, p).T for j in range(d)))  # type: ignore
        yield fj == hinv * xp

        # Continuity constraint: the polynomial formed by the first `d` points must evaluate to the next point
        yield D.T @ xc == x[ub:ub+1, :]
