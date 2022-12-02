import casadi as ca


def collocate(dynamics: ca.Function, N: int, d: int):
    """Create an Opti instance and initially constrain it to the given dynamics using a direct collocation method with
    N d-degree polynomials. `dynamics` must take state, control, and parameter vectors and compute x'. Returns the Opti
    instance and the created variables:
    - T (final time)
    - x (state)
    - u (control)
    - p (parameters)"""

    tau = ca.collocation_points(d, 'legendre')
    C, D, B = ca.collocation_coeff(tau)

    opti = ca.Opti()

    # I'm not sure it's so principled to just make this an optimization variable, but empirically it works just fine.
    T = opti.parameter()
    h = T / N

    # Assumes x, u, and p each have shape of (n, 1) for various n
    nx, nu, np = [v.shape[0] for v in dynamics.mx_in()]

    # Each of the N polynomials of degree d is uniquely determined by d + 1 points. The endpoint is determined by
    # the continuity constraint with the last polynomial.
    x = opti.variable(N * (d + 1) + 1, nx)
    u = opti.variable(N, nu)
    p = opti.parameter(np)

    for k in range(N):
        lb = (d + 1) * k
        ub = lb + d + 1
        xc = x[lb:ub, :]

        # Derivative constraint: the polynomial must have a derivative equal to the given dynamics
        xp = C.T @ xc
        fj = ca.vertcat(
            *(dynamics(xc[j + 1, :].T, u[k, :].T, p).T for j in range(d)))  # type: ignore
        opti.subject_to(h * fj == xp)

        # Continuity constraint: the polynomial formed by the first `d` points must evaluate to the next point
        opti.subject_to(D.T @ xc == x[ub, :])

    return opti, T, x, u, p
