from firedrake import *

def voigt_inner(u, v, alpha):
    return inner(u, v) + alpha**2 * inner(grad(u), grad(v))

# --------------
# BDF2 weak form
# --------------

def make_weak_form_BDF2(idt, f, f_old, g, g_old, U_older, U_old, dx, dsN, gamma, nu, alpha):
    """
    BDF2 Navier-Stokes
    - Oseen linearization
    - skew-symmetric convection
    - implicit viscosity
    - implicit grad-div
    """

    # Extrapolated forcing
    f_bdf2 = 2.0*f.sub(0) - f_old.sub(0)
    g_bdf2 = 2.0*g.sub(0) - g_old.sub(0)

    # Old velocities
    u_old,  _ = split(U_old)
    u_older, _ = split(U_older)

    def forms(U, V):
        u, p = split(U)
        v, q = split(V)

        # ------
        # LHS
        # ------
        a = (
            # time derivative
            idt * voigt_inner((3/2)*u, v, alpha) * dx

            # Viscosity
            + nu * inner(grad(u), grad(v)) * dx

            # Skew-symmetric Oseen advection
            + 0.5 * (
                inner(dot(u_old, nabla_grad(u)), v)
              - inner(dot(u_old, nabla_grad(v)), u)
            ) * dx

            # Pressure coupling
            - p * div(v) * dx
            - q * div(u) * dx
        )

        # Grad-div stabilization
        if gamma != 0.0:
            a += gamma * inner(div(u), div(v)) * dx

        # ------
        # RHS
        # ------
        L = (
            # BDF2 history
            idt * voigt_inner(2*u_old - (1/2)*u_older, v, alpha) * dx

            # Forcing
            + inner(f_bdf2, v) * dx

            # Neumann BC
            - nu * inner(g_bdf2, v) * dsN
        )

        return a, L

    return forms


# ------------
# CN weak form
# ------------

def make_weak_form_CN(idt, f, f_old, g, g_old, U_old, dx, dsN, theta, gamma, nu, alpha):
    """
    Crank-Nicolson Navier-Stokes-Voigt
      -> Oseen linearization
      -> grad-div stabilization
      -> bilinear, linear
    """

    # Midpoints
    f_mid = theta*f.sub(0) + (1.0 - theta)*f_old.sub(0)
    g_mid = theta*g.sub(0) + (1.0 - theta)*g_old.sub(0)

    u_old = U_old.sub(0)
    p_old = U_old.sub(1)

    def forms(U, V):
        u, p = split(U)
        v, q = split(V)

        # Bilinear form a(U,V)
        a = (
            # Time derivative
            idt * voigt_inner(u, v, alpha) * dx

            # Skew-symmetric Oseen advection
            + 0.5 * (
                inner(dot(u_old, nabla_grad(u)), v)
              - inner(dot(u_old, nabla_grad(v)), u)
            ) * dx

            # Viscosity
            + theta*nu * inner(grad(u), grad(v)) * dx

            # Pressure / continuity
            - p * div(v) * dx
            - q * div(u) * dx
        )

        # Grad–div stabilization
        if gamma != 0:
            a += theta * gamma * inner(div(u), div(v)) * dx

        # Linear form L(V)
        L = (
            # Time derivative
            idt * voigt_inner(u_old, v, alpha) * dx

            # Explicit viscosity
            - (1 - theta)*nu * inner(grad(u_old), grad(v)) * dx

            # Forcing
            + inner(f_mid, v) * dx

            # Neumann boundary
            - nu * inner(g_mid, v) * dsN 
        )

        # Grad–div stabilization
        if gamma != 0:
            L -= (1 - theta) * gamma * inner(div(u_old), div(v)) * dx

        return a, L

    return forms
