from firedrake import *
from .config_constants import Re, gamma_gd

def make_weak_form(theta, idt, f_new, f_old, g_new, g_old, dx, dsN):
    """
    Weak form for Navier-Stokes equations
        -> Crank Nicolson
        -> Oseen linearization
        -> grad-div stabilization
    """

    def F(U, U_old, V):
        u, p = split(U)
        v, q = split(V)

        u_old = U_old.sub(0)

        # midpoints
        u_mid = theta*u + (1.0 - theta)*u_old
        f_mid = theta*f_new.sub(0) + (1.0 - theta)*f_old.sub(0)

        # Momentum equation
        F_mom = (
            # Time derivative
            idt * inner(u - u_old, v) * dx

            # Oseen convection: (w * grad)u
            + inner(dot(grad(u), u_mid), v) * dx

            # Viscosity
            + (1.0 / Re) * inner(grad(u_mid), grad(v)) * dx

            # Pressure
            - p * div(v) * dx

            # Forcing
            - inner(f_mid, v) * dx
        )

        # Continuity equation
        F_cont = - q * div(u_mid) * dx

        # Grad–div stabilization
        F_gd = gamma_gd * inner(div(u_mid), div(v)) * dx

        return F_mom + F_cont + F_gd

    return F
