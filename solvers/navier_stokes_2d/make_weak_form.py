from firedrake import *
from .config_constants import Re, gamma_gd

def make_bilinear_and_linear_forms(theta, idt, f_new, f_old, g_new, g_old, U_old, dx, dsN):
    """
    Bilinear and linear forms for incompressible Navier–Stokes
      -? Crank-Nicolson
      -> Oseen linearization
      -> grad-div stabilization
      -> bilinear + linear
    """

    def forms(U, V):
        u, p = split(U)
        v, q = split(V)

        u_old = U_old.sub(0)

        # Midpoints
        u_mid = theta*u + (1.0 - theta)*u_old
        f_mid = theta*f_new.sub(0) + (1.0 - theta)*f_old.sub(0)
        g_mid = theta*g_new.sub(0) + (1.0 - theta)*g_old.sub(0)

        # Bilinear form a(U,V)
        a = (
            # Time derivative
            idt * inner(u, v) * dx

            # Oseen convection: (u_old · ∇)u
            + inner(dot(grad(u), u_old), v) * dx

            # Viscosity
            + (theta / Re) * inner(grad(u), grad(v)) * dx

            # Pressure / continuity
            - p * div(v) * dx
            - q * div(u) * dx

            # Grad–div stabilization
            + theta * gamma_gd * inner(div(u), div(v)) * dx
        )

        # Linear form L(V)
        L = (
            # Time derivative
            idt * inner(u_old, v) * dx

            # Explicit viscosity part
            - ((1.0 - theta) / Re) * inner(grad(u_old), grad(v)) * dx

            # Explicit grad–div
            - (1.0 - theta) * gamma_gd * inner(div(u_old), div(v)) * dx

            # Forcing
            + inner(f_mid, v) * dx

            # Neumann boundary traction
            + inner(g_mid, v) * dsN
        )

        return a, L

    return forms
