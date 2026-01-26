from firedrake import *
import yaml
from pathlib import Path

from solvers.config_setup import *

# -------------
# Configuration
# -------------

CFG_PATH1 = Path(__file__).parent / "configs" / "USER_constants.yaml"
cfg = load_config(CFG_PATH1)

theta = cfg["theta"]
gamma = cfg["gamma"]
Re    = cfg["Re"]

# ---------
# Weak form
# ---------

def make_weak_form(idt, f, f_old, g, g_old, U_old, dx, dsN):
    """
    Bilinear and linear forms for incompressible Navier-Stokes
      -> Crank-Nicolson
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
            idt * inner(u, v) * dx

            # Oseen convection: (u_old * grad)u
            + inner(dot(grad(u), u_old), v) * dx

            # Viscosity
            + (theta / Re) * inner(grad(u), grad(v)) * dx

            # Pressure / continuity
            - p * div(v) * dx
            - q * div(u) * dx

            # Grad–div stabilization
             + theta * gamma * inner(div(u), div(v)) * dx

        )

        # Linear form L(V)
        L = (
            # Time derivative
            idt * inner(u_old, v) * dx

            # Explicit viscosity
            - ((1.0 - theta) / Re) * inner(grad(u_old), grad(v)) * dx

            # Explicit grad–div
             - (1.0 - theta) * gamma * inner(div(u_old), div(v)) * dx

            # Forcing
            + inner(f_mid, v) * dx

            # Neumann boundary
            + inner(g_mid, v) * dsN

        )

        return a, L

    return forms
