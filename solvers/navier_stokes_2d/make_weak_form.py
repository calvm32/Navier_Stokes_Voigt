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
Re = cfg["Re"]

# ---------
# Weak form
# ---------

def make_weak_form(idt, f, f_old, g, g_old, U_old, dx, dsN):
    """
    Energy-stable skew-symmetric Crank–Nicolson
    incompressible Navier–Stokes
    """

    # Midpoint forcing
    f_mid = theta*f.sub(0) + (1.0 - theta)*f_old.sub(0)
    g_mid = theta*g.sub(0) + (1.0 - theta)*g_old.sub(0)

    u_old = U_old.sub(0)

    def forms(U, V):
        u, p = split(U)
        v, q = split(V)

        u_mid = theta*u + (1.0 - theta)*u_old

        # -------------
        # Nonlinear LHS
        # -------------

        a = (
            # Time derivative
            idt * inner(u, v) * dx

            # Skew-symmetric convection
            + 0.5 * (inner(dot(u_mid, nabla_grad(u)), v)
                - inner(dot(u_mid, nabla_grad(v)), u)) * dx

            # Viscosity (midpoint)
            + (1.0 / Re) * inner(grad(u_mid), grad(v)) * dx

            # Pressure / incompressibility
            - p * div(v) * dx
            - q * div(u) * dx
        )

        # Grad–div stabilization
        if gamma != 0:
            a += gamma * inner(div(u), div(v)) * dx

        # ----------
        # Linear RHS
        # ----------

        L = (
            idt * inner(u_old, v) * dx
            + inner(f_mid, v) * dx
            + inner(g_mid, v) * dsN
            + (1.0 / Re) * inner(g_mid, v) * dsN
        )

        return a, L

    return forms
