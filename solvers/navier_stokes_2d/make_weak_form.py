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

        # Midpoint velocity
        u_mid = theta*u + (1.0 - theta)*u_old

        # Time derivative
        a = idt * inner(u, v) * dx

        # Skew-symmetric convection
        a += 0.5 * (
            inner(dot(grad(u), u_mid), v)
            - inner(dot(grad(v), u_mid), u)
        ) * dx

        # Viscosity (midpoint)
        a += (1.0 / Re) * inner(grad(u_mid), grad(v)) * dx

        # Pressure / incompressibility
        a += -p * div(v) * dx
        a += -q * div(u) * dx

        # Grad–div stabilization
        if gamma != 0:
            a += gamma * inner(div(u), div(v)) * dx

        # ----------------
        # Linear RHS
        # ----------------

        L = idt * inner(u_old, v) * dx
        L += inner(f_mid, v) * dx
        L += inner(g_mid, v) * dsN

        return a, L

    return forms
