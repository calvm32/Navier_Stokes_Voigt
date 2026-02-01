from firedrake import *
import yaml
from pathlib import Path

from solvers.config_setup import *

# -------------
# Configuration
# -------------

CFG_PATH1 = Path(__file__).parent / "configs" / "constants.yaml"
cfg = load_config(CFG_PATH1)

theta = cfg["theta"]
gamma = cfg["gamma"]
Re = cfg["Re"]

# --------------
# BDF2 Weak form
# --------------

from firedrake import *

def make_weak_form_BDF2(idt, f, f_old, g, g_old, U_older, U_old, dx, dsN):
    """
    Bilinear and linear forms for Navier-Stokes equation
      -> BDF2 time stepping
      -> Oseen linearization about u^n
      -> skew-symmetric convection
      -> fully implicit viscosity
      -> implicit grad-div stabilization
      -> second-order extrapolated forcing
    """

    # Extrapolate
    f_bdf2 = 2.0 * f.sub(0) - f_old.sub(0)
    g_bdf2 = 2.0 * g.sub(0) - g_old.sub(0)

    # Velocities
    u_old,  _ = split(U_old)
    u_older, _ = split(U_older)

    def forms(U, V):
        u, p = split(U)
        v, q = split(V)

        # -------------
        # Bilinear form
        # -------------
        a = (
            # BDF2 time derivative: (3 / 2 dt) u^{n+1}
            (3.0 / 2.0) * idt * inner(u, v) * dx

            # Skew-symmetric Oseen convection
            + 0.5 * (
                inner(dot(grad(u), u_old), v)
                - inner(dot(grad(v), u_old), u)
            ) * dx

            # Viscosity (implicit)
            + (1.0 / Re) * inner(grad(u), grad(v)) * dx

            # Incompressibility
            - p * div(v) * dx
            - q * div(u) * dx
        )

        # Grad–div stabilization (implicit, recommended)
        if gamma != 0.0:
            a += gamma * inner(div(u), div(v)) * dx

        # -----------
        # Linear form
        # -----------
        L = (
            # BDF2 history: (4 u^n - u^{n-1}) / (2 dt)
            idt * inner(2.0 * u_old - 0.5 * u_older, v) * dx

            # Forcing (BDF2 extrapolated)
            + inner(f_bdf2, v) * dx

            # Neumann boundary (BDF2 extrapolated)
            + inner(g_bdf2, v) * dsN
        )

        return a, L

    return forms

# ---------
# Weak form
# ---------

def make_weak_form_CN(idt, f, f_old, g, g_old, U_old, dx, dsN):
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

            # Oseen convection: skew convection (u_old * grad)u
            + 0.5 * (inner(dot(grad(u), u_old), v) - inner(dot(grad(v), u_old), u)) * dx

            # Viscosity
            + (theta / Re) * inner(grad(u), grad(v)) * dx

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
            idt * inner(u_old, v) * dx

            # Explicit viscosity
            - ((1.0 - theta) / Re) * inner(grad(u_old), grad(v)) * dx

            # Forcing
            + inner(f_mid, v) * dx

            # Neumann boundary
            + inner(g_mid, v) * dsN
        )

        # Grad–div stabilization
        if gamma != 0:
            L -= (1.0 - theta) * gamma * inner(div(u_old), div(v)) * dx

        return a, L

    return forms
