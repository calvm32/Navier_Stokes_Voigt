from firedrake import *
import yaml
from pathlib import Path

from solvers_FEM.processing.config_setup import *

# --------------
# BDF2 Weak form
# --------------

from firedrake import *

def make_weak_form_BDF2(idt, f, f_old, g, g_old, U_older, U_old, dx, dsN, gamma, Re, alpha):
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
            # BDF2 mass
            (3.0 / 2.0) * idt * inner(u, v) * dx

            # Skew-symmetric Oseen advection
            + 0.5 * (
                inner(dot(u_old, nabla_grad(u)), v)
              - inner(dot(u_old, nabla_grad(v)), u)
            ) * dx

            # Viscosity
            + (1.0 / Re) * inner(grad(u), grad(v)) * dx

            # Pressure coupling
            - p * div(v) * dx
            - q * div(u) * dx
        )

        # Grad-div
        if gamma != 0.0:
            a += gamma * inner(div(u), div(v)) * dx

        # ------
        # RHS
        # ------
        L = (
            # BDF2 history
            0.5 * idt * inner(4.0*u_old - u_older, v) * dx

            # Forcing
            + inner(f_bdf2, v) * dx

            # Neumann BC
            + inner(g_bdf2, v) * dsN
        )

        return a, L

    return forms


# ---------
# Weak form
# ---------

def make_weak_form_CN(idt, f, f_old, g, g_old, U_old, dx, dsN, theta, gamma, Re, alpha):
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

            # Skew-symmetric Oseen advection
            + 0.5 * (
                inner(dot(u_old, nabla_grad(u)), v)
              - inner(dot(u_old, nabla_grad(v)), u)
            ) * dx

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
