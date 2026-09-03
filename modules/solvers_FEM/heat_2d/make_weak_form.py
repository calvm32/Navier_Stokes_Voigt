from firedrake import *

# --------------
# BDF2 Weak form
# --------------

def make_weak_form_BDF2(idt, f, f_old, g, g_old, u_older, u_old, dx, ds, gamma=0.0, Re=1.0):
    """
    Bilinear and linear forms for heat equation
      -> BDF2 time-stepping
      -> bilinear, linear
    """

    def forms(u, v):

        # -------------
        # Bilinear form
        # -------------
        a = (
            (3.0/2.0) * idt * u * v * dx
            + inner(grad(u), grad(v)) * dx
        )

        # -----------
        # Linear form
        # -----------
        L = (
            idt * (2.0 * u_old - 0.5 * u_older) * v * dx
            + f * v * dx
            + g * v * ds
        )

        return a, L

    return forms

# ------------
# CN Weak form
# ------------

def make_weak_form_CN(idt, f, f_old, g, g_old, u_old, dx, ds, theta, gamma=0.0, Re=1.0):
    """
    Bilinear and linear forms for heat equation
      -> Crank-Nicolson
      -> bilinear, linear
    """
    
    # Midpoints
    f_mid = theta*f + (1.0 - theta)*f_old
    g_mid = theta*g + (1.0 - theta)*g_old

    def forms(u, v):
        
        # Bilinear form a(u,v)
        a = (
            idt * u * v * dx
            + theta * inner(grad(u), grad(v)) * dx
        )

        # Linear form L(v)
        L = (
            idt * u_old * v * dx
            - (1.0 - theta) * inner(grad(u_old), grad(v)) * dx
            + f_mid * v * dx
            + g_mid * v * ds
        )

        return a, L

    return forms
