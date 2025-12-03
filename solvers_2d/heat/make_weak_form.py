from firedrake import *

from .config_constants import Re

def make_weak_form(theta, idt, f, f_old, g, g_old, dsN):
    """
    Returns func F(u, u_old, v), which builds weak form
    using external coefficients
    """
    f_mid = theta * f + (1-theta) * f_old
    g_mid = theta * g + (1-theta) * g_old

    def F(u, u_old, v):
        u_mid = theta*u + (1-theta)*u_old
        F = ( idt*(u - u_old)*v*dx
            + inner(grad(u_mid), grad(v))*dx
            - f_mid*v*dx - g_mid*v*dsN
        )

    return F