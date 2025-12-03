from firedrake import *

from .config_constants import Re

def make_weak_form(theta, idt, f, f_old, g, g_old, dx , dsN):
    """
    Returns func F(u, u_old, p, q, v), 
    which builds weak form
    using external coefficients
    """

    f_mid = theta * f + (1-theta) * f_old
    g_mid = theta * g + (1-theta) * g_old

    def F(u, p, u_old, p_old, v, q):
        u_mid = theta * u + (1 - theta) * u_old
    
        return (
            # Time derivative
            idt*inner(u - u_old, v)*dx

            # Diffusion
            + (1/Re)*inner(grad(u_mid), grad(v))*dx

            # Convection — Crank–Nicolson (implicit midpoint)
            + inner(dot(u_mid, grad(u_mid)), v)*dx

            # Pressure
            - inner(p, div(v))*dx
            + inner(div(u_mid), q)*dx

            # Source, boundary
            - inner(f_mid, v)*dx
            - inner(g_mid, v)*dsN
        )
    
    return F