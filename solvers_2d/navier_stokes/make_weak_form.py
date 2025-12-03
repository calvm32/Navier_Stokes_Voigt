from firedrake import *
from .config import Re

def make_weak_form(theta, idt, f, g, dsN):
    """
    Returns func F(u, u_old, p, q, v), 
    which builds weak form
    using external coefficients
    """

    def F(u, p, u_old, p_old, v, q):
        u_mid = theta * u + (1 - theta) * u_old
    
        return (
            idt * inner(u - u_old, v) * dx
            + 1.0 / Re * inner(grad(u_mid), grad(v)) * dx
            + inner(dot(grad(u_mid), u_old), v) * dx
            - p * div(v) * dx
            + div(u_mid) * q * dx
            - inner(theta * f + (1 - theta) * f, v) * dx
            - inner(theta*g + (1-theta)*g, v) * dsN

        )
    
    return F