from firedrake import *

from .config_constants import Re

def make_weak_form(theta, idt, f, g, dsN):
    """
    Returns func F(u, u_old, v), which builds weak form
    using external coefficients
    """

    def F(u, u_old, v):
        return (
            idt * inner(u - u_old, v) * dx
            + (1.0 / Re) * inner(grad(theta * u + (1 - theta) * u_old), grad(v)) * dx
            - inner(theta * f + (1 - theta) * f, v) * dx
            - inner(theta * g + (1 - theta) * g, v) * dsN
        )

    return F