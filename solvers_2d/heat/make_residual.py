from firedrake import *

def make_residual(u, k, v, f, g, dx, dsN):
    """
    Returns residual, basically the explicit weak form?
    """
    return (
        inner(k, v)*dx
        + inner(grad(u), grad(v))*dx
        - f*v*dx
        - g*v*dsN
    )