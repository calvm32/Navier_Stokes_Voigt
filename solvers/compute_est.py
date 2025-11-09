from firedrake import *

# works for scalars and vectors

def compute_est(theta, u_L, u_H):
    """Return error estimate by Richardson extrapolation"""
    p = 2 if theta == 0.5 else 1
    diff = u_L - u_H
    est = sqrt(assemble(inner(diff, diff)**2*dx)) / (2**p - 1)
    return est