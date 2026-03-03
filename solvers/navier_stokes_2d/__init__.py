"""
Solver for the incompressible Navier-Stokes eqn:
    -> u_t + (u * grad)u - 1/Re*lap(u) + grad p = f         in Omega x (0, T)
    -> partial u/ partial n = g                             on bdy(Omega) x (0,T)
    -> u = u0                                               on Omega x {0}
"""

from .solver import *
from .solver_MMS import *
