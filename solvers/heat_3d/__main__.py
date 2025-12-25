"""
Solver for the heat eqn:
    -> u_t - (1/Re)*lap(u) = f       in Omega x (0, T)
    -> partial u/ partial n = g      on bdy(Omega) x (0,T)
    -> u = u0                        on Omega x {0}
"""

from .solver import *
from .solver_MMS import *
