"""
Solver for the incompressible Navier-Stokes eqn:
    -> u_t = lap(u)                     in Omega x (0, T)
    -> periodic                         on bdy(Omega) x (0,T)
    -> u = u0                           on Omega x {0}
"""

from .solver import *