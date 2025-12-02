"""
Solver for the heat eqn:
    -> u_t - (1/Re)*lap(u) = f       in Omega x (0, T)
    -> partial u/ partial n = g      on bdy(Omega) x (0,T)
    -> u = u0                        on Omega x {0}

----

This code was adapted from https://fenics-handson.readthedocs.io/en/latest/heat/doc.html

"""

from .solver import *
from .MMS import *
