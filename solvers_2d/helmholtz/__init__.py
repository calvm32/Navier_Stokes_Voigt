"""
Solver for the indefinite Helmholtz eqn used in wave problems:
    -> nu*(div)^2 u + u = f      in Omega
    -> (div u) n = g             on bdy(Omega)

----

This code was adapted from https://www.firedrakeproject.org/demos/helmholtz.py.html

"""

from .helmholtz import *