"""
Solver for the incompressible Navier-Stokes eqn:
    -> psi_t + inv(-lap)( (grad^T psi)* grad(lap psi) ) 
            = 1/Re lap(psi) + inv(lap)grad^T * grad(f)      in Omega x (0, T)
    -> partial psi/ partial n = g                           on bdy(Omega) x (0,T)
    -> psi = psi0                                           on Omega x {0}
"""

from .solver import *