"""
Solver for the incompressible Navier-Stokes eqn:
    -> psi_t + inv(-lap)( nonlinear ) + alpha*
            = 1/Re lap(psi) + inv(lap)(curl x f)            in Omega x (0, T)
    -> periodic                                             on bdy(Omega) x (0,T)
    -> psi = psi0                                           on Omega x {0}
"""

from .solver import *