"""
solvers package: provides time-stepping for PDEs

----

We perform theta-scheme discretization, i.e.
    -> theta = 0     =>      explicit/forward Euler
    -> theta = 1/2   =>      Crank - Nicolson
    -> theta = 0     =>      implicit/backward Euler

----

Modules:
    - timestepper.py: fixed-step theta-scheme time integrator
    - timestepper_adaptive.py: adaptive time-stepping version

"""

from .timestepper import timestepper
#from .timestepper_adaptive import timestepper_adaptive

__all__ = [
    "timestepper",
#    "timestepper_adaptive",
]