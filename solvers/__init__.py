"""
solvers package: provides time-stepping for PDEs

----

We perform theta-scheme discretization, i.e.
    -> theta = 0     =>      explicit/forward Euler
    -> theta = 1/2   =>      Crank - Nicolson
    -> theta = 0     =>      implicit/backward Euler

---

Modules:
    -> timestepper.py: fixed-step theta-scheme time integrator
    -> printoff: for logging and printing results

"""

from .timestepper import timestepper
from .printoff import *

__all__ = [
    "timestepper",
    "printoff"
]