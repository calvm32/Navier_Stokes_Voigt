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
    -> timestepper_MMS.py: fixed-step theta-scheme time integrator that returns only a final error
    -> printoff: for logging and printing results

"""

from .timestepper import timestepper
from .timestepper_MMS import timestepper_MMS
from .printoff import *

__all__ = [
    "timestepper",
    "timestepper_MMS",
    "printoff"
]