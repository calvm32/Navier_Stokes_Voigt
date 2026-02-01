"""
solvers package: provides time-stepping for PDEs

----

We perform theta-scheme discretization, i.e.
    -> theta = 0     =>      explicit/forward Euler
    -> theta = 1/2   =>      Crank - Nicolson
    -> theta = 0     =>      implicit/backward Euler

---

Modules:
    -> timestepper_CN.py: fixed-step theta-scheme time integrator
    -> printoff: for logging and printing results

"""

from .timestepper_CN import timestepper
from .timestepper_BDF2 import timestepper
from .printoff import *
from .config_setup import *

__all__ = [
    "timestepper_CN",
    "printoff"
]