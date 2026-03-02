"""
solvers package: provides time-stepping for PDEs

----

We perform theta-scheme discretization, i.e.
    -> theta = 0     =>      explicit/forward Euler
    -> theta = 1/2   =>      Crank - Nicolson
    -> theta = 0     =>      implicit/backward Euler

---

Modules:
    -> timestepper_CN.py: fixed-step theta-scheme time step solver
    -> timestepper_BDF2.py: BDF2 time step solver
    -> printoff: for logging and printing results

"""

# individual files to use in solvers
from .timesteppers import timestepper_CN, timestepper_BDF2

# folder w/ statistics helper functions to use in timesteppers.py
from .statistics import *

# folder w/ processing helper functions to use throughout
from .processing import *