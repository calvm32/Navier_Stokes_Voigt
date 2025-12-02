from firedrake import *

# ---------
# Constants
# ---------

T = 1                   # final time
dt = 0.001               # timestepping length
theta = 1/2             # theta constant
Re = Constant(0.01)     # Reynold's num for viscosity

# ----------------
# For single solve 
# ----------------

N = 10                  # mesh size

# -------------
# For MMS solve
# -------------

P = Constant(10.0)      # pressure constant
H = 1.0                 # height of rectangle, with length = 3H

# MMS loops over mesh sizes in this list
N_list = []
for exp in range(4, 7):
    N = 2**exp
    N_list.append(N)