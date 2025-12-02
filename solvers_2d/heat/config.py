from firedrake import *

# ---------
# Constants
# ---------

t = Constant(0.0)       # initial time
T = Constant(0.0)       # final time
dt = 0.1                # timestepping length
theta = 1/2             # theta constant

# ----------------
# For single solve 
# ----------------

N = 10          # mesh resolution

# -------------
# For MMS solve
# -------------

# MMS loops over mesh resolutions in this list
N_list = []
for exp in range(1, 10):
    N = 2**exp
    N_list.append(N)