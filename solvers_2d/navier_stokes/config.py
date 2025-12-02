from firedrake import *

# ---------
# Constants
# ---------

t = Constant(0.0)       # initial time
T = Constant(0.0)       # final time
dt = 0.0001              # timestepping length
theta = 1/2             # theta constant
Re = Constant(200)      # Reynold's num for viscosity

# ----------------
# For single solve 
# ----------------

N = 10                  # mesh resolutions

# -------------
# For MMS solve
# -------------

P = Constant(10.0)      # pressure constant
H = 1.0                 # height of rectangle, with length = 3H

# MMS loops over mesh resolutions in this list
N_list = []
for exp in range(4, 7):
    N = 2**exp
    N_list.append(N)