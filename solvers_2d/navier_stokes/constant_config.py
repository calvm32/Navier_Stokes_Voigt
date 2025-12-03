from firedrake import *

# ---------
# Constants
# ---------

t0 = 0.0                # initial time
T = 1.0                 # final time
dt = 0.01             # timestepping length
theta = 1/2             # theta constant
Re = Constant(100)      # Reynold's num for viscosity

# -----------------
# Solver parameters
# -----------------

solver_parameters = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "snes_type": "newtonls",
    "snes_monitor": None,
    "snes_converged_reason": None,
    "ksp_monitor_true_residual": None,
    "ksp_converged_reason": None
}

# ----------------
# For single solve 
# ----------------

N = 64                  # mesh resolutions

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