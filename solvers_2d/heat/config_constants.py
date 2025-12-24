from firedrake import *

# ---------
# Constants
# ---------

t0 = 0.0        # initial time
T = 1.0         # final time
dt = 1e-2       # timestepping length
theta = 1/2     # theta constant

vtkfile_name = "Soln"

# ----------------
# For single solve 
# ----------------

N = 10 # mesh resolution

# -------------
# For MMS solve
# -------------

# MMS loops over mesh resolutions in this list
N_list = []
for exp in range(1, 6):
    N = 2**exp
    N_list.append(N)

# -----------------
# Solver parameters
# -----------------

solver_parameters = {
    "ksp_type": "cg",
    "pc_type": "hypre"
}