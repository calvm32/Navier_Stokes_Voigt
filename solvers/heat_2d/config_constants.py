from firedrake import *

# ---------
# Constants
# ---------

t0 = 0.0        # initial time
T = 1.0         # final time
dt = 1e-2       # timestepping length
theta = 1/2     # theta constant
N = 10          # mesh resolution

vtkfile_name = "Soln"

# -----------------
# Solver parameters
# -----------------

solver_parameters = {
    "ksp_type": "cg",
    "pc_type": "hypre"
}