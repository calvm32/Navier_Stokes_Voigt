from firedrake import *

# ---------
# Constants
# ---------

t0 = 0.0                # initial time
T = 1.0                 # final time
dt = 0.001              # timestepping length
theta = 1/2             # theta constant
Re = Constant(100)      # Reynold's num for viscosity

H = 1.0                 # height of box; length = 3*H

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
for exp in range(1, 10):
    N = 2**exp
    N_list.append(N)

P = 5.0                 # initial pressure strength

# -----------------
# Solver parameters
# -----------------

solver_parameters = {
    "mat_type": "matfree",
    "ksp_type": "gmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    "pc_fieldsplit_schur_precondition": "selfp",

    # blocks
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "hypre",
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "jacobi",

    # add velocity space for PCD in solver
    "appctx": { }
}