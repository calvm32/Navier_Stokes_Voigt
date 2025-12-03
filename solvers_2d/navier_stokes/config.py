from firedrake import *

# ---------
# Constants
# ---------

t0 = 0.0        # initial time
T = 1.0                 # final time
dt = 0.0001             # timestepping length
theta = 1/2             # theta constant
Re = Constant(200)      # Reynold's num for viscosity

# -----------------
# Solver parameters
# -----------------

solver_parameters = {
    "mat_type": "matfree",
    "ksp_type": "fgmres",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "lower",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "python",
    "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
    "fieldsplit_0_assembled_pc_type": "lu",
    "fieldsplit_1_ksp_type": "gmres",
    "fieldsplit_1_pc_type": "python",
    "fieldsplit_1_pc_python_type": "firedrake.PCDPC",
    "fieldsplit_1_pcd_Mp_pc_type": "lu",
    "fieldsplit_1_pcd_Kp_pc_type": "lu",
    "fieldsplit_1_pcd_Fp_mat_type": "matfree",
    "snes_monitor": None,
    "snes_converged_reason": None,
    "ksp_monitor_true_residual": None,
    "ksp_converged_reason": None,
}

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
for exp in range(3, 7):
    N = 2**exp
    N_list.append(N)