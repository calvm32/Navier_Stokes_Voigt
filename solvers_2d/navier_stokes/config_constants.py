from firedrake import *

# ---------
# Constants
# ---------

t0 = 0.0                # initial time
T = 1.0                 # final time
dt = 0.01             # timestepping length
theta = 1/2             # theta constant
Re = Constant(100)      # Reynold's num for viscosity

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

# -----------------
# Solver parameters
# -----------------

solver_parameters = {
    "mat_type": "matfree",
    "ksp_type": "fgmres",
    "pc_type": "fieldsplit",
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
    #"snes_monitor": None,
    #"snes_converged_reason": None,
    #"ksp_monitor_true_residual": None,
    #"ksp_converged_reason": None,
}