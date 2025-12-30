from firedrake import *

# ---------
# Constants
# ---------

t0 = 0.0                    # initial time
T = 1.0                     # final time
dt = 0.01                   # timestepping length
theta = 1                   # theta constant
Re = Constant(100)          # Reynold's num = 1 / viscosity
gamma_gd = Constant(0.0)    # grad-div stabilization constant

H = 1.0                     # height of box (if changed, need to adjust mesh)
L = 3.0                     # length of box (can be changed always)

vtkfile_name = "Soln"

# ----------------
# For single solve 
# ----------------

N = 16 # mesh resolution

# -------------
# For MMS solve
# -------------

# MMS loops over mesh resolutions in this list
N_list = []
for exp in range(4, 10):
    N = 2**exp
    N_list.append(N)

G = 5.0                 # initial pressure gradient
P = 1.0                 # pressure strength (P*x + G)

# ------------
# Dictionaries
# ------------

appctx = {"Re": Re, "velocity_space": 0}

solver_parameters = {
    "mat_type": "matfree",

    "ksp_type": "fgmres",
    "ksp_rtol": 1e-6,

    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "upper",

    # Velocity block
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "python",
    "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
    "fieldsplit_0_assembled_pc_type": "lu",

    # Pressure block (PCD)
    "fieldsplit_1_ksp_type": "fgmres",
    "fieldsplit_1_ksp_rtol": 1e-2,
    "fieldsplit_1_ksp_max_it": 20,

    "fieldsplit_1_pc_type": "python",
    "fieldsplit_1_pc_python_type": "firedrake.PCDPC",

    "fieldsplit_1_pcd_Mp_ksp_type": "preonly",
    "fieldsplit_1_pcd_Mp_pc_type": "lu",

    "fieldsplit_1_pcd_Kp_ksp_type": "preonly",
    "fieldsplit_1_pcd_Kp_pc_type": "lu",

    "fieldsplit_1_pcd_Fp_mat_type": "matfree",

    "gamma_gd": gamma_gd,
}
