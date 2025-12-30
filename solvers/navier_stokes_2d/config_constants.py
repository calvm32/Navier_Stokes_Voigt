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

appctx = {
    "Re": Re, 
    "gamma_gd": gamma_gd,
    "velocity_space": 0
}

solver_parameters = {
    # --- Monolithic matrix-free operator for the whole system ---
    "mat_type": "matfree",  # don't assemble the global matrix; use matrix-free kernels

    # --- Outer Krylov solver ---
    "ksp_type": "pipefgmres",  # pipelined flexible GMRES: reduces global reductions, good for clusters
    "ksp_rtol": 1e-5,          # relative tolerance for the outer KSP solver
    "ksp_max_it": 100,         # max iterations to prevent runaway solves

    # --- FieldSplit: Schur complement setup ---
    "pc_type": "fieldsplit",           # treat (velocity, pressure) separately
    "pc_fieldsplit_type": "schur",     # Schur complement factorization
    "pc_fieldsplit_schur_fact_type": "upper",  # apply velocity solve first, then Schur (better for PCD)

    # --- Velocity block ---
    "fieldsplit_0_ksp_type": "preonly",               # apply the preconditioner directly (LU), no Krylov iterations
    "fieldsplit_0_pc_type": "python",                # use a Python-defined PC
    "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",  # assemble velocity block and invert with LU
    "fieldsplit_0_assembled_pc_type": "lu",          # LU for the velocity block
    # --- Reuse factorization across timesteps ---
    "fieldsplit_0_pc_factor_reuse_ordering": True,   # reuse symbolic ordering
    "fieldsplit_0_pc_factor_reuse_fill": True,       # reuse sparsity pattern for faster refactorization
    "fieldsplit_0_pc_factor_shift_type": "NONZERO",  # ensures robustness for near-singular matrices

    # --- Pressure block (PCD) ---
    "fieldsplit_1_ksp_type": "fgmres",  # flexible GMRES for Schur complement solve
    "fieldsplit_1_ksp_rtol": 1e-2,      # loose tolerance, since PCD is approximate
    "fieldsplit_1_ksp_max_it": 10,      # only a few iterations needed per timestep

    "fieldsplit_1_pc_type": "python",                 # Python PC (PCD)
    "fieldsplit_1_pc_python_type": "firedrake.PCDPC",# PCD preconditioner for pressure

    # PCD mass matrix solve
    "fieldsplit_1_pcd_Mp_ksp_type": "preonly",  
    "fieldsplit_1_pcd_Mp_pc_type": "lu",

    # PCD stiffness matrix solve
    "fieldsplit_1_pcd_Kp_ksp_type": "preonly",  
    "fieldsplit_1_pcd_Kp_pc_type": "lu",

    # Convection-diffusion operator for pressure, matrix-free
    "fieldsplit_1_pcd_Fp_mat_type": "matfree",

    # --- Physics parameters for Python PCs ---
    "gamma_gd": gamma_gd,  # grad-div stabilization (used inside PCD and velocity block)
}
