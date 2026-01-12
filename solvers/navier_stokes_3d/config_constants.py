from firedrake import *

# ---------
# Constants
# ---------

t0 = 0.0                    # initial time
T = 1.0                     # final time
dt = 1e-2                   # timestepping length
theta = 0.5                 # theta constant
Re = 1.0                    # Reynold's num = 1 / viscosity
gamma_gd = 0.1              # grad-div stabilization constant

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
N_list = [64]
"""for exp in range(3, 9):
    N = 2**exp
    N_list.append(N)"""

G = 5.0     # initial pressure gradient
P = 1.0     # pressure strength (P*x + G)

# ------------
# Dictionaries
# ------------

appctx = {
    "Re": Re, 
    "gamma_gd": gamma_gd,
    "velocity_space": 0
}

solver_parameters = {
    "mat_type": "matfree",
    "snes_monitor": None,

    # We'll use a non-stationary Krylov solve for the Schur complement, so
    # we need to use a flexible Krylov method on the outside.

    "ksp_type": "fgmres",
    #"ksp_gmres_modifiedgramschmidt": None,
    #"ksp_monitor_true_residual": None,

    # Now to configure the preconditioner::

    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "lower",

    # invert the velocity block with LU::

    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "python",
    "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
    "fieldsplit_0_assembled_pc_type": "lu",

    # invert the schur complement inexactly using GMRES, preconditioned w PCD

    "fieldsplit_1_ksp_type": "gmres",
    "fieldsplit_1_ksp_rtol": 1e-4,
    "fieldsplit_1_pc_type": "python",
    "fieldsplit_1_pc_python_type": "firedrake.PCDPC",

    # We now need to configure the mass and stiffness solvers in the PCD
    # preconditioner.  For this example, we will just invert them with LU,
    # although of course we can use a scalable method if we wish. First the
    # mass solve

    "fieldsplit_1_pcd_Mp_ksp_type": "preonly",
    "fieldsplit_1_pcd_Mp_pc_type": "lu",

    # and the stiffness solve

    "fieldsplit_1_pcd_Kp_ksp_type": "preonly",
    "fieldsplit_1_pcd_Kp_pc_type": "lu",

    # Finally, we just need to decide whether to apply the action of the
    # pressure-space convection-diffusion operator with an assembled matrix
    # or matrix free.  Here we will use matrix-free::

    "fieldsplit_1_pcd_Fp_mat_type": "matfree",

    # finally make the pcd look at the velocity function for more refined meshes

    #"fieldsplit_1_pcd_u": "velocity",
    }