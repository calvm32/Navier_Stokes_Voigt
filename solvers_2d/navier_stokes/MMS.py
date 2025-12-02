from firedrake import *
import matplotlib.pyplot as plt
from solvers_2d.timestepper_MMS import timestepper_MMS
from .make_weak_form import make_weak_form
from .config import T, dt, theta, Re, P, H

N_list = []
error_list = []

solver_parameters = {
    "mat_type": "matfree",
    "snes_monitor": None,
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
    "snes_converged_reason": None,
    #"ksp_monitor_true_residual": None,
    #"ksp_converged_reason": None,
}

# calculate error as mesh size increases
for exp in range(5, 6):
    N = 2**exp
    N_list.append(N)

    print("starting N = {:0d}".format(N))

    # mesh
    mesh = RectangleMesh(3*N, N, 3*H, H) # rectangle btwn (0,0) and (3H, H)
    x, y = SpatialCoordinate(mesh)

    t = Constant(0.0)
    ufl_exp = ufl.exp

    # exact functions for Poiseuille flow  
    ufl_v_exact = as_vector([                                   # velocity ic
        Re*( sin(pi*y/H)*ufl_exp(((pi**2)*t)/(H**2)) + 0.5*P*y**2 + 0.5*P*H*y ), 
        Constant(0.0)
    ])
    ufl_p_exact = P                                             # pressure ic
    ufl_f_exact = as_vector([Constant(0.0), Constant(0.0)])     # source term f
    ufl_g_exact = as_vector([Constant(0.0), Constant(0.0)])     # bdy condition g

    # declare function space and interpolate functions
    V = VectorFunctionSpace(mesh, "CG", 2)
    W = FunctionSpace(mesh, "CG", 1)
    Z = V * W

    appctx = {"Re": Re, "velocity_space": 0}

    u_exact = Function(Z)
    f = Function(V)
    g = Function(V)
    u0 = Function(Z)

    u_exact.subfunctions[0].interpolate(ufl_v_exact)
    u_exact.subfunctions[1].interpolate(ufl_p_exact)
    u0.subfunctions[0].interpolate(ufl_v_exact)
    u0.subfunctions[1].interpolate(ufl_p_exact)
    
    # BCs
    bc_noslip = DirichletBC(Z.sub(0), Constant((0.0, 0.0)), (1, 3))
    bc_pressure_ref = DirichletBC(Z.sub(1), Constant(0.0), (2,))  # pin pressure at boundary id 2
    bcs = [bc_noslip, bc_pressure_ref]

    nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

    # run
    error = timestepper_MMS(V, f, g, ds(1), theta, T, dt, u0, make_weak_form, u_exact,
            bcs=bcs, nullspace=nullspace, solver_parameters=solver_parameters, 
            appctx=appctx, W=W)
    error_list.append(error)

plt.loglog(N_list, error_list, "-o")
plt.xlabel("mesh size h")
plt.ylabel("error")
plt.grid(True)

plt.savefig("convergence_plot.png", dpi=200)