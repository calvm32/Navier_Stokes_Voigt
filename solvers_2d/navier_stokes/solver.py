from firedrake import *
from solvers_2d.timestepper import timestepper
from .make_weak_form import make_weak_form
from .config import T, dt, theta, N, Re

appctx = {"Re": Re, "velocity_space": 0}

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

# mesh
mesh = UnitSquareMesh(N, N)
x, y = SpatialCoordinate(mesh)

# functions
ufl_v = as_vector([1, 0])           # velocity ic
ufl_p = Constant(0.0)               # pressure ic
ufl_f = as_vector([0, 0])           # source term f
ufl_g = as_vector([0, 0])           # bdy condition g

# declare function space and interpolate functions
V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)
Z = V * W

f = Function(V)
g = Function(V)
u0 = Function(Z)

u0.subfunctions[0].interpolate(ufl_v)
u0.subfunctions[1].interpolate(ufl_p)

# setup from demo
bcs = [DirichletBC(Z.sub(0), Constant((1, 0)), (4,)),
       DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3))]

nullspace = MixedVectorSpaceBasis(
    Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

# run
timestepper(V, f, g, ds(1), theta, T, dt, u0, make_weak_form,
        bcs=bcs, nullspace=nullspace, solver_parameters=solver_parameters, 
        appctx=appctx, W=W)