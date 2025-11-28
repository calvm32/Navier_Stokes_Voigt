from firedrake import *
from solvers_2d.timestepper import timestepper
from .make_weak_form import make_weak_form

# constants
T = 2                   # final time
dt = 0.1                # timestepping length
theta = 1/2             # theta constant
N = 256                 # mesh size
Re = Constant(100.0)    # Reynold's num for viscosity

appctx = {"Re": Re, "velocity_space": 0}

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
    "fieldsplit_1_pcd_Fp_mat_type": "matfree"
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

# make data for iterative time stepping
def get_data(t, result=None):
    """Create or update data"""
    if result is None: # only allocate memory if hasn't been yet
        f = Function(V)
        g = Function(V)
    else:
        f, g = result

    f.interpolate(ufl_f)
    g.interpolate(ufl_g)
    return f, g

# setup from demo
bcs = [DirichletBC(Z.sub(0), Constant((1, 0)), (4,)),
       DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3))]

nullspace = MixedVectorSpaceBasis(
    Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

# run
timestepper(V, ds(1), theta, T, dt, u0, get_data, make_weak_form,
        bcs=bcs, nullspace=nullspace, solver_parameters=solver_parameters, 
        appctx=appctx, W=W)