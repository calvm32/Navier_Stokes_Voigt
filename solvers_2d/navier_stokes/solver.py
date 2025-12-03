from firedrake import *
from solvers_2d.timestepper import timestepper
from .make_weak_form import make_weak_form
from solvers_2d.printoff import blue
from .config import t0, T, dt, theta, N, solver_parameters

blue(f"\n*** Starting solve ***\n", spaced=True)

# mesh
mesh = UnitSquareMesh(N, N)
x, y = SpatialCoordinate(mesh)

t = Constant(0.0) # symbolic constant for t
ufl_exp = ufl.exp # ufl e, so t gets calculated correctly 

# functions
ufl_v0 = as_vector([1+sin(pi*x), cos(pi*y)])  # velocity ic
ufl_p0 = Constant(5.0)                      # pressure ic
ufl_f = as_vector([2, 2])                   # source term f
ufl_g = as_vector([2, 2])                   # bdy condition g

# declare function space and interpolate functions
V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)
Z = V * W

function_space_appctx = {
    "velocity_space": V,
    "pressure_space": W,
    "ufl_v0": ufl_v0,
    "ufl_p0": ufl_p0,
    "ufl_f": ufl_f,
    "ufl_g": ufl_g,
    }

# initial condition
z0 = Function(Z)
z0.sub(0).interpolate(as_vector([1 + sin(pi*x), cos(pi*y)]))
z0.sub(1).interpolate(Constant(5.0))

# BCs
bcs = [
    DirichletBC(Z.sub(0), Constant((1, 0)), lambda x, on_b: on_b and near(x[1], 1.0)),  # top lid
    DirichletBC(Z.sub(0), Constant((0, 0)), lambda x, on_b: on_b and x[1] < 1.0),      # walls
]

nullspace = MixedVectorSpaceBasis(
    Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

# run
timestepper(theta, Z, ds, t0, T, dt, make_weak_form, function_space_appctx,
        bcs=bcs, nullspace=nullspace, solver_parameters=solver_parameters)