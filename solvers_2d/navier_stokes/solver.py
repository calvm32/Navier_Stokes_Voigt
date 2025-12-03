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
ufl_v0 = as_vector([1+sin(pi*x), cos(pi*y)])    # velocity ic
ufl_p0 = Constant(5.0)                          # pressure ic
ufl_f = as_vector([2, 2])                       # source term f
ufl_g = as_vector([2, 2])                       # bdy condition g

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
tol = 1e-14

# top lid
def top(x, on_b):
    return on_b and near(x[1], 1.0, tol)

# bottom wall
def bottom(x, on_b):
    return on_b and near(x[1], 0.0, tol)

# left wall
def left(x, on_b):
    return on_b and near(x[0], 0.0, tol)

# right wall
def right(x, on_b):
    return on_b and near(x[0], 1.0, tol)

bcs = [
    DirichletBC(Z.sub(0), Constant((1,0)), top),
    DirichletBC(Z.sub(0), Constant((0,0)), bottom),
    DirichletBC(Z.sub(0), Constant((0,0)), left),
    DirichletBC(Z.sub(0), Constant((0,0)), right)
]

nullspace = MixedVectorSpaceBasis(
    Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

# run
timestepper(theta, Z, ds, t0, T, dt, make_weak_form, function_space_appctx,
        bcs=bcs, nullspace=nullspace, solver_parameters=solver_parameters)