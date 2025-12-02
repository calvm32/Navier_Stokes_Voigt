from firedrake import *
from solvers_2d.timestepper import timestepper
from .make_weak_form import make_weak_form
from .config import t0, T, dt, theta, N

# mesh
mesh = UnitSquareMesh(N, N)
x, y = SpatialCoordinate(mesh)

# functions
ufl_f = cos(x*pi)*cos(y*pi)     # source term f
ufl_g = 0                       # bdy condition g
ufl_u0 = 0                      # initial condition u0

# declare function space and interpolate functions
V = FunctionSpace(mesh, "CG", 1)

f = Function(V)
g = Function(V)
u0 = Function(V)

u0.interpolate(ufl_u0)

function_appctx = {
    "ufl_f": ufl_f,
    "ufl_g": ufl_g,
    "ufl_u0": ufl_u0
    }

# run
timestepper(theta, V, ds(1), t0, T, dt, make_weak_form, function_appctx)