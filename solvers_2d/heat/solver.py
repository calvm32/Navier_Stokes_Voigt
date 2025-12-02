from firedrake import *
from solvers_2d.timestepper import timestepper
from .make_weak_form import make_weak_form
from .config import t0, T, dt, theta, N


blue(f"\n*** Starting solve ***\n", spaced=True)

# mesh
mesh = UnitSquareMesh(N, N)
x, y = SpatialCoordinate(mesh)

t = Constant(0.0) # symbolic constant for t
ufl_exp = ufl.exp # ufl e, so t gets calculated correctly 

# functions
ufl_u0 = ufl_exp(t)*cos(pi*x)   # initial condition u0
ufl_f = cos(x*pi)*cos(y*pi)     # source term f
ufl_g = Constant(0)             # bdy condition g

# declare function space and interpolate functions
V = FunctionSpace(mesh, "CG", 1)

function_appctx = {
    "ufl_u0": ufl_u0,
    "ufl_f": ufl_f,
    "ufl_g": ufl_g
    }

# run
timestepper(theta, V, ds(1), t0, T, dt, make_weak_form, function_appctx)