from firedrake import *
from solvers_2d.timestepper import timestepper
from .make_weak_form import make_weak_form
from .config import T, dt, theta, N

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

# run
timestepper(V, ds(1), theta, T, dt, u0, make_weak_form)