from firedrake import *
from solvers_2d.timestepper import timestepper
from .make_weak_form import make_weak_form
from solvers_2d.printoff import blue
from .constant_config import t0, T, dt, theta, N

blue(f"\n*** Starting solve ***\n", spaced=True)

# mesh
mesh = UnitSquareMesh(N, N)
x, y = SpatialCoordinate(mesh)
ds = Measure("ds", domain=mesh)

# declare function space and interpolate functions
V = FunctionSpace(mesh, "CG", 1)

# time dependant
def get_data(t):

    # functions
    ufl_u0 = ufl.exp(t)*cos(pi*x)   # initial condition u0
    ufl_f = cos(x*pi)*cos(y*pi)     # source term f
    ufl_g = Constant(0)             # bdy condition g

    # returns
    return {"u0": ufl_u0,
            "f": ufl_f,
            "g": ufl_g}

# run
timestepper(get_data, theta, V, ds, t0, T, dt, make_weak_form)