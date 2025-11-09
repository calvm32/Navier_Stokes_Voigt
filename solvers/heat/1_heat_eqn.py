from firedrake import *

from timestepper import timestepper
from timestepper_adaptive import timestepper_adaptive

# mesh
mesh = UnitSquareMesh(10, 10)

# constants
T = 2           # final time
dt = 0.1        # timestepping length
theta = 1/2     # theta constant

# declare function space and interpolate functions
V = FunctionSpace(mesh, "CG", 1)
x, y = SpatialCoordinate(mesh)

# functions
ufl_f = cos(x*pi)*cos(y*pi)     # source term f
ufl_g = 0                       # bdy condition g
ufl_u0 = 0                      # initial condition u0

f = Function(V)
g = Function(V)
u0 = Function(V)

u0.interpolate(ufl_u0)

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


timestepper(V, ds(1), theta, T, dt, u0, get_data)
# timestepper_adaptive(V, ds_left, theta, T, tol, u0, get_data)