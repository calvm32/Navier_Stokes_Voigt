from firedrake import *
import matplotlib.pyplot as plt

from timestepper import timestepper
from timestepper_adaptive import timestepper_adaptive
from create_surface_measure import create_surface_measure_left

# mesh
mesh = UnitSquareMesh(10, 10)
ds_left = create_surface_measure_left(mesh)

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

def get_data_4(t, result=None):
    """Create or update data for Task 4"""
    f, g = result or (Constant(0), Constant(0))
    f.assign(ufl_f)
    g.assign(ufl_g)
    return f, g

timestepper(V, ds_left, theta, T, dt, u0, get_data_4)
# timestepper_adaptive(V, ds_left, theta, T, tol, u0, get_data_4)

# Hold plots before quitting
plt.show()