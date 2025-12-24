from firedrake import *

from solvers_2d.timestepper_CN import timestepper_CN
from solvers_2d.timestepper_RK4 import timestepper_RK4
from .make_weak_form import make_weak_form
from solvers_2d.printoff import blue

from .config_constants import t0, T, dt, theta, N, vtkfile_name, solve_type

blue(f"\n*** Starting solve ***\n", spaced=True)

# ------------
# Setup spaces
# ------------

# mesh and measures
mesh = UnitSquareMesh(N, N)
x, y = SpatialCoordinate(mesh)

dx = Measure("dx", domain=mesh)
ds = Measure("ds", domain=mesh)

# declare function space and interpolate functions
V = FunctionSpace(mesh, "CG", 1)

# ------------------
# Allocate functions
# ------------------

def get_data(t):

    # functions
    ufl_u0 = ufl.exp(t)*cos(pi*x)   # initial condition u0
    ufl_f = cos(x*pi)*cos(y*pi)     # source term f
    ufl_g = Constant(0.0)           # bdy condition g

    # returns
    return {"ufl_u0": ufl_u0,
            "ufl_f": ufl_f,
            "ufl_g": ufl_g}

# ----------
# Run solver
# ----------

error = timestepper_CN(
    get_data, 
    theta, 
    V, dx, ds, 
    t0, T, dt, 
    make_weak_form, 
    vtkfile_name=vtkfile_name)