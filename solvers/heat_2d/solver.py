from firedrake import *

from solvers.timestepper import timestepper
from .make_weak_form import make_weak_form
from solvers.printoff import blue

from .config_constants import t0, T, dt, theta, N, vtkfile_name

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

# time dependant
def get_data(t):

    # exact functions for u=e^t*sin(pix)*cos(piy)
    ufl_u0 = ufl.exp(t)*cos(pi*x)*cos(pi*y)                # initial condition u0 
    ufl_f0 = (1+2*pi**2)*ufl.exp(t)*cos(pi*x)*cos(pi*y)    # source term f 
    ufl_g0 = Constant(0)                                   # bdy condition g

    # returns
    return {"ufl_u0": ufl_u0,
            "ufl_f": ufl_f0,
            "ufl_g": ufl_g0}

# ----------
# Run solver
# ----------

error = timestepper(
    get_data, 
    theta, 
    V, dx, ds, 
    t0, T, dt, 
    make_weak_form, 
    vtkfile_name=vtkfile_name)