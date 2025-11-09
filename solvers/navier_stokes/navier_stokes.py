from firedrake import *

from solvers.timestepper import timestepper
from solvers.timestepper_adaptive import timestepper_adaptive

# constants
T = 2                   # final time
dt = 0.1                # timestepping length
theta = 1/2             # theta constant
tol = 0.001             # tolerance
N = 64                  # mesh size
Re = Constant(100.0)    # Reynold's num for viscosity

# mesh
mesh = UnitSquareMesh(N, N)

# declare function space
V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)
Z = V * W

up = Function(Z)
u, p = split(up)

x, y = SpatialCoordinate(mesh)

# functions
ufl_f = 0           # source term f
ufl_g = 0           # bdy condition g
ufl_u0 = 0          # initial condition u0

f = Function(V)
g = Function(V)
u0 = Function(V)

u0.interpolate(ufl_u0)

def make_weak_form(theta, idt, f_n, f_np1, g_n, g_np1, dsN):
    """
    Returns func F(u, u_old, p, q, v), 
    which builds weak form
    using external coefficients
    """

    def F(u, u_old, p, q, v):
        u_mid = theta * u + (1 - theta) * u_old

        return (
            idt * (u - u_old) * v * dx
            + (1.0 / Re) * inner(grad(u_mid), grad(v)) * dx
            + inner(dot(grad(u_mid), u_mid), v) * dx
            - (theta * f_np1 + (1 - theta) * f_n) * v * dx
            - p * div(v) * dx
            + div(u_mid) * q * dx
        )

    return F

# make data for iterative time stepping
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

# run
timestepper(V, ds(1), theta, T, dt, u0, get_data, make_weak_form)
timestepper_adaptive(V, ds(1), theta, T, tol, u0, get_data, make_weak_form)