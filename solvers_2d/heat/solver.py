from firedrake import *
from solvers_2d.timestepper import timestepper
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

def make_weak_form(theta, idt, f_n, f_np1, g_n, g_np1, dsN):
    """
    Returns func F(u, u_old, v), which builds weak form
    using external coefficients
    """

    def F(u, u_old, v, *args):
        return (
            idt * (u - u_old) * v * dx
            + inner(grad(theta * u + (1 - theta) * u_old), grad(v)) * dx
            - (theta * f_np1 + (1 - theta) * f_n) * v * dx
            - (theta * g_np1 + (1 - theta) * g_n) * v * dsN
        )

    return F

# run
timestepper(V, ds(1), theta, T, dt, u0, get_data, make_weak_form)