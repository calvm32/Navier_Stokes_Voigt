from firedrake import *
from solvers_2d.timestepper import timestepper

# constants
T = 2           # final time
dt = 0.1        # timestepping length
theta = 1/2     # theta constant

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

for exp in range(1, 10):
    N = 2**exp

    # mesh
    mesh = UnitSquareMesh(N, N)

    # declare function space and interpolate functions
    V = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    t = Constant(0.0) # symbolic constant for t

    # exact calculations for u=e^t*sin(pix)*cos(piy)
    ufl_u_exact = exp(t)*sin(pi*x)*cos(pi*y)
    ufl_g_exact = -pi*exp(t)*cos(pi*y)*sin(pi*y)
    ufl_f_exact = (1+2*pi**2)*exp(t)*sin(pi*x)*cos(pi*y)

    # functions
    ufl_f = ufl_f_exact     # source term f
    ufl_g = ufl_g_exact     # bdy condition g

    f = Function(V)
    g = Function(V)
    u_exact = Function(V)

    u_exact.interpolate(ufl_u_exact)

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
    timestepper(V, ds(1), theta, T, dt, u_exact, get_data, make_weak_form, u_exact)