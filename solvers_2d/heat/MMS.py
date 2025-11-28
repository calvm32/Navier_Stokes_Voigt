from firedrake import *
import matplotlib.pyplot as plt
from solvers_2d.timestepper_MMS import timestepper_MMS
from .make_weak_form import make_weak_form
from .config import T, dt, theta

N_list = []
error_list = []

# calculate error as mesh size increases
for exp in range(1, 10):
    N = 2**exp
    N_list.append(N)

    # mesh
    mesh = UnitSquareMesh(N, N)
    x, y = SpatialCoordinate(mesh)

    t = Constant(0.0) # symbolic constant for t
    ufl_exp = ufl.exp # ufl e, so t gets calculated correctly

    # exact calculations for u=e^t*sin(pix)*cos(piy)
    ufl_u_exact = ufl_exp(t)*cos(pi*x)*cos(pi*y)
    ufl_f_exact = (1+2*pi**2)*ufl_exp(t)*cos(pi*x)*cos(pi*y)
    ufl_g_exact = 0

    # functions
    ufl_f = ufl_f_exact     # source term f
    ufl_g = ufl_g_exact     # bdy condition g

    # declare function space and interpolate functions
    V = FunctionSpace(mesh, "CG", 1)

    u_exact = Function(V)
    f = Function(V)
    g = Function(V)
    u0 = Function(V)

    u_exact.interpolate(ufl_u_exact)
    u0.interpolate(ufl_u_exact)

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
    error = timestepper_MMS(V, ds(1), theta, T, dt, u0, get_data, make_weak_form, u_exact)
    error_list.append(error)

plt.loglog(N_list, error_list, "-o")
plt.xlabel("mesh size h")
plt.ylabel("error")
plt.grid(True)

plt.savefig("convergence_plot.png", dpi=200)