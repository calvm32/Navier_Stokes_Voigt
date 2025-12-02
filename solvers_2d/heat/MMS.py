from firedrake import *
import matplotlib.pyplot as plt
from solvers_2d.timestepper_MMS import timestepper_MMS
from .make_weak_form import make_weak_form
from solvers_2d.printoff import blue
from .config import t, T, dt, theta, N_list

error_list = []

# calculate error as mesh size increases
for N in N_list:

    blue(f"\n*** Mesh size N = {N:0d} ***\n", spaced=True)

    # mesh
    mesh = UnitSquareMesh(N, N)
    x, y = SpatialCoordinate(mesh)

    t = Constant(0.0) # symbolic constant for t
    ufl_exp = ufl.exp # ufl e, so t gets calculated correctly

    # exact functions for u=e^t*sin(pix)*cos(piy)
    ufl_u_exact = ufl_exp(t)*cos(pi*x)*cos(pi*y)                # source term f
    ufl_f_exact = (1+2*pi**2)*ufl_exp(t)*cos(pi*x)*cos(pi*y)    # bdy condition g
    ufl_g_exact = 0                                             # initial condition u0

    # declare function space and interpolate functions
    V = FunctionSpace(mesh, "CG", 1)

    u_exact = Function(V)
    f = Function(V)
    g = Function(V)
    u0 = Function(V)

    u_exact.interpolate(ufl_u_exact)
    u0.interpolate(ufl_u_exact)

    # run
    error = timestepper_MMS(V, f, g, ds(1), theta, t, T, dt, u0, make_weak_form, u_exact, N)
    error_list.append(error)

plt.loglog(N_list, error_list, "-o")
plt.xlabel("mesh size h")
plt.ylabel("error")
plt.grid(True)

plt.savefig("convergence_plot.png", dpi=200)