from firedrake import *

import matplotlib.pyplot as plt
from solvers_2d.timestepper import timestepper
from .make_weak_form import make_weak_form
from solvers_2d.printoff import blue

from .config_constants import t0, T, dt, theta, N_list, vtkfile_name

error_list = []

# calculate error as mesh size increases
for N in N_list:

    blue(f"\n*** Mesh size N = {N:0d} ***\n", spaced=True)

    # mesh
    mesh = UnitSquareMesh(N, N)
    x, y = SpatialCoordinate(mesh)
    ds = Measure("ds", domain=mesh)

    # declare function space and interpolate functions
    V = FunctionSpace(mesh, "CG", 1)

    # time dependant
    def get_data(t):

        # exact functions for u=e^t*sin(pix)*cos(piy)
        ufl_u_exact = ufl.exp(t)*cos(pi*x)*cos(pi*y)                # initial condition u0 
        ufl_f_exact = (1+2*pi**2)*ufl.exp(t)*cos(pi*x)*cos(pi*y)    # source term f 
        ufl_g_exact = Constant(0)                                   # bdy condition g

        # returns
        return {"ufl_u0": ufl_u_exact,
                "ufl_f": ufl_f_exact,
                "ufl_g": ufl_g_exact}
    
    vtkfile_name = f"{vtkfile_name}_N{N}.pvd"

    # run
    error = timestepper(get_data, theta, V, ds, t0, T, dt, make_weak_form, vtkfile_name=vtkfile_name)
    error_list.append(error)

plt.loglog(N_list, error_list, "-o")
plt.xlabel("mesh size h")
plt.ylabel("error")
plt.grid(True)

plt.savefig("convergence_plot.png", dpi=200)

