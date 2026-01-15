from firedrake import *

import matplotlib.pyplot as plt
from solvers.timestepper import timestepper
from .make_weak_form import make_weak_form
from solvers.printoff import blue, green

from .config_constants import solver_parameters, vtkfile_name

# ---------
# Constants
# ---------

t0 = 0.0        # initial time
T = 1.0         # final time
dt = 1e-2       # timestepping length
theta = 1/2     # theta constant

# MMS loops over mesh resolutions in this list
N_list = []
for n in range(1, 6):
    N = 2**n
    N_list.append(N)

# calculate error as mesh size increases
final_error_list = [] 

# -------------
# Start solving
# -------------

for N in N_list:

    blue(f"\n*** Mesh size N = {N:0d} ***\n", spaced=True) # report mesh size
    new_vtkfile_name = f"{vtkfile_name}_N{N}" # write to new file

    # ------------
    # Setup spaces
    # ------------

    # mesh and measures
    mesh = UnitCubeMesh(N, N, N)
    x, y, z = SpatialCoordinate(mesh)

    dx = Measure("dx", domain=mesh)
    ds = Measure("ds", domain=mesh)

    # declare function space and interpolate functions
    V = FunctionSpace(mesh, "CG", 1)

    # ------------------
    # Allocate functions
    # ------------------

    # time dependant
    def get_data(t):

        # exact functions for u=e^t*sin(pix)*cos(piy)*cos(pi*z)  
        ufl_u0 = ufl.exp(t)*cos(pi*x)*cos(pi*y)*cos(pi*z)                  # initial condition u0 
        ufl_f0 = (1+2*pi**2)*ufl.exp(t)*cos(pi*x)*cos(pi*y)*cos(pi*z)      # source term f 
        ufl_g0 = Constant(0)                                               # bdy condition g

        # returns
        return {"ufl_u0": ufl_u0,
                "ufl_f": ufl_f0,
                "ufl_g": ufl_g0}

    # ----------
    # Run solver
    # ----------

    u_error_list, palenstrophy_list, stream_func_list, vorticity_list, enstrophy_list = timestepper(get_data, theta, 
            V, dx, ds, 
            t0, T, dt,
            make_weak_form=make_weak_form,
            solver_parameters=solver_parameters,
            vtkfile_name=new_vtkfile_name)
    
    green(f"Final L2 Error (temperature) = {u_error:0.8e}", spaced=True)

    final_error = 0
    for err in u_error_list:
        final_error += err
    
    final_error_list.append(sqrt(final_error))

# ------------------------
# Plot error vs. mesh size
# ------------------------

plt.loglog(N_list, error_list, "-o")
plt.xlabel("mesh size")
plt.ylabel("error")
plt.grid(True)

plt.savefig("convergence_plot.png", dpi=200)