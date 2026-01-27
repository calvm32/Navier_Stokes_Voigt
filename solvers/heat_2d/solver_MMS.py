from firedrake import *
import yaml
from pathlib import Path
import os
import shutil

import matplotlib.pyplot as plt
from solvers.timestepper import timestepper
from .make_weak_form import make_weak_form
from solvers.printoff import blue, green
from solvers.config_setup import *

# -----------------
# MMS Configuration
# -----------------

CFG_PATH1 = Path(__file__).parent / "configs" / "MMS_constants.yaml"
cfg = load_config(CFG_PATH1)

t0 = cfg["t0"]
T = cfg["T"]
dt = cfg["dt"]
theta = cfg["theta"]

CFG_PATH2 = Path(__file__).parent / "configs" / "MMS_solver_params.yaml"
solver_parameters = load_solver_parameters(CFG_PATH2, dt=dt)

vtkfile_name = "Soln"

# -------------
# Archive YAMLs
# -------------

# current working directory
run_dir = Path(os.getcwd())

# copy YAML files to current directory
shutil.copy(CFG_PATH1, run_dir / CFG_PATH1.name)
shutil.copy(CFG_PATH1, run_dir / CFG_PATH1.name)

print(f"[solver.py] YAML configs archived in {run_dir}")

# -------------
# Start solving
# -------------

# MMS loops over mesh resolutions in this list
N_list = []
for n in range(1, 6):
    N = 2**n
    N_list.append(N)

# calculate error as mesh size increases
final_error_list = [] 

for N in N_list:

    blue(f"\n*** Mesh size N = {N:0d} ***\n", spaced=True) # report mesh size
    new_vtkfile_name = f"{vtkfile_name}_N{N}" # write to new file

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

    v_error_list, p_error_list, palinstrophy_list, stream_func_list, vorticity_list, enstrophy_list, every_time_list, energy_list, all_time_list = timestepper(get_data, 
            V, dx, ds, 
            t0, T, dt,
            make_weak_form=make_weak_form,
            solver_parameters=solver_parameters,
            vtkfile_name=new_vtkfile_name)

    final_error = 0
    for err in u_error_list:
        final_error += err
    
    final_error_list.append(sqrt(final_error))
    
    green(f"Final L2 Error (temperature) = {final_error:0.8e}", spaced=True)

# ------------------------
# Plot error vs. mesh size
# ------------------------

plt.loglog(N_list, final_error_list, "-o")
plt.xlabel("mesh size")
plt.ylabel("error")
plt.grid(True)

plt.savefig("convergence_plot.png", dpi=200)