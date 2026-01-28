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

# -------------
# Configuration
# -------------

CFG_PATH1 = Path(__file__).parent / "configs" / "USER_constants.yaml"
cfg = load_config(CFG_PATH1)

# Extract constants
t0 = cfg["t0"]
T = cfg["T"]
dt = cfg["dt"]
theta = cfg["theta"]
N = cfg["N"]

CFG_PATH2 = Path(__file__).parent / "configs" / "MMS_solver_params.yaml"
solver_parameters = load_solver_parameters(CFG_PATH2)

vtkfile_name = "Soln"

# -------------
# Archive YAMLs
# -------------

# current working directory
run_dir = Path(os.getcwd())

# copy YAML files to current directory
shutil.copy(CFG_PATH1, run_dir / CFG_PATH1.name)
shutil.copy(CFG_PATH2, run_dir / CFG_PATH2.name)

print(f"[solver.py] YAML configs archived in {run_dir}")

# ------------
# Setup spaces
# ------------

blue(f"\n*** Starting solve ***\n", spaced=True)

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

u_error_list, palinstrophy_list, stream_func_list, enstrophy_list, time_list = timestepper(get_data, 
        V, dx, ds, 
        t0, T, dt,
        make_weak_form=make_weak_form,
        solver_parameters=solver_parameters,
        vtkfile_name=vtkfile_name)

# -----------------
# Plot palinstrophy
# -----------------

plt.loglog(every_time_list, palinstrophy_list, "-o")
plt.xlabel("time")
plt.ylabel("palinstrophy L2")
plt.grid(True)
plt.tight_layout()
plt.savefig("0_palinstrophy_plot.png", dpi=200, bbox_inches='tight')
plt.close()

# --------------------
# Plot stream function
# --------------------

plt.loglog(every_time_list, stream_func_list, "-o")
plt.xlabel("time")
plt.ylabel("stream function L2")
plt.grid(True)
plt.tight_layout()
plt.savefig("0_stream_func_plot.png", dpi=200, bbox_inches='tight')
plt.close()

# --------------
# Plot Enstrophy
# --------------

plt.loglog(every_time_list, enstrophy_list, "-o")
plt.xlabel("time")
plt.ylabel("enstrophy L2")
plt.grid(True)
plt.tight_layout()
plt.savefig("0_enstrophy_plot.png", dpi=200, bbox_inches='tight')
plt.close()

# --------------
# Plot Enstrophy
# --------------

plt.loglog(all_time_list, energy_list, "-o")
plt.xlabel("time")
plt.ylabel("energy")
plt.grid(True)
plt.tight_layout()
plt.savefig("0_energy_plot.png", dpi=200, bbox_inches='tight')
plt.close()