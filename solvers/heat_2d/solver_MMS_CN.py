from firedrake import *
import yaml
from pathlib import Path
import os
import shutil

import matplotlib.pyplot as plt
from solvers.timesteppers import timestepper_CN
from .make_weak_form import make_weak_form_CN
from solvers.printoff import blue, green
from solvers.config_setup import *

# -------------------
# Get + archive paths
# -------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # adjust if this script moves
TEMPLATES_DIR = PROJECT_ROOT / "templates"

CFG_PATH1 = TEMPLATES_DIR / "settings" / "heat_MMS.yaml"
CFG_PATH2 = TEMPLATES_DIR / "solver_parameters" / "heat_MMS.yaml"
CFG_PATH3 = TEMPLATES_DIR / "ufl_expr" / "heat_MMS.yaml"

# current working directory
run_dir = Path(os.getcwd())

# copy YAML files to current directory
shutil.copy(CFG_PATH1, run_dir / CFG_PATH1.name)
shutil.copy(CFG_PATH2, run_dir / CFG_PATH2.name)
shutil.copy(CFG_PATH3, run_dir / CFG_PATH3.name)

print(f"[solver.py] YAML configs archived in {run_dir}\n")

# -----------------
# MMS Configuration
# -----------------

cfg = load_config(CFG_PATH1)

# extract settings
t0 = cfg["t0"]
T = cfg["T"]
dt = cfg["dt"]
theta = cfg["theta"]

solver_parameters = load_solver_parameters(CFG_PATH2)
print(solver_parameters)

vtkfile_name = "Soln"

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

    # -------------------
    # Configure functions
    # -------------------

    namespace = {
        "as_vector": as_vector,
        "Constant": Constant,
        "x": x,
        "y": y,
        "pi": pi,
        "sin": sin,
        "cos": cos,
        "exp": exp,
    }

    ufl_cfg = load_ufl_expressions(CFG_PATH3, namespace=namespace)

    # ------------------
    # Allocate functions
    # ------------------

    # time dependant
    def get_data(t):

        namespace.update({
            "t": t,
        })

        return {
            "ufl_v0": ufl_cfg["ufl_v0"],
            "ufl_p0": ufl_cfg["ufl_p0"],
            "ufl_f": ufl_cfg["ufl_f"],
            "ufl_g": ufl_cfg["ufl_g"]
        }

    # ----------
    # Run solver
    # ----------

    u_error_list, energy_list, all_time_list = timestepper_CN(get_data, 
            V, dx, ds, 
            t0, T, dt,
            make_weak_form=make_weak_form_CN,
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

plt.semilogy(N_list, final_error_list, "-o")
plt.xlabel("mesh size")
plt.ylabel("error")
plt.grid(True)

plt.savefig("convergence_plot.png", dpi=200)