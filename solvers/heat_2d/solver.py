from firedrake import *
import yaml
from pathlib import Path
import os
import shutil

import matplotlib.pyplot as plt
from solvers.timesteppers import *
from .make_weak_form import *
from solvers.printoff import blue, green
from solvers.config_setup import *

# -------------------
# Get + archive paths
# -------------------

CFG_PATH1 = Path(__file__).parent / "configs" / "settings.yaml"
CFG_PATH2 = Path(__file__).parent / "configs" / "solver_params.yaml"
CFG_PATH3 = Path(__file__).parent / "configs" / "ufl_expr.yaml"

# current working directory
run_dir = Path(os.getcwd())

# copy YAML files to current directory
shutil.copy(CFG_PATH1, run_dir / CFG_PATH1.name)
shutil.copy(CFG_PATH2, run_dir / CFG_PATH2.name)
shutil.copy(CFG_PATH3, run_dir / CFG_PATH3.name)

print(f"[solver.py] YAML configs archived in {run_dir}")

# ------------------
# Configure settings
# ------------------

cfg = load_config(CFG_PATH1)

# Extract settings
t0 = cfg["t0"]
T = cfg["T"]
dt = cfg["dt"]
theta = cfg["theta"]
N = cfg["N"]
solver = cfg["solver"]

vtkfile_name = "Soln"

solver_parameters = load_solver_parameters(CFG_PATH2)

# ------------
# Setup spaces
# ------------

blue(f"\n*** Starting solve ***\n", spaced=True)

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

if solver == "CN":
    u_error_list, energy_list, all_time_list = timestepper_CN(get_data, 
		V, dx, ds, 
		t0, T, dt,
		make_weak_form=make_weak_form_CN,
		solver_parameters=solver_parameters,
		vtkfile_name=vtkfile_name)
elif solver == "BDF2":
    u_error_list, energy_list, all_time_list = timestepper_BDF2(get_data, 
		V, dx, ds, 
		t0, T, dt,
		make_weak_form=make_weak_form_BDF2,
		solver_parameters=solver_parameters,
		vtkfile_name=vtkfile_name)

# -----------
# Plot Energy
# -----------

plt.semilogy(all_time_list, energy_list, "-o")
plt.xlabel("time")
plt.ylabel("energy")
plt.grid(True)
plt.tight_layout()
plt.savefig("0_energy_plot.png", dpi=200, bbox_inches='tight')
plt.close()