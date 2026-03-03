from firedrake import *
import yaml
from pathlib import Path
import os
import shutil

import matplotlib.pyplot as plt
from solvers.timesteppers import *
from .make_weak_form import *
from solvers.processing.printoff import blue, green
from solvers.processing.config_setup import *

# ---------
# Get paths
# ---------

def load_run_configs(save_dir):
    save_dir = Path(save_dir)
    cfg_path = save_dir / "settings.yaml"
    solver_params_path = save_dir / "solver_params.yaml"
    ufl_path = save_dir / "ufl_expr.yaml"

    cfg = load_config(cfg_path)
    solver_parameters = load_solver_parameters(solver_params_path)

    # optionally load UFL expressions
    namespace = {"Constant": Constant, "as_vector": as_vector}  # extend as needed
    ufl_cfg = load_ufl_expressions(ufl_path, namespace=namespace)

    return cfg, solver_parameters, ufl_cfg

# ------------------
# Configure settings
# ------------------

# Extract settings
t0 = cfg["t0"]
T = cfg["T"]
dt = cfg["dt"]
theta = cfg["theta"]
N = cfg["N"]
solver = cfg["solver"]

vtkfile_name = "Soln"

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
		t0, T, dt, theta=theta,
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