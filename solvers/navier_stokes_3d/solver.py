from firedrake import *
import yaml
from pathlib import Path
import os
import shutil

from solvers.timestepper import timestepper
from .make_weak_form import make_weak_form
from solvers.printoff import blue
from solvers.config_setup import *
import matplotlib.pyplot as plt

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
gamma = cfg["gamma"]
Re = cfg["Re"]
G = cfg["G"]
P = cfg["P"]
R = cfg["R"]
L = cfg["L"]

# Build appctx
appctx = {
    "Re": Re,
    "gamma": gamma,
    "velocity_space": cfg.get("velocity_space", 0)
}

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
shutil.copy(CFG_PATH2, run_dir / CFG_PATH2.name)

print(f"[solver.py] YAML configs archived in {run_dir}")

# ------------
# Setup spaces
# ------------

blue(f"\n*** Starting solve ***\n", spaced=True)

mesh = RectangleMesh(N, N, L, H)
x, y = SpatialCoordinate(mesh)

dx = Measure("dx", domain=mesh)
ds = Measure("ds", domain=mesh)

V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)
Z = V * W

# -------------------
# Boundary conditions
# -------------------

bc_noslip = DirichletBC(Z.sub(0), Constant((0.0, 0.0)), (3, 4))
bcs = [bc_noslip]

nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

# ------------------
# Allocate functions
# ------------------

def get_data(t):

    # velocity exact
    ufl_v0 = as_vector([
        Re*(sin(pi*y/H)*exp((-1*pi**2*t)/(H**2)) + 0.5*P*y**2 - 0.5*P*H*y),
        0.0
    ])

    # pressure exact
    ufl_p0 = P*x + G

    # v time derivative
    v_t = as_vector([
        Re*(-1*pi**2/(H**2))*(sin(y*pi/H)*exp(-1*pi**2*t/(H**2))), 
        0.0
    ])

    # v Laplacian
    lap_v = div(grad(ufl_v0))

    # pressure gradient
    grad_p = as_vector([P, 0.0])

    # source termexact
    ufl_f0 = as_vector([0.0,0.0])

    # boundary term
    ufl_g0 = as_vector([(L-x)*G/L - x*(P*L-G)/L, 0.0])

    return {
        "ufl_v0": ufl_v0,
        "ufl_p0": ufl_p0,
        "ufl_f": ufl_f0,
        "ufl_g": ufl_g0
    }

# ----------
# Run solver
# ----------

v_error_list, p_error_list, palinstrophy_list, stream_func_list, enstrophy_list, every_time_list, energy_list, all_time_list = timestepper(get_data, 
            Z, dx, ds, 
            t0, T, dt,
            make_weak_form=make_weak_form,
            bcs=bcs, nullspace=nullspace,
            solver_parameters=solver_parameters,
            appctx=appctx, vtkfile_name=vtkfile_name)

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