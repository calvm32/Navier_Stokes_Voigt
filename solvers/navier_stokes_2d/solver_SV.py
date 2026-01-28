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

# Build appctx
appctx = {
    "Re": Re,
    "gamma": gamma,
    "velocity_space": 0
}

CFG_PATH2 = Path(__file__).parent / "configs" / "USER_solver_params_SV.yaml"
solver_parameters = load_solver_parameters(CFG_PATH2, dt=dt)

vtkfile_name = "Soln"

# ------------------
# Mesh Configuration
# ------------------

HERE = os.path.dirname(os.path.abspath(__file__))
MESH_PATH = os.path.join(HERE, "meshes", "step_fine.msh")

if not os.path.exists(MESH_PATH):
    raise FileNotFoundError(f"Mesh not found at {MESH_PATH}")

print(f"[solver.py] Loading mesh from: {MESH_PATH}")

# -------------
# Archive YAMLs
# -------------

# current working directory
run_dir = Path(os.getcwd())

# copy YAML files to current directory
shutil.copy(CFG_PATH1, run_dir / CFG_PATH1.name)
shutil.copy(CFG_PATH2, run_dir / CFG_PATH2.name)

print(f"[solver.py] YAML configs archived in {run_dir}\n")

# ------------
# Setup spaces
# ------------

blue(f"\n*** Starting solve ***\n", spaced=True)

# Load the mesh
fine_mesh = Mesh(MESH_PATH)
x, y = SpatialCoordinate(fine_mesh)

# get height
y_coords = fine_mesh.coordinates.dat.data[:, 1]
H = y_coords.max() - y_coords.min()

# get length
x_coords = fine_mesh.coordinates.dat.data[:, 0]
L = x_coords.max() - x_coords.min()

dx = Measure("dx", domain=fine_mesh)
ds = Measure("ds", domain=fine_mesh)

k = 3  # or higher for stability on arbitrary triangles
V = VectorFunctionSpace(fine_mesh, "CG", k)
W = FunctionSpace(fine_mesh, "DG", k-1)
Z = V * W

# -------------------
# Boundary conditions
# -------------------

u_inflow = as_vector((
    4*y*(H-y)/(H**2), # normalize at center line
    0.0
))

bc_inflow = DirichletBC(Z.sub(0), u_inflow, (1,2))
bc_walls = DirichletBC(Z.sub(0), Constant((0.0, 0.0)), (3,4))

bcs = [bc_walls, bc_inflow]

pressure_nullspace = VectorSpaceBasis(constant=True)
nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), pressure_nullspace])

# ------------------
# Allocate functions
# ------------------

def get_data(t):

    # velocity
    ufl_v0 = as_vector([
        0.0, #4*P*y*(y - H)/(H**2), #P*y*(y - H),
        0.0
    ])

    # pressure
    ufl_p0 = 0

    # source term
    ufl_f0 = as_vector([0.0,0.0])

    # boundary term
    ufl_g0 = as_vector([0.0,0.0]) #as_vector([(L-x)*G/L - x*(P*L-G)/L, 0.0])

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

# -----------
# Plot Energy
# -----------

plt.loglog(all_time_list, energy_list, "-o")
plt.xlabel("time")
plt.ylabel("energy")
plt.grid(True)
plt.tight_layout()
plt.savefig("0_energy_plot.png", dpi=200, bbox_inches='tight')
plt.close()