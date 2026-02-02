from firedrake import *
import yaml
from pathlib import Path
import os
import shutil
import csv

from solvers.timesteppers import *
from .make_weak_form import *
from solvers.printoff import blue
from solvers.config_setup import *
import matplotlib.pyplot as plt
import numpy as np

# -----------------
# Paths wrt current
# -----------------

CFG_PATH1 = Path(__file__).parent / "configs" / "constants.yaml"
cfg = load_config(CFG_PATH1)

CFG_PATH2 = Path(__file__).parent / "configs" / "solver_params.yaml"
solver_parameters = load_solver_parameters(CFG_PATH2)

# -------------
# Configuration
# -------------

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
solver = cfg["solver"]
elements = cfg["elements"]

# Build appctx
appctx = {
    "Re": Re,
    "gamma": gamma,
    "velocity_space": 0
}

solver_parameters = load_solver_parameters(CFG_PATH2)

# views = news
"""
solver_parameters.update({
    'ksp_view': None, 
    'pc_view': None,
    'snes_view': None, 
    'pc_fieldsplit_view': None,
    'firedrake_0_ksp_view': None,
    'firedrake_0_pc_view': None,
    'firedrake_1_ksp_view': None,
    'firedrake_1_pc_view': None,
})
"""

vtkfile_name = "Soln"

# ------------------
# Mesh Configuration
# ------------------

HERE = os.path.dirname(os.path.abspath(__file__))
MESH_PATH = os.path.join(HERE, "meshes", "step1.msh")

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

if elements == "SV":
    k = 3  # or higher for stability on arbitrary triangles
    V = VectorFunctionSpace(fine_mesh, "CG", k)
    W = FunctionSpace(fine_mesh, "DG", k-1)
    Z = V * W
elif elements == "TH":
    V = VectorFunctionSpace(mesh, "CG", 2)
    W = FunctionSpace(mesh, "CG", 1)
    Z = V * W

print(f"V Total DoFs: {V.dof_count}")
print(f"W Total DoFs: {W.dof_count}")

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

# Pressure space
# Q = Z.sub(1)

# nullspace = VectorSpaceBasis(
#     constant=True,
#     comm=Q.mesh().comm
# )

nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

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

if solver == "CN":
    v_error_list, p_error_list, palinstrophy_list, stream_func_list, enstrophy_list, every_time_list, energy_list, all_time_list, u_plus, y_plus, velocity_vals, omega_vals, r_vals, S2 = timestepper_CN(get_data, 
            Z, dx, ds, 
            t0, T, dt, sample_height=H, sample_length=L,
            make_weak_form=make_weak_form_CN, Re=Re,
            bcs=bcs, nullspace=nullspace,
            solver_parameters=solver_parameters,
            appctx=appctx, vtkfile_name=vtkfile_name)

elif solver == "BDF2":
    v_error_list, p_error_list, palinstrophy_list, stream_func_list, enstrophy_list, every_time_list, energy_list, all_time_list, u_plus, y_plus, velocity_vals, omega_vals, r_vals, S2 = timestepper_BDF2(get_data, 
            Z, dx, ds, 
            t0, T, dt, sample_height=H, sample_length=L,
            make_weak_form_BDF2=make_weak_form_BDF2, Re=Re,
            bcs=bcs, nullspace=nullspace,
            solver_parameters=solver_parameters,
            appctx=appctx, vtkfile_name=vtkfile_name)

# Data logging dict
plot_data = {}

# -----------------
# Plot palinstrophy
# -----------------

plot_data["palinstrophy"] = (every_time_list, palinstrophy_list)
plt.semilogy(every_time_list, palinstrophy_list, "-o")
plt.xlabel("time")
plt.ylabel("palinstrophy L2")
plt.grid(True)
plt.tight_layout()
plt.savefig("1_palinstrophy_plot.png", dpi=200, bbox_inches='tight')
plt.close()

# --------------------
# Plot stream function
# --------------------

plot_data["stream_func"] = (every_time_list, stream_func_list)
plt.semilogy(every_time_list, stream_func_list, "-o")
plt.xlabel("time")
plt.ylabel("stream function L2")
plt.grid(True)
plt.tight_layout()
plt.savefig("1_stream_func_plot.png", dpi=200, bbox_inches='tight')
plt.close()

# --------------
# Plot Enstrophy
# --------------

plot_data["enstrophy"] = (every_time_list, enstrophy_list)
plt.semilogy(every_time_list, enstrophy_list, "-o")
plt.xlabel("time")
plt.ylabel("enstrophy L2")
plt.grid(True)
plt.tight_layout()
plt.savefig("1_enstrophy_plot.png", dpi=200, bbox_inches='tight')
plt.close()

# -----------
# Plot Energy
# -----------

plot_data["energy"] = (all_time_list, energy_list)
plt.semilogy(all_time_list, energy_list, "-o")
plt.xlabel("time")
plt.ylabel("energy")
plt.grid(True)
plt.tight_layout()
plt.savefig("1_energy_plot.png", dpi=200, bbox_inches='tight')
plt.close()

# ------------------------
# plot Log Law of the Wall
# ------------------------

plot_data["loglaw"] = (y_plus, u_plus)
plt.semilogx(y_plus, u_plus, "o")
plt.xlabel(r"$y^+$")
plt.ylabel(r"$U^+$")
plt.grid(True)
plt.tight_layout()
plt.savefig("1_LogLaw.png", dpi=200, bbox_inches='tight')
plt.close()

# ------------
# Velocity PDF
# ------------

plot_data["velocity_pdf"] = (np.arange(len(velocity_vals)), velocity_vals)
plt.hist(velocity_vals, bins=100, density=True)
plt.xlabel("samples")
plt.ylabel("velocity")
plt.grid(True)
plt.tight_layout()
plt.savefig("1_velocity_PDF.png", dpi=200, bbox_inches='tight')
plt.close()

# -------------
# Vorticity PDF
# -------------

plot_data["vorticity_pdf"] = (np.arange(len(omega_vals)), omega_vals)
plt.hist(omega_vals, bins=100, density=True)
plt.xlabel("samples")
plt.ylabel("vorticity")
plt.grid(True)
plt.tight_layout()
plt.savefig("1_vorticity_PDF.png", dpi=200, bbox_inches='tight')
plt.close()

# --------------
# structure func
# --------------

plot_data["structure_function"] = (r_vals, S2)
plt.plot(r_vals, S2, "-o")
plt.xlabel(r"$r$")
plt.ylabel(r"$S_2(r)$")
plt.grid(True)
plt.tight_layout()
plt.savefig("1_structure_function.png", dpi=200)
plt.close()

# Save all data to CSV
with open("all_plot_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for key, (x_vals, y_vals) in plot_data.items():
        writer.writerow([f"# {key}"])
        writer.writerow(["x", "y"])
        for x, y in zip(x_vals, y_vals):
            writer.writerow([x, y])
        writer.writerow([])  # empty row between datasets

print("[solver.py] All plot data saved to '0_all_plot_data.csv'")