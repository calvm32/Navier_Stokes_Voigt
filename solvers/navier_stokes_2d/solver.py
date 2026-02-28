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

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

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

print(f"[solver.py] YAML configs archived in {run_dir}\n")

# ------------------
# Configure settings
# ------------------

cfg = load_config(CFG_PATH1)

# Extract settings
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
views = cfg["views"]

# Build appctx
appctx = {
    "Re": Re,
    "gamma": gamma,
    "velocity_x_space": 0
}

solver_parameters = load_solver_parameters(CFG_PATH2)

# views = news for solver param debugging
if views == "Full":
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
elif views == "Some":
    solver_parameters.update({
        'ksp_monitor_true_residual': None, 
        'snes_monitor': None,
    })

vtkfile_name = "Soln"

# --------------
# Configure mesh
# --------------

HERE = os.path.dirname(os.path.abspath(__file__))
MESH_PATH = os.path.join(HERE, "meshes", "step_big.msh")

print(f"[solver.py] Loading mesh from: {MESH_PATH}")

# ------------
# Setup spaces
# ------------

blue(f"\n*** Starting solve ***\n", spaced=True)

# Load the mesh
fine_mesh = Mesh(MESH_PATH)
x, y = SpatialCoordinate(fine_mesh)

# get height H
y_coords = fine_mesh.coordinates.dat.data[:, 1]

local_ymin = y_coords.min()
local_ymax = y_coords.max()

global_ymin = comm.allreduce(local_ymin, op=MPI.MIN)
global_ymax = comm.allreduce(local_ymax, op=MPI.MAX)

H = global_ymax - global_ymin

# get length L
x_coords = fine_mesh.coordinates.dat.data[:, 0]

local_xmin = x_coords.min()
local_xmax = x_coords.max()

global_xmin = comm.allreduce(local_xmin, op=MPI.MIN)
global_xmax = comm.allreduce(local_xmax, op=MPI.MAX)

L = global_xmax - global_xmin

if elements == "SV":
    k = 3  # or higher for stability on arbitrary triangles
    V = VectorFunctionSpace(fine_mesh, "CG", k)
    W = FunctionSpace(fine_mesh, "DG", k-1)
    Z = V * W
elif elements == "TH":
    V = VectorFunctionSpace(fine_mesh, "CG", 2)
    W = FunctionSpace(fine_mesh, "CG", 1)
    Z = V * W

# print(f"// V Total DoFs: {V.dof_count}")
# print(f"// W Total DoFs: {W.dof_count}")

# -------------------
# Configure functions
# -------------------

namespace = {
    "as_vector": as_vector,
    "Constant": Constant,
    "x": x,
    "y": y,
    "H": H,
    "L": L,
    "G": G,
    "P": P,
    "sin": sin,
    "cos": cos,
    "pi": pi,
}

ufl_cfg = load_ufl_expressions(CFG_PATH3, namespace=namespace)

# -------------------
# Boundary conditions
# -------------------

u_inflow = ufl_cfg["u_inflow"]

bc_inflow = DirichletBC(Z.sub(0), u_inflow, (1,2))
bc_walls = DirichletBC(Z.sub(0), Constant((0.0, 0.0)), (3,4))

bcs = [bc_walls, bc_inflow]
nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True, comm=Z.mesh().comm)])

# ------------------
# Allocate functions
# ------------------

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
    v_error_list, p_error_list, palinstrophy_list, stream_func_list, enstrophy_list, every_time_list, energy_list, all_time_list, velocity_x_vals, velocity_y_vals, omega_vals, r_vals, S2, energy_spec_list, energy_spec_probe = timestepper_CN(get_data, 
            Z, dx, ds, 
            t0, T, dt,
            sample_length=L, sample_height=H,
            make_weak_form=make_weak_form_CN, 
            bcs=bcs, nullspace=nullspace,
            solver_parameters=solver_parameters,
            appctx=appctx, vtkfile_name=vtkfile_name)

elif solver == "BDF2":
    v_error_list, p_error_list, palinstrophy_list, stream_func_list, enstrophy_list, every_time_list, energy_list, all_time_list, velocity_x_vals, velocity_y_vals, omega_vals, r_vals, S2, energy_spec_list, energy_spec_probe = timestepper_BDF2(get_data, 
            Z, dx, ds, 
            t0, T, dt, 
            sample_length=L, sample_height=H,
            make_weak_form_BDF2=make_weak_form_BDF2,
            make_weak_form_CN=make_weak_form_CN,
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
plt.ylabel("palinstrophy")
plt.title('Palinstrophy')
plt.grid(True)
plt.tight_layout()
if rank == 0:
    plt.savefig("1_palinstrophy_plot.png", dpi=200, bbox_inches='tight')
plt.close()

# --------------------
# Plot stream function
# --------------------

plot_data["stream_func"] = (every_time_list, stream_func_list)
plt.semilogy(every_time_list, stream_func_list, "-o")
plt.xlabel("time")
plt.ylabel("stream function")
plt.title('Stream Function')
plt.grid(True)
plt.tight_layout()
if rank == 0:
    plt.savefig("1_stream_func_plot.png", dpi=200, bbox_inches='tight')
plt.close()

# --------------
# Plot Enstrophy
# --------------

plot_data["enstrophy"] = (every_time_list, enstrophy_list)
plt.semilogy(every_time_list, enstrophy_list, "-o")
plt.xlabel("time")
plt.ylabel("enstrophy")
plt.title('Enstrophy')
plt.grid(True)
plt.tight_layout()
if rank == 0:
    plt.savefig("1_enstrophy_plot.png", dpi=200, bbox_inches='tight')
plt.close()

# -----------
# Plot Energy
# -----------

# pop first values 
all_time_list_del = all_time_list
all_time_list_del.pop(0)
energy_list_del = energy_list
energy_list_del.pop(0)

plot_data["energy"] = (all_time_list, energy_list)
plt.semilogy(all_time_list_del, energy_list_del, "-o")
plt.xlabel("time")
plt.ylabel("energy")
plt.title('Total Kinetic Energy')
plt.grid(True)
plt.tight_layout()
if rank == 0:
    plt.savefig("1_energy_plot.png", dpi=200, bbox_inches='tight')
plt.close()

# ------------
# Velocity PDF
# ------------

plot_data["velocity_x_pdf"] = (np.arange(len(velocity_x_vals)), velocity_x_vals)
plt.hist(velocity_x_vals, bins=100, density=True)
plt.xlabel("samples")
plt.ylabel("x-velocity")
plt.title('x-Velocity Probabiility Density Function')
plt.grid(True)
plt.tight_layout()
if rank == 0:
    plt.savefig("1_velocity_x_PDF.png", dpi=200, bbox_inches='tight')
plt.close()

plot_data["velocity_y_pdf"] = (np.arange(len(velocity_y_vals)), velocity_y_vals)
plt.hist(velocity_y_vals, bins=100, density=True)
plt.xlabel("samples")
plt.ylabel("y-velocity")
plt.title('y-Velocity Probabiility Density Function')
plt.grid(True)
plt.tight_layout()
if rank == 0:
    plt.savefig("1_velocity_y_PDF.png", dpi=200, bbox_inches='tight')
plt.close()

# -------------
# Vorticity PDF
# -------------

plot_data["vorticity_pdf"] = (np.arange(len(omega_vals)), omega_vals)
plt.hist(omega_vals, bins=100, density=True)
plt.xlabel("samples")
plt.ylabel("vorticity")
plt.title('Vorticity Probability Density Function')
plt.grid(True)
plt.tight_layout()
if rank == 0:
    plt.savefig("1_vorticity_PDF.png", dpi=200, bbox_inches='tight')
plt.close()

# --------------
# structure func
# --------------

plot_data["structure_function"] = (r_vals, S2)
plt.plot(r_vals, S2, "-o")
plt.xlabel(r"$r$")
plt.ylabel(r"$S_2(r)$")
plt.title('2nd-Order Longitudinal Structure Function')
plt.grid(True)
plt.tight_layout()
if rank == 0:
    plt.savefig("1_structure_function.png", dpi=200)
plt.close()

# -----------------------------------------------
# Plot Energy spectrum FF1 (at time, vary points)
# -----------------------------------------------

all_E = np.array([E for k, E in energy_spec_list])
mean_E = np.mean(all_E, axis=0)
k = energy_spec_list[0][0]

plt.figure()
plt.loglog(k, mean_E, 'r-', label="Spectrum at time")

# reference slope k^-3
C1 = mean_E[1] * k[1]**3
C2 = mean_E[1] * k[1]**(5/3)

plt.loglog(k, C1 * k**(-3), '--', label=r"$k^{-3}$")
plt.loglog(k, C2 * k**(-5/3), ':', label=r"$k^{-5/3}$")

plot_data["energy_spec_list"] = (energy_spec_list)
plt.xlabel('Wavenumber k')
plt.ylabel('E(k)')
plt.title('Point-Averaged Energy Spectrum at Final Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
if rank == 0:
    plt.savefig("1_energy_spec_FF1.png", dpi=200)
plt.close()

# -----------------------------------------------
# Plot Energy spectrum FF2 (at point, vary times)
# -----------------------------------------------

probe_values = np.array(energy_spec_probe)
time_values = np.array(every_time_list)

dt = time_values[1] - time_values[0]
T_total = len(time_values) * dt
N = len(time_values)

ux = probe_values[:, 0]
ux = ux - np.mean(ux) # remove mean

# FFT
u_hat = np.fft.fft(ux)

# frequencies
f = np.fft.fftfreq(N, d=dt)

# keep positive frequencies only
pos = f > 0
f = f[pos]
u_hat = u_hat[pos]

# temporal energy spectrum
E_f = (1/(2*T_total)) * np.abs(u_hat)**2

U_mean = np.mean(probe_values[:,0])

k = 2*np.pi*f / U_mean
E_k = E_f / U_mean

plt.figure()
plt.loglog(k[1:], E_k[1:], 'r-', label="Spectrum at probe")

C1 = E_k[5] * k[5]**3
C2 = E_k[5] * k[5]**(5/3)

plt.loglog(k, C1*k**(-3), '--', label=r"$k^{-3}$")
plt.loglog(k, C2*k**(-5/3), ':', label=r"$k^{-5/3}$")

plot_data["energy_spec_probe"] = (energy_spec_list)
plt.xlabel("Wavenumber k")
plt.ylabel("E(k)")
plt.title('Time-Averaged Energy Spectrum at Single Pt.')
plt.legend()
plt.grid(True)
plt.tight_layout()
if rank == 0:
    plt.savefig("1_energy_spec_FF2.png", dpi=200)
plt.close()

# --------------------
# Save all data to CSV
# --------------------

if rank == 0:
    with open("0_all_plot_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for key, (x_vals, y_vals) in plot_data.items():
            writer.writerow([f"# {key}"])
            writer.writerow(["x", "y"])
            for x, y in zip(x_vals, y_vals):
                writer.writerow([x, y])
            writer.writerow([])  # empty row between datasets

print("[solver.py] All plot data saved to '0_all_plot_data.csv'")