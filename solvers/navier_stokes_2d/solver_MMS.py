from firedrake import *
import yaml
import os
from pathlib import Path
import shutil

from solvers.timesteppers import timestepper_BDF2
from .make_weak_form import make_weak_form_BDF2
from solvers.processing.printoff import blue, green
from solvers.processing.config_setup import *
import matplotlib.pyplot as plt

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

save_dir = Path(sys.argv[1])
cfg, solver_params, ufl_cfg = load_run_configs(save_dir)

# ------------------
# Configure settings
# ------------------

t0 = cfg["t0"]
T = cfg["T"]
theta = cfg["theta"]
gamma = cfg["gamma"]
H = cfg["H"]
L = cfg["L"]
Re = cfg["Re"]
G = cfg["G"]
P = cfg["P"]

solver = cfg["solver"]
elements = cfg["elements"]
views = cfg["views"]

vtkfile_name = "Soln"

# -------------
# Start solving
# -------------

# Loop over mesh resolutions
N_list = []
for n in range(4, 9):
    N = 2**n
    N_list.append(N)

# calculate error as mesh size increases
v_final_error_list = []
p_final_error_list = []

for N in N_list:

    dt = 1/N # CFL

    blue(f"\n*** Mesh size N = {N:0d} ***\n", spaced=True) # report mesh size
    new_vtkfile_name = f"{vtkfile_name}_N{N}" # write to new file

    appctx = {
        "Re": Re, 
        "gamma": gamma,
        "velocity_space": 0
    }

    # ------------
    # Setup spaces
    # ------------

    mesh = RectangleMesh(int(L*N), int(H*N), L, H)
    x, y = SpatialCoordinate(mesh)

    dx = Measure("dx", domain=mesh)
    ds = Measure("ds", domain=mesh)

    V = VectorFunctionSpace(mesh, "CG", 2)
    W = FunctionSpace(mesh, "CG", 1)
    Z = V * W

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
        "Re": Re,
        "pi": pi,
        "sin": sin,
        "cos": cos,
        "exp": exp,
    }

    # -------------------
    # Boundary conditions
    # -------------------

    bcs = [DirichletBC(Z.sub(0), Constant((0.0, 0.0)), (3, 4))]
    nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

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
        v_error_list, p_error_list, palinstrophy_list, stream_func_list, enstrophy_list, every_time_list, energy_list, all_time_list, velocity_x_vals, velocity_y_vals, omega_vals, r_vals, S2, energy_spec_list, energy_spec_probe, compute_every_large = timestepper_CN(get_data, 
                Z, dx, ds, 
                t0, T, dt, theta=theta, gamma=gamma, Re=Re,
                sample_length=L, sample_height=H,
                make_weak_form=make_weak_form_CN, 
                bcs=bcs, nullspace=nullspace,
                solver_parameters=solver_parameters,
                appctx=appctx, vtkfile_name=vtkfile_name)

    elif solver == "BDF2":
        v_error_list, p_error_list, palinstrophy_list, stream_func_list, enstrophy_list, every_time_list, energy_list, all_time_list, velocity_x_vals, velocity_y_vals, omega_vals, r_vals, S2, energy_spec_list, energy_spec_probe, compute_every_large = timestepper_BDF2(get_data, 
                Z, dx, ds, 
                t0, T, dt, gamma=gamma, Re=Re,
                sample_length=L, sample_height=H,
                make_weak_form_BDF2=make_weak_form_BDF2,
                make_weak_form_CN=make_weak_form_CN,
                bcs=bcs, nullspace=nullspace,
                solver_parameters=solver_parameters,
                appctx=appctx, vtkfile_name=vtkfile_name)

    v_final_error = 0
    for err in v_error_list:
        v_final_error += err
    
    v_final_error_list.append(sqrt(v_final_error))

    p_final_error = 0
    for err in p_error_list:
        p_final_error += err

    p_final_error_list.append(sqrt(p_final_error))

    green(f"Final L2 Error (velocity) = {v_final_error:0.8e}", spaced=True)
    green(f"Final L2 Error (pressure) = {p_final_error:0.8e}", spaced=True)

# -------------
# Velocity plot
# -------------

plt.figure()
plt.semilogy(N_list, v_final_error_list, "-o")
plt.xlabel("mesh size")
plt.ylabel("velocity error")
plt.grid(True)
plt.tight_layout()
plt.savefig("velocity_convergence_plot.png", dpi=200, bbox_inches='tight')
plt.close()

# --------------
# Presssure plot
# --------------

plt.figure()
plt.semilogy(N_list, p_final_error_list, "-o")
plt.xlabel("mesh size")
plt.ylabel("pressure error")
plt.grid(True)
plt.tight_layout()
plt.savefig("pressure_convergence_plot.png", dpi=200, bbox_inches='tight')
plt.close()