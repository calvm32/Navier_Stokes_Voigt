from firedrake import *
import yaml
import os
from pathlib import Path
import shutil

from solvers.timesteppers import timestepper_BDF2
from .make_weak_form import make_weak_form_BDF2
from solvers.printoff import blue, green
from solvers.config_setup import *
import matplotlib.pyplot as plt

# -------------------
# Get + archive paths
# -------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # adjust if this script moves
TEMPLATES_DIR = PROJECT_ROOT / "templates"

CFG_PATH1 = TEMPLATES_DIR / "settings" / "NS_MMS.yaml"
CFG_PATH2 = TEMPLATES_DIR / "solver_parameters" / "NS_MMS.yaml"
CFG_PATH3 = TEMPLATES_DIR / "ufl_expr" / "NS_MMS.yaml"

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

t0 = cfg["t0"]
T = cfg["T"]
theta = cfg["theta"]
gamma = cfg["gamma"]
H = cfg["H"]
L = cfg["L"]
Re = cfg["Re"]
G = cfg["G"]
P = cfg["P"]

solver_parameters = load_solver_parameters(CFG_PATH2)

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

    ufl_cfg = load_ufl_expressions(CFG_PATH3, namespace=namespace)

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

    v_error_list, p_error_list, palinstrophy_list, stream_func_list, enstrophy_list, every_time_list, energy_list, all_time_list = timestepper_BDF2(get_data, 
            Z, dx, ds, 
            t0, T, dt,
            make_weak_form=make_weak_form_BDF2,
            bcs=bcs, nullspace=nullspace,
            solver_parameters=solver_parameters,
            appctx=appctx, vtkfile_name=new_vtkfile_name)

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