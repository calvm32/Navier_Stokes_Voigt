from firedrake import *
import yaml
import os
from pathlib import Path
import shutil

from solvers.timestepper_BDF2 import timestepper_BDF2
from .make_weak_form import make_weak_form_BDF2
from solvers.printoff import blue
from solvers.config_setup import *
import matplotlib.pyplot as plt

# ----------------------
# Paths wrt project root
# ----------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # adjust if this script moves
TEMPLATES_DIR = PROJECT_ROOT / "templates"

CFG_PATH1 = TEMPLATES_DIR / "constants" / "NS_MMS.yaml"
CFG_PATH2 = TEMPLATES_DIR / "solver_parameters" / "NS_MMS.yaml"

# -----------------
# MMS Configuration
# -----------------

cfg = load_config(CFG_PATH1)

t0 = cfg["t0"]
T = cfg["T"]
gamma = cfg["gamma"]
H = cfg["H"]
L = cfg["L"]
Re = cfg["Re"]
G = cfg["G"]
P = cfg["P"]

solver_parameters = load_solver_parameters(CFG_PATH2)

vtkfile_name = "Soln"

# -------------
# Archive YAMLs
# -------------

# current working directory
run_dir = Path(os.getcwd())

# copy YAML files to current directory
shutil.copy(CFG_PATH1, run_dir / CFG_PATH1.name)
shutil.copy(CFG_PATH1, run_dir / CFG_PATH2.name)

print(f"[solver.py] YAML configs archived in {run_dir}")

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
    # Boundary conditions
    # -------------------

    bcs = [DirichletBC(Z.sub(0), Constant((0.0, 0.0)), (3, 4))]
    nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

    # ------------------
    # Allocate functions
    # ------------------

    def get_data(t):

        # velocity exact
        ufl_v0 = as_vector([
            Re*(sin(y*pi/H)*exp((-1*pi**2*t)/(H**2*Re)) + 0.5*P*y*(y - H)),
            0.0
        ])

        # pressure exact
        ufl_p0 = P*x + G

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