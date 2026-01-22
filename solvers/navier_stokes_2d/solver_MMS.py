from firedrake import *
import yaml
from pathlib import Path
import shutil

from solvers.timestepper import timestepper
from .make_weak_form import make_weak_form
from solvers.printoff import blue, green
from solvers.config_setup import *
import matplotlib.pyplot as plt

# -----------------
# MMS Configuration
# -----------------

CFG_PATH1 = Path(__file__).parent / "configs" / "MMS_constants.yaml"
cfg = load_config(CFG_PATH)

t0    = cfg["t0"]
T     = cfg["T"]
theta = cfg["theta"]
gamma = cfg["gamma"]
H     = cfg["H"]
L     = cfg["L"]
Re    = cfg["Re"]
G     = cfg["G"]
P     = cfg["P"]

CFG_PATH2 = Path(__file__).parent / "configs" / "MMS_solver_params.yaml"
solver_parameters = load_solver_parameters(CFG_PATH, dt=dt)

vtkfile_name = "Soln"

# Loop over mesh resolutions
N_list = []
for n in range(4, 9):
    N = 2**n
    N_list.append(N)

# calculate error as mesh size increases
v_finaL_error_list = []
p_final_error_list = []

# -------------
# Start solving
# -------------

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

    mesh = RectangleMesh(L*N, H*N, L, H)
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

    v_error_list, p_error_list, palinstrophy_list, stream_func_list, vorticity_list, enstrophy_list, time_list = timestepper(get_data, 
            Z, dx, ds, 
            t0, T, dt,
            make_weak_form=make_weak_form,
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
plt.loglog(N_list, v_final_error_list, "-o")
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
plt.loglog(N_list, p_final_error_list, "-o")
plt.xlabel("mesh size")
plt.ylabel("pressure error")
plt.grid(True)
plt.tight_layout()
plt.savefig("pressure_convergence_plot.png", dpi=200, bbox_inches='tight')
plt.close()

# -------------
# Archive YAMLs
# -------------

# current working directory
run_dir = Path(os.getcwd())

# copy YAML files to current directory
shutil.copy(CFG_PATH1, run_dir / CFG_PATH1.name)
shutil.copy(CFG_PATH2, run_dir / CFG_PATH2.name)

print(f"[solver.py] YAML configs archived in {run_dir}")