from firedrake import *

from solvers.timestepper import timestepper
from .make_weak_form import make_weak_form
from solvers.printoff import blue

from .config_constants import t0, T, dt, N, vtkfile_name, solver_parameters

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
N = cfg["N"]

blue(f"\n*** Starting solve ***\n", spaced=True)

# -------------
# Archive YAMLs
# -------------

# current working directory
run_dir = Path(os.getcwd())

# copy YAML files to current directory
shutil.copy(CFG_PATH1, run_dir / CFG_PATH1.name)
shutil.copy(CFG_PATH1, run_dir / CFG_PATH1.name)

print(f"[solver.py] YAML configs archived in {run_dir}")

# ------------
# Setup spaces
# ------------

# mesh and measures
mesh = UnitCubeMesh(N, N, N)
x, y, z = SpatialCoordinate(mesh)

dx = Measure("dx", domain=mesh)
ds = Measure("ds", domain=mesh)

# declare function space and interpolate functions
V = FunctionSpace(mesh, "CG", 1)

# ------------------
# Allocate functions
# ------------------

# time dependant
def get_data(t):

    # exact functions for u=e^t*sin(pix)*cos(piy)*cos(pi*z)  
    ufl_u0 = ufl.exp(t)*cos(pi*x)*cos(pi*y)*cos(pi*z)                  # initial condition u0 
    ufl_f0 = (1+2*pi**2)*ufl.exp(t)*cos(pi*x)*cos(pi*y)*cos(pi*z)      # source term f 
    ufl_g0 = Constant(0)                                               # bdy condition g

    # returns
    return {"ufl_u0": ufl_u0,
            "ufl_f": ufl_f0,
            "ufl_g": ufl_g0}

# ----------
# Run solver
# ----------

u_error_list, palinstrophy_list, stream_func_list, vorticity_list, enstrophy_list, time_list = timestepper(get_data, 
        V, dx, ds, 
        t0, T, dt,
        make_weak_form=make_weak_form,
        solver_parameters=solver_parameters,
        vtkfile_name=new_vtkfile_name)