import os

from firedrake import *

from solvers.timestepper import timestepper
from .make_weak_form import make_weak_form
from solvers.printoff import blue
import matplotlib.pyplot as plt

from .config_constants import t0, T, dt, theta, gamma, P, G, Re, solver_parameters, vtkfile_name, appctx

HERE = os.path.dirname(os.path.abspath(__file__))
MESH_PATH = os.path.join(HERE, "meshes", "poiseuille_with_step.msh")

if not os.path.exists(MESH_PATH):
    raise FileNotFoundError(f"Mesh not found at {MESH_PATH}")

print(f"[solver.py] Loading mesh from: {MESH_PATH}")

blue(f"\n*** Starting solve ***\n", spaced=True)

# ------------
# Setup spaces
# ------------

# Load the mesh
mesh = Mesh(MESH_PATH)
x, y = SpatialCoordinate(mesh)

# get height
y_coords = mesh.coordinates.dat.data[:, 1]
H = y_coords.max() - y_coords.min()

# get length
x_coords = mesh.coordinates.dat.data[:, 0]
L = x_coords.max() - x_coords.min()

dx = Measure("dx", domain=mesh)
ds = Measure("ds", domain=mesh)

V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)
Z = V * W

# -------------------
# Boundary conditions
# -------------------

bc_walls = DirichletBC(Z.sub(0), Constant((0.0, 0.0)), (3,4))
bcs = [bc_walls]

nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

"""# -------------
# CFL Condition
# -------------

hmin = mesh.cell_sizes.dat.data.min()

tol = 1e-12
x_coords = mesh.coordinates.dat.data[:, 0]
y_inflow = y_coords[np.abs(x_coords) < tol]

Umax = Re*(0.5*P*y_inflow*(H - y_inflow)).max()

CFL = 0.3
dt = CFL * hmin / Umax"""

# ------------------
# Allocate functions
# ------------------

def get_data(t):
    
    # Time-dependent modulation forcing (keeps solution unsteady)
    ramp = min(t / 0.5, 1.0)
    Pt = ramp * P * (1.0 + 0.2*sin(2*pi*t))
    ufl_f = as_vector((Pt, 0.0))

    # Breaks symmetry and avoids immediate steady-state lock-in
    ufl_v0 = as_vector((
        1e-3 * sin(pi * y / H),
        0.0
    ))

    ufl_p0 = Constant(0.0)
    ufl_g = as_vector([0.0,0.0])

    return {
        "ufl_v0": ufl_v0,
        "ufl_p0": ufl_p0,
        "ufl_f": ufl_f,
        "ufl_g": ufl_g
    }

# ----------
# Run solver
# ----------

v_error_list, p_error_list, palinstrophy_list, stream_func_list, vorticity_list, enstrophy_list, time_list = timestepper(get_data, theta, 
            Z, dx, ds, 
            t0, T, dt,
            make_weak_form=make_weak_form,
            bcs=bcs, nullspace=nullspace,
            solver_parameters=solver_parameters,
            appctx=appctx, vtkfile_name=vtkfile_name)

# -----------------
# Plot palinstrophy
# -----------------

plt.loglog(time_list, palinstrophy_list, "-o")
plt.xlabel("time")
plt.ylabel("palinstrophy L2")
plt.grid(True)
plt.tight_layout()
plt.savefig("palinstrophy_plot.png", dpi=200, bbox_inches='tight')
plt.close()

# --------------------
# Plot stream function
# --------------------

plt.loglog(time_list, stream_func_list, "-o")
plt.xlabel("time")
plt.ylabel("stream function L2")
plt.grid(True)
plt.tight_layout()
plt.savefig("stream_func_plot.png", dpi=200, bbox_inches='tight')
plt.close()

# --------------
# Plot vorticity
# --------------

plt.loglog(time_list, vorticity_list, "-o")
plt.xlabel("time")
plt.ylabel("vorticity L2")
plt.grid(True)
plt.tight_layout()
plt.savefig("vorticity_plot.png", dpi=200, bbox_inches='tight')
plt.close()

# --------------
# Plot Enstrophy
# --------------

plt.loglog(time_list, enstrophy_list, "-o")
plt.xlabel("time")
plt.ylabel("enstrophy L2")
plt.grid(True)
plt.tight_layout()
plt.savefig("enstrophy_plot.png", dpi=200, bbox_inches='tight')
plt.close()