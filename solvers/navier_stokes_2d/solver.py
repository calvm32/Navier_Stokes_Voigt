import os

from firedrake import *

from solvers.timestepper import timestepper
from .make_weak_form import make_weak_form
from solvers.printoff import blue
import matplotlib.pyplot as plt

from .config_constants import t0, T, theta, gamma, P, G, Re, solver_parameters, vtkfile_name, appctx

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
dt = 0.01

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

u_inflow = as_vector((
    4*P*y*(y - H)/(H**2), # normalize at center line
    0.0
))

bc_inflow = DirichletBC(Z.sub(0), u_inflow, (1))
bc_walls = DirichletBC(Z.sub(0), Constant((0.0, 0.0)), (3,4))

bcs = [bc_walls, bc_inflow]

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

    # velocity exact
    ufl_v0 = as_vector([
        P*y*(y - H),
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

timestepper(get_data, theta, 
            Z, dx, ds, 
            t0, T, dt,
            make_weak_form=make_weak_form,
            bcs=bcs, nullspace=nullspace,
            solver_parameters=solver_parameters,
            appctx=appctx, vtkfile_name=vtkfile_name)
