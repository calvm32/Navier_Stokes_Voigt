from firedrake import *

from solvers.timestepper import timestepper
from .make_weak_form import make_weak_form
from solvers.printoff import blue
import matplotlib.pyplot as plt

from .config_constants import t0, T, dt, theta, Re, gamma_gd, P, G, R, L, N_list, solver_parameters, vtkfile_name

# calculate error as mesh size increases
v_error_list = []
p_error_list = []

for N in N_list:

    dt = 1/N # CFL

    blue(f"\n*** Mesh size N = {N:0d} ***\n", spaced=True) # report mesh size
    new_vtkfile_name = f"{vtkfile_name}_N{N}" # write to new file

    appctx = {
        "Re": Re, 
        "gamma_gd": gamma_gd,
        "velocity_space": 0
    }

    # ------------
    # Setup spaces
    # ------------

    disk = UnitDiskMesh(N)
    mesh = ExtrudedMesh(disk, layers=N, layer_height=L/N)
    V = VectorFunctionSpace(mesh, "CG", 2)
    Z = V * FunctionSpace(mesh, "CG", 1)
    x, y, z = SpatialCoordinate(mesh)

    from firedrake import Function, FunctionSpace, FacetNormal, MeshFunction, MeshValueCollection, File, ds_h, ds_v

    # Create a DG0 function over facets
    mvc = MeshValueCollection("size_t", mesh, mesh.topological_dimension()-1)
    fmesh = MeshFunction(mvc, mesh)

    # Mark vertical facets
    for f in mesh.exterior_facets:
        coords = f.midpoint().array()
        # check if vertical (x-y distance ~ R) or horizontal (z = 0 or z = L)
        if abs(coords[2]) < 1e-10:
            fmesh[f] = 1  # bottom
        elif abs(coords[2]-L) < 1e-10:
            fmesh[f] = 2  # top
        else:
            fmesh[f] = 3  # side

    File("facet_ids.pvd").write(fmesh)

    dx = Measure("dx", domain=mesh)
    ds1 = Measure("ds", domain=mesh, subdomain_id=1)
    ds2 = Measure("ds", domain=mesh, subdomain_id=2)
    ds3 = Measure("ds", domain=mesh, subdomain_id=3)
    ds = ds1 + ds2 + ds3

    V = VectorFunctionSpace(mesh, "CG", 2)
    W = FunctionSpace(mesh, "CG", 1)
    Z = V * W

    # -------------------
    # Boundary conditions
    # -------------------

    eps = 1e-10

    def lateral_wall(x, on_boundary):
        r = sqrt(x[0]**2 + x[1]**2)
        return on_boundary and abs(r - R) < eps

    # Lateral walls are automatically marker 1 in ExtrudedMesh
    bcs = [DirichletBC(Z.sub(0), Constant((0,0,0)), 1)]
    nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

    # ------------------
    # Allocate functions
    # ------------------

    def get_data(t):

        # velocity exact
        ufl_v_exact = as_vector([
            0.0, 0.0,
            Re*(sin(sqrt(x**2+y**2)*pi/R)*exp((-1*pi**2*t)/(R**2*Re)) + 0.5*P*sqrt(x**2+y**2)*(sqrt(x**2+y**2) - R))
        ])

        # pressure exact
        ufl_p_exact = P*x + G

        # v time derivative
        v_t = as_vector([
            0.0, 0.0,
            (-1*pi**2/(R**2))*(sin(sqrt(x**2+y**2)*pi/R)*exp(-1*pi**2*t/(R**2*Re)))
        ])

        # v Laplacian
        lap_v = div(grad(ufl_v_exact))

        # pressure gradient
        grad_p = as_vector([P, 0.0, 0.0])

        # source termexact
        ufl_f_exact = as_vector([0.0, 0.0, 0.0])

        # boundary term
        ufl_g_exact = as_vector([0.0, 0.0, (P*z+G)*cos(z*pi/L)])

        return {
            "ufl_v0": ufl_v_exact,
            "ufl_p0": ufl_p_exact,
            "ufl_f": ufl_f_exact,
            "ufl_g": ufl_g_exact
        }

    # ----------
    # Run solver
    # ----------

    v_error, p_error = timestepper(get_data, theta, 
            Z, dx, ds, 
            t0, T, dt,
            make_weak_form=make_weak_form,
            bcs=bcs, nullspace=nullspace,
            solver_parameters=solver_parameters,
            appctx=appctx, vtkfile_name=new_vtkfile_name)


    v_error_list.append(v_error)
    p_error_list.append(p_error)

# -------------
# Velocity plot
# -------------

plt.figure()
plt.loglog(N_list, v_error_list, "-o")
plt.xlabel("mesh size h")
plt.ylabel("velocity error")
plt.grid(True)

plt.tight_layout()
plt.savefig("velocity_convergence_plot.png", dpi=200, bbox_inches='tight')
plt.close()

# --------------
# Presssure plot
# --------------

plt.figure()
plt.loglog(N_list, p_error_list, "-o")
plt.xlabel("mesh size h")
plt.ylabel("pressure error")
plt.grid(True)

plt.tight_layout()
plt.savefig("pressure_convergence_plot.png", dpi=200, bbox_inches='tight')
plt.close()