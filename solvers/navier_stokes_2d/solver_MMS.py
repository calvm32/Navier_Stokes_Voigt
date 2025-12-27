from firedrake import *

from solvers_2d.timestepper import timestepper
from .make_weak_form import make_weak_form
from solvers_2d.printoff import blue
import matplotlib as plt

from .config_constants import t0, T, dt, theta, Re, P, G, H, L, N_list, solver_parameters, appctx, vtkfile_name

# calculate error as mesh size increases
error_list = []
for N in N_list:

    blue(f"\n*** Mesh size N = {N:0d} ***\n", spaced=True) # report mesh size
    new_vtkfile_name = f"{vtkfile_name}_N{N}" # write to new file

    # ------------
    # Setup spaces
    # ------------

    mesh = RectangleeMesh(H, L)
    x, y = SpatialCoordinate(mesh)

    dx = Measure("dx", domain=mesh)
    ds = Measure("ds", domain=mesh)

    V = VectorFunctionSpace(mesh, "CG", 2)
    W = FunctionSpace(mesh, "CG", 1)
    Z = V * W

    # -------------------
    # Boundary conditions
    # -------------------

    bc_noslip = DirichletBC(Z.sub(0), Constant((0.0, 0.0)), (3, 4))
    bc_pressure_ref = DirichletBC(Z.sub(1), Constant(G), (1,))  # pin pressure at left side
    bcs = [bc_noslip, bc_pressure_ref]

    nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

    # ------------------
    # Allocate functions
    # ------------------

    def get_data(t):

        # velocity exact
        ufl_v_exact = as_vector([
            Re*(sin(pi*y/H)*exp((-1*pi**2*t)/(H**2)) + 0.5*P*y**2 - 0.5*P*H*y),
            Constant(0.0)
        ])

        # pressure exact
        ufl_p_exact = Constant(P)*x + Constant(G)

        # source term for velocity (forcing)
        ufl_f_exact = as_vector([Constant(0.0), Constant(0.0)])

        # boundary term (velocity)
        ufl_g_exact = as_vector([Constant(0.0), Constant(0.0)])

        return {
            "ufl_v0": ufl_v_exact,
            "ufl_p0": ufl_p_exact,
            "ufl_f": ufl_f_exact,
            "ufl_g": ufl_g_exact
        }

    # ----------
    # Run solver
    # ----------

    error = timestepper(get_data, theta, 
            Z, dx, ds, 
            t0, T, dt,
            make_weak_form=make_weak_form,
            bcs=bcs, nullspace=nullspace,
            solver_parameters=solver_parameters,
            appctx=appctx, vtkfile_name=new_vtkfile_name)


    error_list.append(error)

plt.loglog(N_list, error_list, "-o")
plt.xlabel("mesh size h")
plt.ylabel("error")
plt.grid(True)

plt.savefig("convergence_plot.png", dpi=200)
