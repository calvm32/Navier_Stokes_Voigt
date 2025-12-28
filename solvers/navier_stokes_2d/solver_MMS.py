from firedrake import *

from solvers.timestepper import timestepper
from .make_weak_form import make_weak_form
from solvers.printoff import blue
import matplotlib as plt

from .config_constants import t0, T, dt, theta, Re, P, G, H, L, N_list, solver_parameters, appctx, vtkfile_name

# calculate error as mesh size increases
v_error_list = []
p_error_list = []

for N in N_list:

    dt /= N**2 # CFL

    blue(f"\n*** Mesh size N = {N:0d} ***\n", spaced=True) # report mesh size
    new_vtkfile_name = f"{vtkfile_name}_N{N}" # write to new file

    # ------------
    # Setup spaces
    # ------------

    mesh = RectangleMesh(N, N, L, H)
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

        # v_t
        diff = as_vector([
            Re*(-1*pi**2/(H**2))*(sin(y*pi/H)*e**(-1*pi**2*t/(H**2))), 
            Constant(0.0)
        ])

        # source term exact
        ufl_f_exact = as_vector([
            diff - (1.0/Re)*div(grad(ufl_v_exact)) + P,
            Constant(0.0)
        ])

        # boundary term
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

    v_error, p_error = timestepper(get_data, theta, 
            Z, dx, ds, 
            t0, T, dt,
            make_weak_form=make_weak_form,
            bcs=bcs, nullspace=nullspace,
            solver_parameters=solver_parameters,
            appctx=appctx, vtkfile_name=new_vtkfile_name)


    v_error_list.append(v_error)
    p_error_list.append(p_error)

plt.loglog(N_list, v_error_list, "-o")
plt.xlabel("mesh size h")
plt.ylabel("velocity error")
plt.grid(True)

plt.savefig("velocity_convergence_plot.png", dpi=200)

plt.loglog(N_list, p_error_list, "-o")
plt.xlabel("mesh size h")
plt.ylabel("pressure error")
plt.grid(True)

plt.savefig("pressure_convergence_plot.png", dpi=200)