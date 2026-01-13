from firedrake import * 

from solvers.timestepper import timestepper
from .make_weak_form import make_weak_form
from solvers.printoff import blue

from .config_constants import t0, T, dt, theta, N, solver_parameters, appctx, vtkfile_name

blue(f"\n*** Starting solve ***\n", spaced=True)

# ------------
# Setup spaces
# ------------

mesh = Mesh("meshes/poiseuille_with_step.msh")
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
bcs = [bc_noslip]

nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

# ------------------
# Allocate functions
# ------------------

def get_data(t):

    # velocity exact
    ufl_v_exact = as_vector([
        Re*(sin(pi*y/H)*exp((-1*pi**2*t)/(H**2)) + 0.5*P*y**2 - 0.5*P*H*y),
        0.0
    ])

    # pressure exact
    ufl_p_exact = P*x + G

    # v time derivative
    v_t = as_vector([
        Re*(-1*pi**2/(H**2))*(sin(y*pi/H)*exp(-1*pi**2*t/(H**2))), 
        0.0
    ])

    # v Laplacian
    lap_v = div(grad(ufl_v_exact))

    # pressure gradient
    grad_p = as_vector([P, 0.0])

    # source termexact
    ufl_f_exact = as_vector([0.0,0.0])

    # boundary term
    ufl_g_exact = as_vector([(L-x)*G/L - x*(P*L-G)/L, 0.0])

    return {
        "ufl_v0": ufl_v_exact,
        "ufl_p0": ufl_p_exact,
        "ufl_f": ufl_f_exact,
        "ufl_g": ufl_g_exact
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
