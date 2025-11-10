from firedrake import *

from solvers.timestepper import timestepper
from solvers.timestepper_adaptive import timestepper_adaptive

# constants
T = 2                   # final time
dt = 0.1                # timestepping length
theta = 1/2             # theta constant
tol = 0.001             # tolerance
N = 64                  # mesh size
Re = Constant(100.0)    # Reynold's num for viscosity

# mesh
mesh = UnitSquareMesh(N, N)

# declare function space
V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)
Z = V * W

up = Function(Z)
u, p = split(up)

x, y = SpatialCoordinate(mesh)

# functions
ufl_f = as_vector([0, 0])           # source term f
ufl_g = as_vector([0, 0])           # bdy condition g
ufl_velocity = as_vector([0, 0])    # velocity ic
ufl_pressure = Constant(0.0)        # pressure ic

f = Function(V)
g = Function(V)
u0 = Function(Z)

u0.sub(0).interpolate(ufl_velocity)  # velocity 
u0.sub(1).interpolate(ufl_pressure)  # pressure 

def make_weak_form(theta, idt, f_n, f_np1, g_n, g_np1, dsN):
    """
    Returns func F(u, u_old, p, q, v), 
    which builds weak form
    using external coefficients
    """

    def F(u, p, u_old, p_old, v, q):
        u_mid = theta * u + (1 - theta) * u_old

        return (
            idt * inner(u - u_old, v) * dx
            + (1.0 / Re) * inner(grad(u_mid), grad(v)) * dx
            + inner(dot(grad(u_mid), u_mid), v) * dx
            - inner(theta * f_np1 + (1 - theta) * f_n, v) * dx
            - p * div(v) * dx
            + div(u_mid) * q * dx
        )

    return F

# make data for iterative time stepping
def get_data(t, result=None):
    """Create or update data"""
    if result is None: # only allocate memory if hasn't been yet
        f = Function(V)
        g = Function(V)
    else:
        f, g = result

    f.interpolate(ufl_f)
    g.interpolate(ufl_g)
    return f, g

# setup from demo
bcs = [DirichletBC(Z.sub(0), Constant((1, 0)), (4,)),
       DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3))]

nullspace = MixedVectorSpaceBasis(
    Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

appctx = {"Re": Re, "velocity_space": 0}

solver_parameters = {
    "mat_type": "matfree",
    "snes_monitor": None,
    "ksp_type": "fgmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "lower",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "python",
    "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
    "fieldsplit_0_assembled_pc_type": "lu",
    "fieldsplit_1_ksp_type": "gmres",
    "fieldsplit_1_pc_type": "python",
    "fieldsplit_1_pc_python_type": "firedrake.PCDPC",
    "fieldsplit_1_pcd_Mp_pc_type": "lu",
    "fieldsplit_1_pcd_Kp_pc_type": "lu",
    "fieldsplit_1_pcd_Fp_mat_type": "matfree"
}

# run
timestepper(V, ds(1), theta, T, dt, u0, get_data, make_weak_form,
        bcs=bcs, nullspace=nullspace, solver_parameters=solver_parameters, 
        appctx=appctx, W=W)

timestepper_adaptive(V, ds(1), theta, T, tol, u0, get_data, make_weak_form,
        bcs=bcs, nullspace=nullspace, solver_parameters=solver_parameters, 
        appctx=appctx, W=W)