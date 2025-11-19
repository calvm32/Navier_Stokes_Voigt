from firedrake import *
import matplotlib.pyplot as plt
from solvers_2d.timestepper_MMS import timestepper_MMS

# constants
T = 2                   # final time
dt = 0.1                # timestepping length
theta = 1/2             # theta constant
Re = Constant(100.0)    # Reynold's num for viscosity

N_list = []
error_list = []

# setup from demo
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
            + 1.0 / Re * inner(grad(u_mid), grad(v)) * dx +
            inner(dot(grad(u_mid), u_mid), v) * dx -
            p * div(v) * dx +
            div(u_mid) * q * dx
            - inner((theta * f_np1 + (1 - theta) * f_n), v) * dx
        )

    return F

for exp in range(5, 20):
    N = 2**exp
    N_list.append(N)

    # mesh
    mesh = UnitSquareMesh(N, N)
    x, y = SpatialCoordinate(mesh)

    t = Constant(0.0) # symbolic constant for t
    exp = ufl.exp # ufl e, so t gets calculated correctly

    # exact calculations for u=e^t*sin(pix)*cos(piy)
    ufl_v_exact = exp(t)*cos(pi*x)*cos(pi*y)
    ufl_p_exact = exp(t)*cos(pi*x)*cos(pi*y)
    ufl_f_exact = (1+2*pi**2)*exp(t)*cos(pi*x)*cos(pi*y)
    ufl_g_exact = 0
    ufl_u0 = cos(pi*x)*cos(pi*y)

    # functions
    ufl_f = ufl_f_exact     # source term f
    ufl_g = ufl_g_exact     # bdy condition g
    ufl_v = ufl_v_exact     # velocity ic
    ufl_p = ufl_p_exact     # pressure ic

    # declare function space and interpolate funcs
    V = VectorFunctionSpace(mesh, "CG", 2)
    W = FunctionSpace(mesh, "CG", 1)
    Z = V * W

    u_exact = Function(Z)
    f = Function(V)
    g = Function(V)
    u0 = Function(Z)

    u_exact.subfunctions[0].interpolate(ufl_v)
    u_exact.subfunctions[1].interpolate(ufl_p)
    u0.subfunctions[0].interpolate(ufl_v)
    u0.subfunctions[1].interpolate(ufl_p)

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

    nullspace = MixedVectorSpaceBasis(
        Z, [Z.sub(0), VectorSpaceBasis(constant=True)])
    
    bcs = [DirichletBC(Z.sub(0), Constant((1, 0)), (4,)),
    DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3))]
    
    # run
    error = timestepper_MMS(V, ds(1), theta, T, dt, u_exact, get_data, make_weak_form, u_exact)

    # run
    error = timestepper_MMS(V, ds(1), theta, T, dt, u0, get_data, make_weak_form, u_exact, 
            bcs=bcs, nullspace=nullspace, solver_parameters=solver_parameters, 
            appctx=appctx, W=W)
    error_list.append(error)

plt.loglog(N_list, error_list, "-o")
plt.xlabel("mesh size h")
plt.ylabel("error")
plt.grid(True)

plt.savefig("convergence_plot.png", dpi=200)