from firedrake import *
import matplotlib.pyplot as plt
from solvers_2d.timestepper_MMS import timestepper_MMS
import numpy as np

# constants
T = 10.0                 # final time
dt = 0.05                # timestepping length
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
    # mesh
    mesh = UnitSquareMesh(N, N)
    x, y = SpatialCoordinate(mesh)

    t = Constant(0.0) # symbolic constant for t
    exp = ufl.exp # ufl e, so t gets calculated correctly

    # exact calculations for u=e^t*sin(pix)*cos(piy)
    ufl_v_exact = 0
    ufl_p_exact = 0
    ufl_f_exact = 0
    ufl_g_exact = 0
    ufl_u0 = 0

    # functions
    ufl_f = ufl_f_exact     # source term f
    ufl_g = ufl_g_exact     # bdy condition g
    ufl_v = ufl_v_exact     # velocity ic
    ufl_p = ufl_p_exact     # pressure ic

    # declare function space and interpolate funcs
    V = VectorFunctionSpace(mesh, "CG", 2)
    W = FunctionSpace(mesh, "CG", 1)
    Z = V * W

    # functions
    ufl_f = ufl_f_exact     # source term f
    ufl_g = ufl_g_exact     # bdy condition g
    ufl_v = ufl_v_exact     # velocity ic
    ufl_p = ufl_p_exact     # pressure ic

    def u_exact(x):
        values = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = 4 * x[1] * (1.0 - x[1])
        return values

    # declare function space and interpolate funcs
    v_cg2 = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
    s_cg1 = element("Lagrange", mesh.basix_cell(), 1)
    V = functionspace(mesh, v_cg2)
    W = functionspace(mesh, s_cg1)
    Z = V * W

    u_exact = Function(Z)
    f = Function(V)
    g = Function(V)
    u0 = Function(Z)

    u_exact.subfunctions[0].interpolate(ufl_v)
    u_exact.subfunctions[1].interpolate(ufl_p)
    u0.subfunctions[0].interpolate(ufl_v)
    u0.subfunctions[1].interpolate(ufl_p)

    # Boundary identifiers via callables (Firedrake accepts callables for sub_domain)
    def walls(x):
        # x is a numpy array with shape (ndim, npoints)
        return np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))

    def inflow(x):
        return np.isclose(x[0], 0.0)

    def outflow(x):
        return np.isclose(x[0], 1.0)
    
    # Dirichlet BCs
    bc_noslip = DirichletBC(V, Constant((0.0, 0.0)), walls)
    bc_inflow = DirichletBC(Q, Constant(8.0), inflow)
    bc_outflow = DirichletBC(Q, Constant(0.0), outflow)
    bcu = [bc_noslip]
    bcp = [bc_inflow, bc_outflow]


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