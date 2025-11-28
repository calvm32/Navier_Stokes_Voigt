from firedrake import *
import matplotlib.pyplot as plt
from solvers_2d.timestepper_MMS import timestepper_MMS
from .make_weak_form import make_weak_form
import numpy as np
from .config import T, dt, theta, Re

P = Constant(10.0)      # pressure constant
H = Constant(5.0)       # height of rectangle, just take length = 3H

N_list = []
error_list = []

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

# calculate error as mesh size increases
for exp in range(1, 10):
    N = 2**exp
    N_list.append(N)

    # mesh
    mesh = RectangleMesh(3*N, N, 3*H, H) # rectangle btwn (0,0) and (3H, H)
    x, y = SpatialCoordinate(mesh)

    t = Constant(0.0) # symbolic constant for t
    ufl_exp = ufl.exp # ufl e, so t gets calculated correctly

    # exact calculations for Poiseuille flow 
    # \tfrac{1}{\nu}\big(\sin(\tfrac{\pi}{H} y)e^{\pi^2t/H^2} + \tfrac{1}{2}Py^2 + \tfrac{1}{2}PHy\big)
    ufl_v_exact = as_vector([Re*( sin(pi*y/H)*ufl_exp(((pi**2)*t)/(H**2)) + 0.5*P*y**2 + 0.5*P*H*y ), Constant(0.0)])
    ufl_p_exact = P
    ufl_f_exact = 0
    ufl_g_exact = 0

    # functions
    ufl_v = ufl_v_exact     # velocity ic
    ufl_p = ufl_p_exact     # pressure ic
    ufl_f = ufl_f_exact     # source term f
    ufl_g = ufl_g_exact     # bdy condition g

    # declare function space and interpolate functions
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
    
    # Boundary identifiers via callables (Firedrake accepts callables for sub_domain)
    def walls(x):
        return np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))

    def inflow(x):
        return np.isclose(x[0], 0.0)

    def outflow(x):
        return np.isclose(x[0], 1.0)
    
    # Dirichlet BCs
    bc_noslip = DirichletBC(V, Constant((0.0, 0.0)), walls)
    bc_inflow = DirichletBC(W, P, inflow)
    bc_outflow = DirichletBC(W, Constant(0.0), outflow)
    bcu = [bc_noslip]
    bcp = [bc_inflow, bc_outflow]

    # run
    error = timestepper_MMS(V, ds(1), theta, T, dt, u0, get_data, make_weak_form, u_exact,
            bcs=(bcu, bcp), nullspace=nullspace, solver_parameters=solver_parameters, 
            appctx=appctx, W=W)
    error_list.append(error)

plt.loglog(N_list, error_list, "-o")
plt.xlabel("mesh size h")
plt.ylabel("error")
plt.grid(True)

plt.savefig("convergence_plot.png", dpi=200)