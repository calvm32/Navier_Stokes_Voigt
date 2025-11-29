from firedrake import *
import matplotlib.pyplot as plt
from solvers_2d.timestepper_MMS import timestepper_MMS
from .config import T, dt, theta, Re, P, H

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
for exp in range(3, 10):
    N = 2**exp
    N_list.append(N)

    # mesh
    mesh = RectangleMesh(3*N, N, 3*H, H) # rectangle btwn (0,0) and (3H, H)
    x, y = SpatialCoordinate(mesh)

    t = Constant(0.0)
    ufl_exp = ufl.exp

    # exact functions for Poiseuille flow  
    ufl_v_exact = as_vector([                                   # velocity ic
        Re*( sin(pi*y/H)*ufl_exp(((pi**2)*t)/(H**2)) + 0.5*P*y**2 + 0.5*P*H*y ), 
        Constant(0.0)
    ])
    ufl_p_exact = P                                             # pressure ic
    ufl_f_exact = as_vector([Constant(0.0), Constant(0.0)])     # source term f
    ufl_g_exact = as_vector([Constant(0.0), Constant(0.0)])     # bdy condition g

    # declare function space and interpolate functions
    V = VectorFunctionSpace(mesh, "CG", 2)
    W = FunctionSpace(mesh, "CG", 1)
    Z = V * W

    u_exact = Function(Z)
    f = Function(V)
    g = Function(V)
    u0 = Function(Z)

    u_exact.subfunctions[0].interpolate(ufl_v_exact)
    u_exact.subfunctions[1].interpolate(ufl_p_exact)
    u0.subfunctions[0].interpolate(ufl_v_exact)
    u0.subfunctions[1].interpolate(ufl_p_exact)

    # make data for iterative time stepping
    def get_data(t, result=None):
        """Create or update data"""
        if result is None: # only allocate memory if hasn't been yet
            f = Function(V)
            g = Function(V)
        else:
            f, g = result

        f.interpolate(ufl_f_exact)
        g.interpolate(ufl_g_exact)
        return f, g
    
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
    
    # BCs
    bc_noslip = DirichletBC(Z.sub(0), Constant((0.0, 0.0)), (1, 3))
    bc_pressure_ref = DirichletBC(Z.sub(1), Constant(0.0), (2,))  # pin pressure at boundary id 2
    bcs = [bc_noslip, bc_pressure_ref]

    nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

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