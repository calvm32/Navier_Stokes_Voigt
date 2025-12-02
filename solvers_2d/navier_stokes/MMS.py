from firedrake import *
import matplotlib.pyplot as plt
from solvers_2d.timestepper_MMS import timestepper_MMS
from .make_weak_form import make_weak_form
from solvers_2d.printoff import blue
from .config import t0, T, dt, theta, Re, P, H, N_list, solver_parameters

error_list = []

# calculate error as mesh size increases
for N in N_list:

    blue(f"\n*** Mesh size N = {N:0d} ***\n", spaced=True)

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

    function_appctx = {
              "ufl_v_exact": ufl_v_exact,
              "ufl_p_exact": ufl_p_exact,
              "ufl_f": ufl_f_exact,
              "ufl_g": ufl_g_exact
              }

    # declare function space and interpolate functions
    V = VectorFunctionSpace(mesh, "CG", 2)
    W = FunctionSpace(mesh, "CG", 1)
    Z = V * W
    
    # BCs
    bcs = DirichletBC(Z.sub(0), Constant((0.0, 0.0)), (1, 3))
    nullspace = MixedVectorSpaceBasis(Z, [VectorSpaceBasis(constant=True), None])
    
    # run
    error = timestepper_MMS(theta, V, ds(1), t0, T, dt, N, make_weak_form, 
                            function_appctx, W=W, bcs=bcs, 
                            nullspace=nullspace, solver_parameters=solver_parameters)
    error_list.append(error)

plt.loglog(N_list, error_list, "-o")
plt.xlabel("mesh size h")
plt.ylabel("error")
plt.grid(True)

plt.savefig("convergence_plot.png", dpi=200)