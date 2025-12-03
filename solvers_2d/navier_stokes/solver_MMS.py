from firedrake import *

from solvers_2d.timestepper import timestepper
from .make_weak_form import make_weak_form
from solvers_2d.printoff import blue
import matplotlib as plt

from .config_constants import t0, T, dt, theta, Re, P, H, N_list, solver_parameters, vtkfile_name

error_list = []

# calculate error as mesh size increases
for N in N_list:

    blue(f"\n*** Mesh size N = {N:0d} ***\n", spaced=True)

    # mesh and measures
    mesh = RectangleMesh(3*N, N, 3*H, H) # rectangle btwn (0,0) and (3H, H)
    x, y = SpatialCoordinate(mesh)

    dx = Measure("dx", domain=mesh)
    ds = Measure("ds", domain=mesh)

    # declare function space and interpolate functions
    V = VectorFunctionSpace(mesh, "CG", 2)
    W = FunctionSpace(mesh, "CG", 1)
    Z = V * W

    # time dependant
    def get_data(t):
        
        # exact functions for Poiseuille flow  
        ufl_v_exact = as_vector([                                   # velocity ic
            Re*( sin(pi*y/H)*ufl.exp(((pi**2)*t)/(H**2)) + 0.5*P*y**2 + 0.5*P*H*y ), 
            Constant(0.0)
        ])
        ufl_p_exact = P                                             # pressure ic
        ufl_f_exact = as_vector([Constant(0.0), Constant(0.0)])     # source term f
        ufl_g_exact = as_vector([Constant(0.0), Constant(0.0)])     # bdy condition g

        # returns
        return {"ufl_v0": ufl_v_exact,
                "ufl_p0": ufl_p_exact,
                "ufl_f": ufl_f_exact,
                "ufl_g": ufl_g_exact,
                }

    # BCs
    bc_noslip = DirichletBC(Z.sub(0), Constant((0.0, 0.0)), (1, 3))
    bc_pressure_ref = DirichletBC(Z.sub(1), Constant(0.0), (2,))  # pin pressure at boundary id 2
    bcs = [bc_noslip, bc_pressure_ref]

    nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

    # run and get error
    new_vtkfile_name = f"{vtkfile_name}_N{N}"

    error = timestepper(get_data, theta, Z, dx, ds(1), t0, T, dt, make_weak_form,
        bcs=bcs, nullspace=nullspace, solver_parameters=solver_parameters, 
        vtkfile_name=new_vtkfile_name)
    error_list.append(error)

plt.loglog(N_list, error_list, "-o")
plt.xlabel("mesh size h")
plt.ylabel("error")
plt.grid(True)

plt.savefig("convergence_plot.png", dpi=200)

