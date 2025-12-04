from firedrake import *

from solvers_2d.timestepper import timestepper
from .make_weak_form import make_weak_form
from solvers_2d.printoff import blue

from .config_constants import t0, T, dt, theta, N, solver_parameters, vtkfile_name

blue(f"\n*** Starting solve ***\n", spaced=True)

# mesh and measures
mesh = UnitSquareMesh(N, N)
x, y = SpatialCoordinate(mesh)

dx = Measure("dx", domain=mesh)
ds = Measure("ds", domain=mesh)

# declare function space and interpolate functions
V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)
Z = V * W

V_space = Z.sub(0).function_space()  # plain FunctionSpace
solver_parameters["appctx"]["velocity_space"] = V_space

# time dependant
def get_data(t):
    
    # functions
    ufl_v0 = as_vector([sin(pi*x), cos(pi*y)])    # velocity ic
    ufl_p0 = Constant(5.0)                          # pressure ic
    ufl_f = as_vector([0, 0])                       # source term f
    ufl_g = as_vector([0, 0])                       # bdy condition g

    # returns
    return {"ufl_v0": ufl_v0,
            "ufl_p0": ufl_p0,
            "ufl_f": ufl_f,
            "ufl_g": ufl_g,
            }

# BCs from demo
bcs = [DirichletBC(Z.sub(0), Constant((1, 0)), (4,)),
       DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3))]

nullspace = MixedVectorSpaceBasis(
    Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

solver_parameters["fieldsplit_1_pc_type"] = "python"
solver_parameters["fieldsplit_1_pc_python_type"] = "firedrake.AssembledPC"
solver_parameters["fieldsplit_1_pcd_Mp_pc_type"] = "lu"
solver_parameters["fieldsplit_1_pcd_Kp_pc_type"] = "lu"

solver_parameters["mat_type"] = "aij"


print("velocity_space type:", type(solver_parameters["appctx"]["velocity_space"]))


solver_parameters_test = solver_parameters.copy()
solver_parameters_test.pop("fieldsplit_1_pc_type", None)
solver_parameters_test.pop("fieldsplit_1_pc_python_type", None)
solver_parameters_test.pop("fieldsplit_1_pcd_Mp_pc_type", None)
solver_parameters_test.pop("fieldsplit_1_pcd_Kp_pc_type", None)

print(solver_parameters_test)

# run
timestepper(get_data, theta, Z, dx, ds(1), t0, T, dt, make_weak_form, vtkfile_name=vtkfile_name,
        bcs=bcs, nullspace=nullspace, solver_parameters=solver_parameters_test)