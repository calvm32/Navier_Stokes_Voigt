from firedrake import *

from solvers_2d.timestepper import timestepper
from .make_weak_form import make_weak_form
from solvers_2d.printoff import blue

from .config_constants import t0, T, dt, theta, N, solver_parameters, vtkfile_name

blue(f"\n*** Starting solve ***\n", spaced=True)

# mesh
mesh = UnitSquareMesh(N, N)
x, y = SpatialCoordinate(mesh)
ds = Measure("ds", domain=mesh)

# declare function space and interpolate functions
V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)
Z = V * W

# pass velocity space
solver_parameters["appctx"]["velocity_space"] = Z.sub(0).collapse()

# time dependant
def get_data(t):
    
    # functions
    ufl_v0 = as_vector([1+sin(pi*x), cos(pi*y)])    # velocity ic
    ufl_p0 = Constant(5.0)                          # pressure ic
    ufl_f = as_vector([2, 2])                       # source term f
    ufl_g = as_vector([2, 2])                       # bdy condition g

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

# run
timestepper(get_data, theta, Z, ds, t0, T, dt, make_weak_form, vtkfile_name=vtkfile_name,
        bcs=bcs, nullspace=nullspace, solver_parameters=solver_parameters)