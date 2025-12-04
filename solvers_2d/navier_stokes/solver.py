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

solver_parameters["appctx"]["velocity_space"] = V # Z.sub(0).topological.ufl_function_space() # necessary so it's a function space

# allocate Functions for timestepping
u_old = Function(Z, name="u_old")
u_new = Function(Z, name="u_new")

# Initialize velocity and pressure as real Functions
u_old.sub(0).interpolate(as_vector([sin(pi*x), cos(pi*y)]))  # velocity
u_old.sub(1).interpolate(Constant(5.0))                       # pressure

# Make "get_data" return Functions instead of symbolic UFL expressions
def get_data(t):
    
    # functions
    ufl_v0 = as_vector([t, cos(pi*y)])    # velocity ic
    ufl_p0 = Constant(5.0)                # pressure ic
    ufl_f = as_vector([0, 0])             # source term f
    ufl_g = as_vector([0, 0])             # bdy condition g

    # Allocate functions
    v = Function(V)
    p = Function(W)
    f = Function(V)
    g = Function(V)
    
    # Interpolate ufl symbolic expressions
    v.interpolate(ufl_v0)
    p.interpolate(ufl_p0)
    f.interpolate(ufl_f)
    g.interpolate(ufl_g)
    
    return {"ufl_v0": v,
            "ufl_p0": p,
            "ufl_f": f,
            "ufl_g": g}

# BCs from demo
bcs = [DirichletBC(Z.sub(0), Constant((1, 0)), (4,)),
       DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3))]

nullspace = MixedVectorSpaceBasis(
    Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

# run
timestepper(get_data, theta, Z, dx, ds(1), t0, T, dt, make_weak_form, vtkfile_name=vtkfile_name,
        bcs=bcs, nullspace=nullspace, solver_parameters=solver_parameters)