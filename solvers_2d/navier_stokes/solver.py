from firedrake import *
from solvers_2d.timestepper import timestepper
from .make_weak_form import make_weak_form
from solvers_2d.printoff import blue
from .config import t0, T, dt, theta, N, solver_parameters

blue(f"\n*** Starting solve ***\n", spaced=True)

# mesh
mesh = UnitSquareMesh(N, N)
x, y = SpatialCoordinate(mesh)

t = Constant(0.0) # symbolic constant for t
ufl_exp = ufl.exp # ufl e, so t gets calculated correctly 

# functions
ufl_v0 = as_vector([1, ufl_exp(t)*cos(pi*x)])       # velocity ic
ufl_p0 = Constant(5.0)                              # pressure ic
ufl_f = as_vector([0, 0])           # source term f
ufl_g = as_vector([0, 0])           # bdy condition g

# declare function space and interpolate functions
V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)
Z = V * W

function_space_appctx = {
    "velocity_space": V,
    "pressure_space": W,
    "ufl_v0": ufl_v0,
    "ufl_p0": ufl_p0,
    "ufl_f": ufl_f,
    "ufl_g": ufl_g,
    }

# setup from demo
bcs = [DirichletBC(Z.sub(0), Constant((1, 0)), (4,)),
       DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3))]

nullspace = MixedVectorSpaceBasis(
    Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

# 1) What is Z and subspaces?
print("Z:", Z)
print("Z.ufl_element():", Z.ufl_element())
print("Velocity subspace:", Z.sub(0))
print("Pressure subspace:", Z.sub(1))

# 2) u_new space and u_old
print("u_new.function_space():", u_new.function_space())
print("u_old.function_space():", u_old.function_space())

# 3) Boundary conditions list and their function spaces
print("bcs:", bcs)
if bcs:
    for bc in bcs:
        try:
            print(" - bc.apply to space:", bc.function_space())
        except Exception as e:
            print(" - bc inspect error:", e)

# 4) Print the (nonlinear) residual form and its shape info
print("Residual form F:")
print(F)   # only if F is in scope; if not, print what create_timestep_solver constructs

# 5) Check solver parameters being used
print("solver_parameters passed to solve (solver_kwargs in create_timestep_solver):")
print(solver_parameters)

# run
timestepper(theta, Z, ds(1), t0, T, dt, make_weak_form, function_space_appctx,
        bcs=bcs, nullspace=nullspace, solver_parameters=solver_parameters)