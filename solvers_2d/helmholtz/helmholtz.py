from firedrake import *
from helmholtz_eqn_solver import helmholtz_eqn_solver

# mesh
mesh = UnitSquareMesh(10, 10)

# constant
nu = 1

# functions
ufl_f = cos(x*pi)*cos(y*pi)     # source term f
ufl_g = 0                       # bdy condition g
ufl_u0 = 0                      # initial condition u0

# declare function space and interpolate functions
V = FunctionSpace(mesh, "CG", 1)
x, y = SpatialCoordinate(mesh)

f = Function(V)
g = Function(V)
u0 = Function(V)

f.interpolate(ufl_f)
g.interpolate(ufl_g)
u0.interpolate(ufl_u0)

# run solvers
helmholtz_eqn_solver(f, g, nu, V)