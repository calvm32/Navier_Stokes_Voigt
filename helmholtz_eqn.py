from firedrake import *

"""
Solver for the indefinite Helmholtz equation used in wave problems:
    - (div)^2 u + u = f on the unit square
    - (div u) n = 0 on the boundary

For the Helmholtz equation used in meteorology, solve -(div)^2 u + u = f
"""

mesh = UnitSquareMesh(10, 10)

V = FunctionSpace(mesh, "CG", 1)

u = TrialFunction(V)
v = TestFunction(V)

# declare function space, define function f
f = Function(V)
x, y = SpatialCoordinate(mesh)
f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))

# RHS weak form, LHS weak form
a = ( - inner(grad(u), grad(v)) + inner(u, v)) * dx
L = inner(f, v) * dx

# solve
u = Function(V)
solve(a == L, u, solver_parameters={'ksp_type': 'cg', 'pc_type': 'none'})

VTKFile("helmholtz.pvd").write(u)