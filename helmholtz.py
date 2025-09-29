from firedrake import *
mesh = UnitSquareMesh(10, 10)

V = FunctionSpace(mesh, "CG", 1)

u = TrialFunction(V)
v = TestFunction(V)

# declare function space, define function f
f = Function(V)
x, y = SpatialCoordinate(mesh)
f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))

# RHS weak form, LHS weak form
a = (inner(grad(u), grad(v)) + inner(u, v)) * dx
L = inner(f, v) * dx

# solve
u = Function(V)
solve(a == L, u, solver_parameters={'ksp_type': 'cg', 'pc_type': 'none'})

VTKFile("helmholtz.pvd").write(u)