from firedrake import *
import numpy
import math

"""
Solver for the heat equation with source term
  - du/dt = nu Laplace(u) + f
  - weak form: d/dt (inner(u,v) * dx) = ((-1)*(nu)*inner(grad(u), grad(v)) + inner(f,v)) * dx

"""

nu = Constant(1)

dt = 0.05   # length of time step
T = 5.0     # final time
t = 0.0     # initial time

mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "CG", 1)

# functions
u = Function(V, name="soln")
u_next = Function(V)
v = TestFunction(V)

# rk4 functions
k1 = Function(V)
k2 = Function(V)
k3 = Function(V)
k4 = Function(V)

# declare function space, define source function f
f = Function(V)
x, y = SpatialCoordinate(mesh)
f.interpolate(-t*(x**2))

u.assign(1.0)

# time stepping
while float(t) < float(T):
  f.interpolate(-t*(x**2))

  # weak forms for rk4
  F1 = inner(k1,v) * dx - (((-1)*(nu)*inner(grad(k1), grad(v)) + inner(f,v)) * dx)
  solve(F1 == 0, k1, solver_parameters={'ksp_type': 'cg', 'pc_type': 'none'})

  F2 = inner(k2,v) * dx - (((-1)*(nu)*inner(grad(k2 + (dt*k1)/2), grad(v)) + inner(f,v)) * dx)
  solve(F2 == 0, k2, solver_parameters={'ksp_type': 'cg', 'pc_type': 'none'})

  F3 = inner(k3,v) * dx - (((-1)*(nu)*inner(grad(k3 + (dt*k2)/2), grad(v)) + inner(f,v)) * dx)
  solve(F3 == 0, k3, solver_parameters={'ksp_type': 'cg', 'pc_type': 'none'})

  F4 = inner(k4,v) * dx - (((-1)*(nu)*inner(grad(k4 + (dt*k3)), grad(v)) + inner(f,v)) * dx)
  solve(F4 == 0, k4, solver_parameters={'ksp_type': 'cg', 'pc_type': 'none'})

  u_next = u + (dt/6)*(k1 + (2*k2) + (2*k3)+k4)
  u.assign(u_next)

  t += dt

VTKFile("heat-eqn.pvd").write(u)