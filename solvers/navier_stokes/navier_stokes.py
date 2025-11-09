from firedrake import *

from solvers.timestepper import timestepper
from solvers.timestepper_adaptive import timestepper_adaptive

# constants
T = 2                   # final time
dt = 0.1                # timestepping length
theta = 1/2             # theta constant
tol = 0.001             # tolerance
N = 64                  # mesh size
Re = Constant(100.0)    # Reynold's num

# mesh
mesh= UnitSquareMesh(N, N)

# declare function space
V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)
Z = V * W

up = Function(Z)
u, p = split(up)
v, q = TestFunctions(Z)

F = (
    1.0 / Re * inner(grad(u), grad(v)) * dx +
    inner(dot(grad(u), u), v) * dx -
    p * div(v) * dx +
    div(u) * q * dx
)

bcs = [DirichletBC(Z.sub(0), Constant((1, 0)), (4,)),
       DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3))]

nullspace = MixedVectorSpaceBasis(
    Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

# preconditioner dictionary setup in case more info is needed
appctx = {"Re": Re, "velocity_space": 0}

# Now we'll solve the problem.  First, using a direct solver.  Again, if
# MUMPS is not installed, this solve will not work, so we wrap the solve
# in a ``try/except`` block. ::

from firedrake.petsc import PETSc

try:
    solve(F == 0, up, bcs=bcs, nullspace=nullspace,
          solver_parameters={"snes_monitor": None,
                             "ksp_type": "gmres",
                             "mat_type": "aij",
                             "pc_type": "lu",
                             "pc_factor_mat_solver_type": "mumps"})
except PETSc.Error as e:
    if e.ierr == 92:
        warning("MUMPS not installed, skipping direct solve")
    else:
        raise e

# Now we'll show an example using the :class:`~.PCDPC` preconditioner
# that implements the pressure convection-diffusion approximation to the
# pressure Schur complement.  We'll need more solver parameters this
# time, so again we'll set those up in a dictionary. ::

parameters = {"mat_type": "matfree",
              "snes_monitor": None,

# We'll use a non-stationary Krylov solve for the Schur complement, so
# we need to use a flexible Krylov method on the outside. ::

             "ksp_type": "fgmres",
             "ksp_gmres_modifiedgramschmidt": None,
             "ksp_monitor_true_residual": None,

# Now to configure the preconditioner::

             "pc_type": "fieldsplit",
             "pc_fieldsplit_type": "schur",
             "pc_fieldsplit_schur_fact_type": "lower",

# we invert the velocity block with LU::

             "fieldsplit_0_ksp_type": "preonly",
             "fieldsplit_0_pc_type": "python",
             "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
             "fieldsplit_0_assembled_pc_type": "lu",

# and invert the schur complement inexactly using GMRES, preconditioned
# with PCD. ::

             "fieldsplit_1_ksp_type": "gmres",
             "fieldsplit_1_ksp_rtol": 1e-4,
             "fieldsplit_1_pc_type": "python",
             "fieldsplit_1_pc_python_type": "firedrake.PCDPC",

# We now need to configure the mass and stiffness solvers in the PCD
# preconditioner.  For this example, we will just invert them with LU,
# although of course we can use a scalable method if we wish. First the
# mass solve::

             "fieldsplit_1_pcd_Mp_ksp_type": "preonly",
             "fieldsplit_1_pcd_Mp_pc_type": "lu",

# and the stiffness solve.::

             "fieldsplit_1_pcd_Kp_ksp_type": "preonly",
             "fieldsplit_1_pcd_Kp_pc_type": "lu",

# Finally, we just need to decide whether to apply the action of the
# pressure-space convection-diffusion operator with an assembled matrix
# or matrix free.  Here we will use matrix-free::

             "fieldsplit_1_pcd_Fp_mat_type": "matfree"}

# With the parameters set up, we can solve the problem, remembering to
# pass in the application context so that the PCD preconditioner can
# find the Reynolds number. ::

up.assign(0)

solve(F == 0, up, bcs=bcs, nullspace=nullspace, solver_parameters=parameters,
      appctx=appctx)


u, p = up.subfunctions
u.rename("Velocity")
p.rename("Pressure")

VTKFile("cavity.pvd").write(u, p)