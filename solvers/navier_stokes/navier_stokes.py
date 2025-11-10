from firedrake import *

# ----------------------------
# Constants
# ----------------------------
T = 2             # final time
dt = 0.1          # timestep
theta = 0.5       # theta-scheme
N = 32            # mesh size
Re = Constant(100)  # Reynolds number

# ----------------------------
# Mesh and function spaces
# ----------------------------
mesh = UnitSquareMesh(N, N)
V = VectorFunctionSpace(mesh, "CG", 2)  # velocity
W = FunctionSpace(mesh, "CG", 1)        # pressure
Z = V * W

up = Function(Z)       # solution at new time
u_old = Function(Z)    # solution at previous time
u, p = split(up)
v, q = TestFunctions(Z)

# ----------------------------
# Initial condition
# ----------------------------
ufl_velocity = as_vector([0, 0])
ufl_pressure = Constant(0.0)
u_old.sub(0).interpolate(ufl_velocity)
u_old.sub(1).interpolate(ufl_pressure)
up.assign(u_old)  # initialize solution

# ----------------------------
# Source term and boundary conditions
# ----------------------------
ufl_f = as_vector([0, 0])  # source term
ufl_g = as_vector([0, 0])  # Dirichlet BCs

f = Function(V)
g = Function(V)

def get_data(t, result=None):
    if result is None:
        f, g = Function(V), Function(V)
    else:
        f, g = result
    f.interpolate(ufl_f)
    g.interpolate(ufl_g)
    return f, g

bcs = [
    DirichletBC(Z.sub(0), Constant((1, 0)), 4),       # top lid
    DirichletBC(Z.sub(0), Constant((0, 0)), (1,2,3))  # other walls
]

# Nullspace for pressure
nullspace = MixedVectorSpaceBasis(
    Z, [Z.sub(0), VectorSpaceBasis(constant=True)]
)

# Solver parameters for nonlinear solve with PCD
solver_parameters = {
    "mat_type": "matfree",
    "snes_monitor": None,
    "ksp_type": "fgmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "lower",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "python",
    "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
    "fieldsplit_0_assembled_pc_type": "lu",
    "fieldsplit_1_ksp_type": "gmres",
    "fieldsplit_1_pc_type": "python",
    "fieldsplit_1_pc_python_type": "firedrake.PCDPC",
    "fieldsplit_1_pcd_Mp_pc_type": "lu",
    "fieldsplit_1_pcd_Kp_pc_type": "lu",
    "fieldsplit_1_pcd_Fp_mat_type": "matfree"
}

appctx = {"Re": Re, "velocity_space": 0}

# ----------------------------
# Time-stepping
# ----------------------------
idt = Constant(1/dt)  # mass term for theta-scheme
outfile = VTKFile("solutions/soln.pvd")
t = 0.0

while t < T:
    print(f"t = {t:.4f}")

    # Update source term
    f_n, g_n = get_data(t)
    f_np1, g_np1 = get_data(t + dt)

    # Split previous solution
    u_old_v, p_old = u_old.sub(0), u_old.sub(1)

    # Build weak form for this timestep
    u_mid = theta*up.sub(0) + (1-theta)*u_old_v
    f_mid = theta*f_np1 + (1-theta)*f_n

    F = (
        idt*inner(up.sub(0) - u_old_v, v)*dx
        + 1/Re*inner(grad(u_mid), grad(v))*dx
        + inner(dot(grad(u_mid), u_mid), v)*dx
        - up.sub(1)*div(v)*dx
        + div(u_mid)*q*dx
        - inner(f_mid, v)*dx
    )

    # Solve nonlinear system
    solve(F == 0, up, bcs=bcs, nullspace=nullspace,
          solver_parameters=solver_parameters, appctx=appctx)

    # Save solution
    u, p = up.subfunctions
    u.rename("Velocity")
    p.rename("Pressure")
    outfile.write(u, p)

    # Update for next timestep
    u_old.assign(up)
    t += dt
