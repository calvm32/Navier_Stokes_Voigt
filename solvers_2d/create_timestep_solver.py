from firedrake import *
from firedrake.functionspaceimpl import MixedFunctionSpace

def create_timestep_solver(theta, Z, dsN, u_old, u_new, make_weak_form,
                           function_appctx, bcs, nullspace, solver_parameters):
    """
    Prepare timestep solver by theta-scheme for given
    solution u_old at time t and unknown u_new at time t + dt.
    Return a solve function taking (t, dt).
    """

    # Default solver settings
    solver_kwargs = {}
    if bcs is not None:
        solver_kwargs["bcs"] = bcs
    if nullspace is not None:
        solver_kwargs["nullspace"] = nullspace
    if solver_parameters is not None:
        solver_kwargs["solver_parameters"] = solver_parameters

    # Initialize coefficients
    idt = Constant(1.0)

    f = Function(Z.sub(0))
    g = Function(Z.sub(0))
    f.interpolate(function_appctx["ufl_f"])
    g.interpolate(function_appctx["ufl_g"])

    # Make weak form
    weak_form = make_weak_form(theta, idt, f, g, dsN)

    if isinstance(Z.ufl_element(), MixedElement):
        u, p = split(u_new)
        v, q = TestFunctions(Z)
        u_old, p_old = split(u_old)
        F = weak_form(u, p, u_old, p_old, v, q)
        
    else:
        u = u_new
        v = TestFunction(Z)
        F = weak_form(u, u_old, v)

    # Build Jacobian (trial function on same mixed space)
    W_trial = TrialFunction(Z)
    J = derivative(F, u_new, W_trial)

    # Create a NonlinearVariationalProblem and solver (applies solver kwargs including SNES/KSP)
    problem = NonlinearVariationalProblem(F, u_new, bcs=bcs, J=J)
    solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters)

    def solve_(t, dt):
        """
        Update problem data to interval (t, t+dt) and run solver
        """
        u_new.assign(u_old)
        idt.assign(1/dt)

        # re-interpolate forcing
        f.interpolate(function_appctx["ufl_f"])
        g.interpolate(function_appctx["ufl_g"])

        # Run the solver
        solver.solve()

    return solve_