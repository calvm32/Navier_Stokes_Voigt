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
    idt = Constant(0.0)

    f = Function(Z)
    g = Function(Z)
    f.sub(0).interpolate(function_appctx["ufl_f"])
    g.sub(0).interpolate(function_appctx["ufl_g"])

    # Make weak form
    weak_form = make_weak_form(theta, idt, f.sub(0), g.sub(0), dsN)

    if isinstance(Z.ufl_element(), MixedElement):
        (u, p) = split(u_new)
        (u_old_, p_old_) = split(u_old)
        (v, q) = TestFunctions(Z)
        F = weak_form(u, p, u_old_, p_old_, v, q)
        
    else:
        u = u_new
        v = TestFunction(Z)
        F = weak_form(u, u_old, v)

    def solve_(t, dt):
        """
        Update problem data to interval (t, t+dt) and run solver
        """

        idt.assign(1/dt)

        # Run the solver
        solve(F == 0, u_new, **solver_kwargs)

    return solve_