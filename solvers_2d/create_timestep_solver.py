from firedrake import *
from firedrake.functionspaceimpl import MixedFunctionSpace

def create_timestep_solver(theta, Z, dsN, u_old, u_new, make_weak_form,
                           function_space_appctx, bcs, nullspace, solver_parameters):
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

    # --- Create forcing / boundary Functions on the *correct* subspaces ---
    # prefer using the explicit velocity/pressure spaces from appctx rather than Function(Z).sub(...)
    V = function_space_appctx.get("velocity_space", None)
    W = function_space_appctx.get("pressure_space", None)

    if V is None:
        # fallback: if no spaces provided, use Z or its first subspace
        try:
            V = Z.sub(0).collapse()
        except Exception:
            V = Z

    # Create f, g on velocity space (adjust if g should be on another space)
    f = Function(V)
    g = Function(V)

    # if ufl_f / ufl_g are provided as UFL expressions depending on a symbolic t,
    # we will re-interpolate them inside solve_ each timestep (see below).
    ufl_f = function_space_appctx.get("ufl_f", None)
    ufl_g = function_space_appctx.get("ufl_g", None)

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
        if ufl_f is not None:
            # If ufl_f is an Expression that references a Symbolic Constant 't_sym', it will evaluate
            # correctly when interpolated into f if that Constant was updated externally.
            try:
                f.interpolate(ufl_f)
            except Exception:
                # fallback: if ufl_f is material for Function(V).interpolate to fail,
                # try assign if shapes match
                try:
                    f.assign(ufl_f)
                except Exception:
                    pass

        if ufl_g is not None:
            try:
                g.interpolate(ufl_g)
            except Exception:
                try:
                    g.assign(ufl_g)
                except Exception:
                    pass

        idt.assign(1/dt)

        # Run the solver
        solve(F == 0, u_new, **solver_kwargs)

    return solve_