from firedrake import *

def create_timestep_solver(get_data, dsN, theta, u_old, u_new, make_weak_form,
                           bcs, nullspace, solver_parameters, appctx):
    """
    Prepare timestep solver by theta-scheme for given
    function get_data(t) returning data (f(t), g(t)), given
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
    if appctx is not None:
        solver_kwargs["appctx"] = appctx

    # Initialize coefficients
    f_n, g_n = get_data(0)
    f_np1, g_np1 = get_data(0)
    idt = Constant(0)

    # Extract function space
    V = u_new.function_space()

    # callable weak form
    weak_form = make_weak_form(theta, idt, f_n, f_np1, g_n, g_np1, dsN)

    # Detect if space is mixed
    is_mixed = hasattr(V, "num_sub_spaces") and V.num_sub_spaces() > 1

    # Build weak form
    if is_mixed:
        trial_vars = TrialFunctions(V)
        test_vars = TestFunctions(V)
        F = weak_form(*trial_vars, *u_old.split(), *test_vars)
    else:
        u, v = TrialFunction(V), TestFunction(V)
        F = weak_form(u, u_old, v)

    a, L = lhs(F), rhs(F)

    def solve_(t, dt):
        """
        Update problem data to interval (t, t+dt) and run solver
        """

        # Update coefficients to current t, dt
        get_data(t, (f_n, g_n))
        get_data(t+dt, (f_np1, g_np1))
        idt.assign(1/dt)

        # Run the solver
        solve(a == L, u_new)

    return solve_