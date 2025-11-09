from firedrake import *

def create_timestep_solver(get_data, dsN, theta, u_old, u_new, make_weak_form):
    """
    Prepare timestep solver by theta-scheme for given
    function get_data(t) returning data (f(t), g(t)), given
    solution u_old at time t and unknown u_new at time t + dt.
    Return a solve function taking (t, dt).
    """

    # Initialize coefficients
    f_n, g_n = get_data(0)
    f_np1, g_np1 = get_data(0)
    idt = Constant(0)

    # Extract function space
    V = u_new.function_space()

    # Make a callable weak form
    weak_form = make_weak_form(theta, idt, f_n, f_np1, g_n, g_np1, dsN)

    # Prepare weak formulation
    u, v = TrialFunction(V), TestFunction(V)
    F = weak_form(u,u_old,v)
    a, L = lhs(F), rhs(F)

    def solve_(t, dt):
        """
        Update problem data to interval (t, t+dt) and run solver
        """

        # Update coefficients to current t, dt
        get_data(t, (f_n, g_n))
        get_data(t+dt, (f_np1, g_np1))
        idt.assign(1/dt)

        # Push log level - NEEDS DIFF IMPORT, BUT IS IT NECESSARY?
        # old_level = get_log_level()
        # warning = LogLevel.WARNING if cpp.__version__ > '2017.2.0' else WARNING
        # set_log_level(warning)

        # Run the solver
        solve(a == L, u_new)

        # Pop log level
        # set_log_level(old_level)

    return solve_