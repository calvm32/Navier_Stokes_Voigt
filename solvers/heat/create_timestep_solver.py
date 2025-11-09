from firedrake import *

def create_timestep_solver(get_data, dsN, theta, u_old, u_new):
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

    # Prepare weak formulation
    u, v = TrialFunction(V), TestFunction(V)
    theta = Constant(theta)
    F = ( idt*(u - u_old)*v*dx
        + inner(grad(theta*u + (1-theta)*u_old), grad(v))*dx
        - (theta*f_np1 + (1-theta)*f_n)*v*dx
        - (theta*g_np1 + (1-theta)*g_n)*v*dsN
    )
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