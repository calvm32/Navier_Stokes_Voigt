from firedrake import *

def create_timestep_solver(get_data, theta, Z, dx , dsN, u_old, u_new, make_weak_form,
                           bcs=None, nullspace=None, solver_parameters=None, appctx=None):
    """
    Prepare timestep solver by theta-scheme for 
        - given solution u_old at time t 
        - unknown u_new at time t+dt

    Return a solve function taking (t, dt)
    """

    # ---------------
    # Make new solver
    # ---------------

    # Initialize coefficients
    idt = Constant(0.0)
    v = TestFunction(Z)

    # Initial weak form with placeholders
    data_init = get_data(0.0)
    f_old = data_init.get("ufl_f")
    g_old = data_init.get("ufl_g")
    f_new = data_init.get("ufl_f")
    g_new = data_init.get("ufl_g")

    F_expr = make_weak_form(theta, idt, f_new, f_old, g_new, g_old, dx, dsN)(u_new, u_old, v)

    # Create the solver once
    J = derivative(F_expr, u_new)
    problem_var = NonlinearVariationalProblem(F_expr, u_new, bcs=bcs, J=J)
    solver = NonlinearVariationalSolver(problem_var,
                                        solver_parameters=solver_parameters,
                                        nullspace=nullspace, appctx=appctx)

    # -------------
    # Update solver
    # -------------

    def solve_one_step(t, dt):
        """
        Update problem data to interval (t, t+dt) and run solver
        """
        idt.assign(1.0/dt)
        data_old = get_data(t)
        data_new = get_data(t+dt)

        f_old = data_old.get("ufl_f")
        g_old = data_old.get("ufl_g")
        f_new = data_new.get("ufl_f")
        g_new = data_new.get("ufl_g")

        # Run the solver
        solver.solve()

    return solve_one_step