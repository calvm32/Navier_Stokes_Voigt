from firedrake import *

def create_timestep_solver(get_data, theta, Z, dx , dsN, u_old, u_new, make_weak_form,
                           bcs=None, nullspace=None, solver_parameters=None, appctx=None):
    """
    Prepare timestep solver by theta-scheme for 
        - given solution u_old at time t 
        - unknown u_new at time t+dt

    Return a solve function taking (t, dt)
    """

    # Initialize coefficients
    idt = Constant(0.0)
    v = TestFunction(Z)

    # Initial weak form placeholders
    f_old = Function(Z)
    f_new = Function(Z)
    g_old = Function(Z)
    g_new = Function(Z)

    # Create the problem + solver + compute Jacobian once
    F_expr = make_weak_form(theta, idt, f_new, f_old, g_new, g_old, dx, dsN)(u_new, u_old, v)
    J = derivative(F_expr, u_new)
    
    problem_var = NonlinearVariationalProblem(F_expr, u_new, bcs=bcs, J=J)
    solver = NonlinearVariationalSolver(problem_var,
                                        solver_parameters=solver_parameters,
                                        nullspace=nullspace, appctx=appctx)

    # ------
    # Update
    # ------

    def solve_one_step(t, dt):
        """
        Update problem data to interval (t, t+dt) and run solver
        """
        idt.assign(1.0/dt)
        data_old = get_data(t)
        data_new = get_data(t+dt)

        f_old.sub(0).interpolate(data_old["ufl_f"])
        f_new.sub(0).interpolate(data_new["ufl_f"])
        g_old.sub(0).interpolate(data_old["ufl_g"])
        g_new.sub(0).interpolate(data_new["ufl_g"])

        # Run the solver
        solver.solve()

    return solve_one_step