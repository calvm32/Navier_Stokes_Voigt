from firedrake import *

def create_timestep_solver(get_data, theta, Z, dx , dsN, u_old, u_new, make_weak_form,
                           bcs=None, nullspace=None, solver_parameters=None):
    """
    Prepare timestep solver by theta-scheme for given
    solution u_old at time t and unknown u_new at time t + dt.
    Return a solve function taking (t, dt).
    """

    # Initialize coefficients
    idt = Constant(1.0)

    def solve_one_step(t, dt):
        """
        Update problem data to interval (t, t+dt) and run solver
        """
        idt.assign(1.0/dt)
        data_old = get_data(t)
        data_new = get_data(t+dt)

        f_old = data_old.get("ufl_f")
        g_old = data_old.get("ufl_g")
        f = data_new.get("ufl_f")
        g = data_new.get("ufl_g")
        
        v = TestFunction(Z)

        F = make_weak_form(theta, idt, f, f_old, g, g_old, dx , dsN)(u_new, u_old, v)

        problem_var = NonlinearVariationalProblem(F, u_new, bcs=bcs)
        solver = NonlinearVariationalSolver(problem_var,
                                        solver_parameters=solver_parameters,
                                        nullspace=nullspace, appctx=appctx)

        # Run the solver
        solver.solve()

    return solve_one_step