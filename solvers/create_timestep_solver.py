from firedrake import *

def create_timestep_solver(get_data, theta, Z, dx , dsN, u_old, u, make_weak_form, is_mixed,
                           bcs=None, nullspace=None, solver_parameters=None, appctx=None):
    """
    Prepare timestep solver by theta-scheme for 
        - given solution u_old at time t 
        - unknown u at time t+dt

    Return a solve function taking (t, dt)
    """

    # Initialize coefficients
    idt = Constant(0.0)
    u_trial = TrialFunction(Z)
    v = TestFunction(Z)

    # Initial weak form placeholders
    f = Function(Z)
    g = Function(Z)

    f_old = Function(Z)
    g_old = Function(Z)

    # Create the problem + solver once
    a, L = make_weak_form(
        theta, idt, 
        f, f_old, 
        g, g_old, 
        u_old,
        dx, dsN
    )(u_trial, v)
    
    problem_var = LinearVariationalProblem(a, L, u, bcs=bcs)
    solver = LinearVariationalSolver(
        problem_var,
        solver_parameters=solver_parameters,
        nullspace=nullspace, appctx=appctx
    )

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

        if is_mixed:
            f.sub(0).assign(project(data_new["ufl_f"], f.sub(0).function_space()))
            g.sub(0).assign(project(data_new["ufl_f"], g.sub(0).function_space()))
            f_old.sub(0).assign(project(data_old["ufl_f"], f.sub(0).function_space()))
            g_old.sub(0).assign(project(data_old["ufl_f"], g.sub(0).function_space()))
        else:
            f.assign(project(data_new["ufl_f"], f.sub(0).function_space()))
            g.assign(project(data_new["ufl_f"], g.sub(0).function_space()))
            f_old.assign(project(data_old["ufl_f"], f.sub(0).function_space()))
            g_old.assign(project(data_old["ufl_f"], g.sub(0).function_space()))

        # Run the solver
        solver.solve()
        return data_new

    return solve_one_step