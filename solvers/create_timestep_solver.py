from firedrake import *

def create_timestep_solver(get_data, theta, Z, dx , dsN, u_old, u, make_weak_form,
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

        if isinstance(Z.ufl_element(), MixedElement):
            f.sub(0).interpolate(data_new["ufl_f"])
            g.sub(0).interpolate(data_new["ufl_g"])
            f_old.sub(0).interpolate(data_old["ufl_f"])
            g_old.sub(0).interpolate(data_old["ufl_g"])
        else:
            f.interpolate(data_new["ufl_f"])
            g.interpolate(data_new["ufl_g"])
            f_old.interpolate(data_old["ufl_f"])
            g_old.interpolate(data_old["ufl_g"])

        # Run the solver
        solver.solve()

    return solve_one_step