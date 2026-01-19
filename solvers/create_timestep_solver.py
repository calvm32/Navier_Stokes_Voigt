from firedrake import *

def create_timestep_solver(get_data, theta, Z, dx, dsN, u_old, u, make_weak_form, is_mixed,
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

    data = get_data(0)
    is_dict = isinstance(data["ufl_g"], dict)

    # Initial weak form placeholders
    # for f:
    f = Function(Z)
    f_old = Function(Z)

    # for g:
    if is_dict:
        g = None
        g_old = None
    else:
        g = Function(Z)
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

        # ----------
        # new, old f
        # ----------

        f.sub(0).interpolate(data_new["ufl_f"])
        f_old.sub(0).interpolate(data_old["ufl_f"])

        # ----------
        # new, old g
        # ----------

        nonlocal g, g_old

        if is_dict:
            g = data_new["ufl_g"]
            g_old = data_old["ufl_g"]
        else:
            g.sub(0).interpolate(data_new["ufl_g"])
            g_old.sub(0).interpolate(data_old["ufl_g"])

        # Run the solver
        solver.solve()

    return solve_one_step