from firedrake import *

def create_stage_solver(get_data, Z, dx, dsN,
                        u_stage, k_out, make_residual,
                        bcs=None, solver_parameters=None):
    """
    Prepare an RK4 stage solver, with:
        k_out = F(u_stage, t_stage)

    Returns a function solve_stage(t_stage).
    """

    # Placeholders
    v = TestFunction(Z)
    k = TrialFunction(Z)

    f = Function(Z)
    g = Function(Z)

    # Create the problem + solver
    F_expr = make_residual(u_stage, k, v, f, g, dx, dsN)

    problem = LinearVariationalProblem(
        lhs(F_expr), rhs(F_expr), k_out
    )

    solver = LinearVariationalSolver(
        problem, solver_parameters=solver_parameters
    )

    # ------
    # Update
    # ------

    def update_data(t):
        data = get_data(t)
        f.interpolate(data["ufl_f"])
        g.interpolate(data["ufl_g"])

    return solver, update_data