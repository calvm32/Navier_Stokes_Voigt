from firedrake import *
from firedrake.functionspaceimpl import MixedFunctionSpace

def create_timestep_solver(get_data, theta, Z, dsN, u_old, u_new, make_weak_form,
                           bcs = None, nullspace = None, solver_parameters = None):
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
        data = get_data(t)

        if isinstance(Z.ufl_element(), MixedElement):
            f = data.get("f", Constant((0.0,0.0)))
            g = data.get("g", Constant((0.0,0.0)))

            u, p = split(u_new)
            v, q = TestFunctions(Z)

            F = make_weak_form(theta, idt, f=f, g=g, dsN=dsN)(u, p, u_old.sub(0), u_old.sub(1), v, q)

            problem_var = NonlinearVariationalProblem(F, u_new, bcs=bcs, J=None)
            solver = NonlinearVariationalSolver(problem_var, solver_parameters=solver_parameters, nullspace=nullspace)
            
        else:
            f = data.get("f", Constant(0.0))
            g = data.get("g", Constant((0.0,0.0)))

            v = TestFunction(Z)

            F = make_weak_form(theta, idt, f=f, g=g, dsN=dsN)(u_new, u_old, v)

            problem_var = NonlinearVariationalProblem(F, u_new, bcs=bcs)
            solver = NonlinearVariationalSolver(problem_var, solver_parameters=solver_parameters)

        # Run the solver
        solver.solve()

    return solve_one_step