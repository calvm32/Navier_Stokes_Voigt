from firedrake import *
from firedrake.functionspaceimpl import MixedFunctionSpace

def create_timestep_solver(get_data, theta, Z, dx , dsN, u_old, u_new, make_weak_form,
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
        data_old = get_data(t)
        data_new = get_data(t+dt)

        f_old = data_old.get("ufl_f")
        g_old = data_old.get("ufl_g")
        f = data_new.get("ufl_f")
        g = data_new.get("ufl_g")

        if isinstance(Z.ufl_element(), MixedElement):

            u, p = split(u_new)
            v, q = TestFunctions(Z)

            F = make_weak_form(theta, idt, f, f_old, g, g_old, dx , dsN)(u, p, u_old.sub(0), u_old.sub(1), v, q)

            problem_var = NonlinearVariationalProblem(F, u_new, bcs=bcs, J=None)
            solver = NonlinearVariationalSolver(problem_var, solver_parameters=solver_parameters, nullspace=nullspace)

            # === Attach Python-level appctx to any PCs that PCD will inspect ===
            # This ensures keys like "velocity_space" and "Re" are visible to PCDPC.
            try:
                # SNES -> KSP -> PC
                snes = solver.snes
                ksp = snes.getKSP()
                pc = ksp.getPC()

                # helper to copy appctx into a PC's python context (if present)
                def copy_appctx_to_pc(pc_obj, appctx):
                    pyctx = None
                    try:
                        pyctx = pc_obj.getPythonContext()
                    except Exception:
                        pyctx = None
                    if pyctx is not None:
                        # ensure appctx exists and then update it
                        if not hasattr(pyctx, "appctx") or pyctx.appctx is None:
                            pyctx.appctx = {}
                        pyctx.appctx.update(appctx)
                        # debug print to show keys visible to this PC
                        print("Attached appctx to PC (type={}): {}".format(pc_obj.getType(), pyctx.appctx.keys()))

                appctx = solver_parameters.get("appctx", {}) if solver_parameters else {}

                # Attach to top-level PC python context if present
                copy_appctx_to_pc(pc, appctx)

                # If this is a fieldsplit PC, attach to each sub-KSP's PC too
                if pc.getType() and pc.getType().lower().startswith("fieldsplit"):
                    # getFieldSplitSubKSP returns a list of sub-KSP objects in PETSc 3.14+;
                    try:
                        subksp_list = pc.getFieldSplitSubKSP()
                    except Exception:
                        # older PETSc API: try to fetch with getFieldSplitSubKSPS or similar
                        subksp_list = []
                    for subksp in subksp_list:
                        try:
                            subpc = subksp.getPC()
                            copy_appctx_to_pc(subpc, appctx)
                        except Exception:
                            pass
            except Exception as e:
                print("Warning: failed to attach appctx to PC(s):", e)
            # === end attach block ===
            
        else:
            v = TestFunction(Z)

            F = make_weak_form(theta, idt, f, f_old, g, g_old, dx , dsN)(u_new, u_old, v)

            problem_var = NonlinearVariationalProblem(F, u_new, bcs=bcs)
            solver = NonlinearVariationalSolver(problem_var, solver_parameters=solver_parameters)

        # Run the solver
        solver.solve()

    return solve_one_step