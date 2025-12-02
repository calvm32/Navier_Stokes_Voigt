from firedrake import *
from .create_timestep_solver import create_timestep_solver
from .printoff import iter_info_verbose, text, green

def timestepper_MMS(theta, Z, dsN, t, T, dt, N, make_weak_form,
                function_appctx, bcs=None, nullspace=None, solver_parameters=None):
    """
    Perform timestepping using theta-scheme with
    final time T, timestep dt, initial datum u0
    
    Also writes the DIFFERENCE between the 
    final solved solution and the exact solution
    """

    # -------------
    # Setup problem
    # -------------

    # Initialize solution function
    u_old = Function(Z)
    u_new = Function(Z)

    # initial condition
    ufl_u0 = function_appctx["ufl_u_exact"]
    u_old.interpolate(ufl_u0)

    # Prepare solver for computing time step
    solver = create_timestep_solver(theta, Z, dsN, u_old, u_new, make_weak_form,
                                    function_appctx, bcs, nullspace, solver_parameters)

    # Print table header
    energy = assemble(inner(u_old.sub(0), u_old.sub(0)) * dx)
    iter_info_verbose("INITIAL CONDITIONS", f"energy = {energy}", i=0, spaced=True)

    text(f"*** Beginning solve with step size {dt} ***", spaced=True)

    # --------------------
    # Perform timestepping
    # --------------------

    step = 0
    outfile = VTKFile(f"soln_N={N}.pvd")
    while t < T:
        # Perform time step
        solver(t, dt)
        t += dt
        u_old.assign(u_new)

        # count steps to print
        step += 1

        # Report some numbers
        energy = assemble(inner(u_new.sub(0), u_new.sub(0)) * dx)
        iter_info_verbose("TIME STEP COMPLETED", f"energy = {energy}", i=step)

        # -------------
        # Write to file
        # -------------
        if isinstance(Z, MixedFunctionSpace):
            ufl_v_exact = function_appctx["ufl_v_exact"]
            ufl_p_exact = function_appctx["ufl_p_exact"]
            u_exact.subfunctions[0].interpolate(ufl_v_exact)
            u_exact.subfunctions[1].interpolate(ufl_p_exact)

            # write to file
            outfile.write(u_new.sub(0), u_new.sub(1))
        
        else:
            ufl_u_exact = function_appctx["ufl_u_exact"]
            u_exact.subfunctions[0].interpolate(ufl_u_exact)

            # write to file
            outfile.write(u_new)
            

    # Write FINAL error to file
    u_error = errornorm(u_exact.sub(0), u_new.sub(0))
    green(f"Final L2 Error = {u_error:0.8e}", spaced=True)

    return(u_error)