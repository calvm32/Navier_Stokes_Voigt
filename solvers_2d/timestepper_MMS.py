from firedrake import *
from .create_timestep_solver import create_timestep_solver
from .printoff import iter_info_verbose, text, green

def timestepper_MMS(V, f, g,dsN, theta, T, dt, u0, make_weak_form, u_exact, 
                bcs=None, nullspace=None, solver_parameters=None, appctx=None, W=None):
    """
    Perform timestepping using theta-scheme with
    final time T, timestep dt, initial datum u0
    
    ONLY writes the DIFFERENCE between the 
    final solved solution and the exact solution
    """

    if W is not None:
        Z = V * W
    else:
        Z = V

    # Initialize solution function
    u_old = Function(Z)
    u_new = Function(Z)

    # Prepare solver for computing time step
    solver = create_timestep_solver(f, g, dsN, theta, u_old, u_new, make_weak_form,
                                    bcs, nullspace, solver_parameters, appctx, W)

    # Set initial condition
    u_old.assign(u0)

    # Print table header
    energy = assemble(inner(u_old.sub(0), u_old.sub(0)) * dx)
    iter_info_verbose("INITIAL CONDITIONS", f"energy = {energy}", i=0, spaced=True)

    text(f"*** Beginning solve with step size {dt} ***", spaced=True)

    # Perform timestepping
    t = 0
    step = 1
    while t < T:

        # Report some numbers
        energy = assemble(inner(u_new.sub(0), u_new.sub(0)) * dx)
        iter_info_verbose("TIME STEP COMPLETED", f"energy = {energy}", i=step)

        # Perform time step
        solver(t, dt)
        t += dt
        u_old.assign(u_new)

        # count steps to print
        step += 1

    # Write FINAL error to file
    u_error = errornorm(u_exact.sub(0), u_new.sub(0))
    green(f"Final L2 Error = {u_error:0.8e}", spaced=True)

    return(u_error)