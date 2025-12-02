from firedrake import *
from .create_timestep_solver import create_timestep_solver

def timestepper_MMS(V, dsN, theta, T, dt, u0, make_weak_form, u_exact, 
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
    solver = create_timestep_solver(dsN, theta, u_old, u_new, make_weak_form,
                                    bcs, nullspace, solver_parameters, appctx, W)

    # Set initial condition
    u_old.assign(u0)

    # Print table header
    print("{:10s} | {:10s} | {:10s}".format("t", "dt", "energy"))

    # Perform timestepping
    t = 0
    while t < T:

        # Report some numbers
        energy = assemble(inner(u_new.sub(0), u_new.sub(0)) * dx)
        print("{:10.4f} | {:10.4f} | {:#10.4g}".format(t, dt, energy))

        # Perform time step
        solver(t, dt)
        t += dt
        u_old.assign(u_new)

    # Write FINAL error to file
    u_error = errornorm(u_exact.sub(0), u_new.sub(0))
    return(u_error)