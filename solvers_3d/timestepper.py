from firedrake import *
from .create_timestep_solver import create_timestep_solver

def timestepper(V, dsN, theta, T, dt, u0, get_data, make_weak_form,
                bcs=None, nullspace=None, solver_parameters=None, appctx=None, W=None):
    """
    Perform timestepping using theta-scheme with
    final time T, timestep dt, initial datum u0 and
    function get_data(t) returning (f(t), g(t))
    """

    if W is not None:
        Z = V * W
    else:
        Z = V

    # Initialize solution function
    u_old = Function(Z)
    u_new = Function(Z)

    # Prepare solver for computing time step
    solver = create_timestep_solver(get_data, dsN, theta, u_old, u_new, make_weak_form,
                                    bcs, nullspace, solver_parameters, appctx, W)

    # Set initial condition
    u_old.assign(u0)

    # Perform timestepping
    t = 0
    outfile = VTKFile("soln.pvd")
    while t < T:

        # Report some numbers
        energy = assemble(inner(u_new.sub(0), u_new.sub(0)) * dx)
        print("{:10.4f} | {:10.4f} | {:#10.4g}".format(t, dt, energy))

        # Perform time step
        solver(t, dt)
        t += dt
        u_old.assign(u_new)

        # Write to file
        if W is None:
            outfile.write(u_new)
        else:
            outfile.write(u_new.sub(0), u_new.sub(1))