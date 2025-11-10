from firedrake import *
from .create_timestep_solver import create_timestep_solver

def timestepper(V, dsN, theta, T, dt, u0, get_data, make_weak_form,
                bcs, nullspace, solver_parameters, appctx, W):
    """
    Perform timestepping using theta-scheme with
    final time T, timestep dt, initial datum u0 and
    function get_data(t) returning (f(t), g(t))
    """

    # Initialize solution function
    u = Function(V)

    # Prepare solver for computing time step
    solver = create_timestep_solver(V, get_data, dsN, theta, u, u, make_weak_form,
                                    bcs, nullspace, solver_parameters, appctx, W)

    # Set initial condition
    u.interpolate(u0)

    # Perform timestepping
    t = 0
    outfile = VTKFile("solutions/soln.pvd")
    while t < T:

        # Report some numbers
        energy = assemble(u*dx)
        print("{:10.4f} | {:10.4f} | {:#10.4g}".format(t, dt, energy))

        # Perform time step
        solver(t, dt)
        t += dt

        # Write to file
        outfile.write(u)