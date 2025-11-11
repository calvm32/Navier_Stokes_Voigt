from firedrake import *

from .compute_est import compute_est
from .compute_new_dt import compute_new_dt
from .create_timestep_solver import create_timestep_solver

def timestepper_adaptive(V, dsN, theta, T, tol, u0, get_data, make_weak_form,
                        bcs=None, nullspace=None, solver_parameters=None, appctx=None, W=None):
    """
    Perform adaptive timestepping using theta-scheme with
    final time T, tolerance tol, initial datum u0 and
    function get_data(t) returning (f(t), g(t))
    """

    if W is not None:
        Z = V * W
    else:
        Z = V

    # Initialize needed functions
    u_n = Function(Z)
    u_np1_low = Function(Z)
    u_np1_high = Function(Z)

    # Prepare solvers for computing tentative time steps
    solver_low = create_timestep_solver(get_data, dsN, theta, u_n, u_np1_low, make_weak_form,
                                        bcs, nullspace, solver_parameters, appctx, W)
    solver_high_1 = create_timestep_solver(get_data, dsN, theta, u_n, u_np1_high, make_weak_form,
                                        bcs, nullspace, solver_parameters, appctx, W)
    solver_high_2 = create_timestep_solver(get_data, dsN, theta, u_np1_high, u_np1_high, make_weak_form,
                                        bcs, nullspace, solver_parameters, appctx, W)

    # Initial time step; the value does not really matter
    dt = T/2

    # Set initial conditions
    u_n.interpolate(u0)

    # Perform timestepping
    t = 0
    outfile = VTKFile("soln_adaptive.pvd")
    while t < T:

        # Report some numbers
        energy = assemble(inner(u_n.sub(0), u_n.sub(0)) * dx)
        print("{:10.4f} | {:10.4f} | {:#10.4g}".format(t, dt, energy))

        # Compute tentative time steps
        solver_low(t, dt)
        solver_high_1(t, dt/2)
        solver_high_2(t+dt, dt/2)

        # Compute error estimate and new timestep
        est = compute_est(theta, u_np1_low, u_np1_high)
        dt_new = compute_new_dt(theta, est, tol, dt)

        if est > tol:
            # Tolerance not met; repeat the step with new timestep
            dt = dt_new
            continue

        # Move to next time step
        u_n.vector()[:] = u_np1_high.vector()
        t += dt
        dt = dt_new

        # Write to file
        if W is None:
            outfile.write(u_n)
        else:
            outfile.write(u_n.sub(0), u_n.sub(1))