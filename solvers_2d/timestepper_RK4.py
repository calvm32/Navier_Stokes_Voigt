from firedrake import *

from .printoff import iter_info_verbose, text, green
from .create_stage_solver import create_stage_solver

def timestepper_RK4(get_data, Z, dx, dsN, t0, T, dt,
                    make_residual,
                    bcs=None, solver_parameters=None,
                    vtkfile_name="Soln"):
    """
    Fourth-order Runge–Kutta timestepper for scalar problems
    (tested on heat equation)
    """

    # -------------
    # Setup problem
    # -------------

    u = Function(Z)
    u_trial = TrialFunction(Z)
    u_exact = Function(Z)
    u_stage = Function(Z)
    v = TestFunction(Z)

    # for each RK4 stage
    k1 = Function(Z)
    k2 = Function(Z)
    k3 = Function(Z)
    k4 = Function(Z)

    k = Function(Z, name="k")  # scratch stage variable

    # initial condition
    data_t0 = get_data(t0)

    if isinstance(Z.ufl_element(), MixedElement):
        u.sub(0).interpolate(data_t0["ufl_v0"])  # velocity
        u.sub(1).interpolate(data_t0["ufl_p0"])  # pressure
    else:
        u.interpolate(data_t0["ufl_u0"])  # just velocity

    # create timestep solver to be used at each stage
    solver, update_data = create_stage_solver(
        get_data, Z, dx, dsN,
        u_stage, k1,
        make_residual,
        bcs=bcs, solver_parameters=solver_parameters
    )

    # report run starting
    energy = assemble(u*u*dx)
    iter_info_verbose("INITIAL CONDITIONS", f"energy = {energy}", i=0, spaced=True)
    text(f"*** Beginning solve with step size {dt} ***", spaced=True)
    
    # --------------------
    # Perform timestepping
    # --------------------

    t = t0
    step = 0
    outfile = VTKFile(f"{vtkfile_name}.pvd")
    while t < T:

        # stage 1
        u_stage.assign(u)
        update_data(t)
        solver.solve()
        k1.assign(k)

        # stage 2
        u_stage.assign(u + 0.5*dt*k1)
        update_data(t + 0.5*dt)
        solver.solve()
        k2.assign(k)

        # stage 3
        u_stage.assign(u + 0.5*dt*k2)
        update_data(t + 0.5*dt)
        solver.solve()
        k3.assign(k)

        # stage 4
        u_stage.assign(u + dt*k3)
        update_data(t + dt)
        solver.solve()
        k4.assign(k)

        # count steps to print
        t += dt
        step += 1

        # Report each time step
        iter_info_verbose("TIME STEP COMPLETED", f"energy = {energy}", i=step)

        # update solution
        u.assign(u + dt/6*(k1 + 2*k2 + 2*k3 + k4))
        energy = assemble(inner(u.sub(0), u.sub(0)) * dx)

        # write to VTK every 50 steps
        if step % 50 == 0:
            if isinstance(Z.ufl_element(), MixedElement):
                u.sub(0).rename("Velocity")
                u.sub(1).rename("Pressure")
                outfile.write(u.sub(0), u.sub(1))
            else:
                u.rename("Temperature")
                outfile.write(u)

    # --------------------
    # Final error analysis
    # --------------------

    data_T = get_data(T)
    u_exact.interpolate(data_T["ufl_u0"])

    u_error = errornorm(u_exact, u)
    green(f"Final L2 Error = {u_error:0.8e}", spaced=True)

    return u_error