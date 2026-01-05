from firedrake import *
from mpi4py import MPI

from .create_timestep_solver import create_timestep_solver
from .printoff import iter_info_verbose, text, green

def timestepper(get_data, theta, Z, dx , dsN, t0, T, dt, make_weak_form,
                bcs=None, nullspace=None, solver_parameters=None, appctx=None, vtkfile_name="Soln"):
    """
    Crank-Nicolson theta-scheme timestepper for velocity or velocity x pressure function spaces
    """

    num_steps = int((T-t0) / dt)

    # -------------
    # Setup problem
    # -------------

    # old and new solutions
    u_old = Function(Z)
    u = Function(Z)
    u_exact = Function(Z)

    data_t0 = get_data(t0) # get the functions at initial time

    if isinstance(Z.ufl_element(), MixedElement):
        u_old.sub(0).interpolate(data_t0["ufl_v0"])  # velocity
        u_old.sub(1).interpolate(data_t0["ufl_p0"])  # pressure

        # for L2 error
        v_error = 0
        p_error = 0
    else:
        u_old.interpolate(data_t0["ufl_u0"])  # just velocity

        # for L2 error
        u_error = 0

    # create timestep solver
    solver = create_timestep_solver(get_data, theta, Z, dx , dsN, u_old, u,
                                    make_weak_form, bcs=bcs, nullspace=nullspace,
                                    solver_parameters=solver_parameters, appctx=appctx)

    # get energy + report run starting
    if isinstance(Z.ufl_element(), MixedElement):
        energy = assemble(inner(u_old.sub(0), u_old.sub(0)) * dx)
    else:
        energy = assemble(inner(u_old, u_old) * dx)
    
    iter_info_verbose("INITIAL CONDITIONS", f"energy = {energy}", i=0, spaced=True)
    text(f"*** Beginning solve with step size {dt} ***", spaced=True)

    # --------------------
    # Perform timestepping
    # --------------------

    # initialize
    t = t0
    step = 0

    # initialize VTK
    outfile = VTKFile(f"{vtkfile_name}.pvd", comm=Z.mesh().comm)

    # rename
    if isinstance(Z.ufl_element(), MixedElement):
        u.sub(0).rename("Velocity")
        u.sub(1).rename("Pressure")

        u.sub(0).assign(u_old.sub(0))
        u.sub(1).assign(u_old.sub(1))
    else:
        u.rename("Temperature")

    u.assign(u_old)
    if isinstance(Z.ufl_element(), MixedElement):
        outfile.write(u_exact.sub(0), u_exact.sub(1), time=t)
    else:
        outfile.write(u, time=t)

    while t < T:

        # Perform time step
        solver(t, dt)
        t += dt
        u_old.assign(u)

        # count steps to print
        step += 1

        # get energy + report each time step
        if isinstance(Z.ufl_element(), MixedElement):
            energy = assemble(inner(u_old.sub(0), u_old.sub(0)) * dx)
        else:
            energy = assemble(inner(u_old, u_old) * dx)

        iter_info_verbose("TIME STEP COMPLETED", f"energy = {energy}", i=step, n=num_steps)

        data_t = get_data(t) # get the functions at current time

        # record L2 error at current time
        if isinstance(Z.ufl_element(), MixedElement):
            u_exact.sub(0).interpolate(data_t["ufl_v0"])  # velocity
            u_exact.sub(1).interpolate(data_t["ufl_p0"])  # pressure

            v_error += assemble(inner(u_exact.sub(0) - u.sub(0), u_exact.sub(0) - u.sub(0))*dx)*dt
            p_error += assemble(inner(u_exact.sub(1) - u.sub(1), u_exact.sub(1) - u.sub(1))*dx)*dt
        else:
            u_exact.interpolate(data_t["ufl_u0"])  # just velocity

            u_error += assemble(inner(u_exact - u, u_exact - u)*dx)*dt

        # write to VTK every 2 steps
        if step % 2 == 0:
            if isinstance(Z.ufl_element(), MixedElement):
                outfile.write(u_exact.sub(0), u_exact.sub(1), time=t)
            else:
                outfile.write(u, time=t)

    # ----------------------------------
    # Report done; find and return error
    # ----------------------------------

    # report completed
    print(f"\n")
    green(f"Completed", spaced=True)

    # Write error to file
    if isinstance(Z.ufl_element(), MixedElement):
        green(f"Final L2 Error (velocity) = {v_error:0.8e}", spaced=True)
        green(f"Final L2 Error (pressure) = {p_error:0.8e}", spaced=True)

        return(v_error, p_error) 

    else:
        green(f"Final L2 Error (temperature) = {u_error:0.8e}", spaced=True)

        return(u_error) 