from firedrake import *

from .create_timestep_solver import create_timestep_solver
from .printoff import iter_info_verbose, text, green

def timestepper(get_data, theta, Z, dx , dsN, t0, T, dt, make_weak_form,
                bcs=None, nullspace=None, solver_parameters=None, appctx=None, vtkfile_name="Soln"):
    """
    Crank-Nicolson theta-scheme timestepper for velocity or velocity x pressure function spaces
    """

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
        v_error = 0

    # create timestep solver
    solver = create_timestep_solver(get_data, theta, Z, dx , dsN, u_old, u,
                                    make_weak_form, bcs=bcs, nullspace=nullspace,
                                    solver_parameters=solver_parameters, appctx=appctx)
    
    # report run starting
    energy = assemble(inner(u_old.sub(0), u_old.sub(0)) * dx)
    iter_info_verbose("INITIAL CONDITIONS", f"energy = {energy}", i=0, spaced=True)
    text(f"*** Beginning solve with step size {dt} ***", spaced=True)

    # --------------------
    # Perform timestepping
    # --------------------

    t = t0
    step = 0
    outfile = VTKFile(f"{vtkfile_name}.pvd")
    while t < T:

        # Perform time step
        solver(t, dt)
        t += dt
        u_old.assign(u)

        # count steps to print
        step += 1

        # update solution
        u.assign(u_old) 

        # Report each time step
        energy = assemble(inner(u.sub(0), u.sub(0)) * dx)
        iter_info_verbose("TIME STEP COMPLETED", f"energy = {energy}", i=step)

        # record L2 error at current time
        if isinstance(Z.ufl_element(), MixedElement):
            u_exact.sub(0).interpolate(data_T["ufl_v0"])  # velocity
            u_exact.sub(1).interpolate(data_T["ufl_p0"])  # pressure

            v_error += assemble(inner(u_exact.sub(0) - u.sub(0), u_exact.sub(0) - u.sub(0))*dt)
            p_error += assemble(inner(u_exact.sub(1) - u.sub(1), u_exact.sub(1) - u.sub(1))*dt)
        else:
            u_exact.interpolate(data_T["ufl_u0"])  # just velocity

            error += assemble(inner(u_exact.sub(0) - u.sub(0), u_exact.sub(0) - u.sub(0))*dt)

        # write to VTK every 50 steps
        if step % 50 == 0:
            if isinstance(Z.ufl_element(), MixedElement):
                u.sub(0).rename("Velocity")
                u.sub(1).rename("Pressure")
                outfile.write(u.sub(0), u.sub(1))
            else:
                u.rename("Velocity")
                outfile.write(u)

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
        u_error = errornorm(u_exact.sub(0), u.sub(0)) # make time integral
        green(f"Final L2 Error (temperature) = {u_error:0.8e}", spaced=True)

        return(u_error) 
