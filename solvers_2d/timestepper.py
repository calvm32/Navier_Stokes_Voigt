from firedrake import *

from .create_timestep_solver import create_timestep_solver
from .printoff import iter_info_verbose, text, green

def timestepper(get_data, theta, Z, dx , dSN, t0, T, dt, make_weak_form,
                bcs=None, nullspace=None, solver_parameters=None, vtkfile_name="Soln"):
    """
    Generic theta-scheme timestepper for heat or Navier-Stokes using get_data(t).
    """

    # -------------
    # Setup problem
    # -------------

    # old and new solutions
    u_old = Function(Z)
    u_new = Function(Z)
    u_exact = Function(Z)

    data_t0 = get_data(t0) # get the functions at initial time

    if isinstance(Z.ufl_element(), MixedElement):
        u_old.sub(0).interpolate(data_t0["ufl_v0"])  # velocity
        u_old.sub(1).interpolate(data_t0["ufl_p0"])  # pressure
    else:
        u_old.interpolate(data_t0["ufl_u0"])  # just velocity

    # create timestep solver
    solver = create_timestep_solver(get_data, theta, Z, dx , dSN, u_old, u_new,
                                    make_weak_form, bcs=bcs, nullspace=nullspace,
                                    solver_parameters=solver_parameters)
    
    # Print table header
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
        u_old.assign(u_new)

        # count steps to print
        step += 1

        # Report some numbers
        energy = assemble(inner(u_new.sub(0), u_new.sub(0)) * dx)
        iter_info_verbose("TIME STEP COMPLETED", f"energy = {energy}", i=step)

        # write to VTK
        if isinstance(Z.ufl_element(), MixedElement):
            outfile.write(u_new.sub(0), u_new.sub(1))
        else:
            outfile.write(u_new)

    # Done
    print(f"\n")
    green(f"Completed", spaced=True)

    data_T = get_data(T) # get the error at final time

    if isinstance(Z.ufl_element(), MixedElement):
        u_exact.sub(0).interpolate(data_T["ufl_v0"])  # velocity
        u_exact.sub(1).interpolate(data_T["ufl_p0"])  # pressure
    else:
        u_exact.interpolate(data_T["ufl_u0"])  # just velocity

    # Write FINAL error to file
    u_error = errornorm(u_exact.sub(0), u_new.sub(0))
    green(f"Final L2 Error = {u_error:0.8e}", spaced=True)

    return(u_error)