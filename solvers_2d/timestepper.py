from firedrake import *

from .create_timestep_solver import create_timestep_solver
from .printoff import iter_info_verbose, text, green

def timestepper(get_data, theta, Z, dsN, t0, T, dt, make_weak_form,
                bcs=None, nullspace=None, solver_parameters=None):
    """
    Generic theta-scheme timestepper for heat or Navier-Stokes using get_data(t).
    """

    # -------------
    # Setup problem
    # -------------

    # old and new solutions
    u_old = Function(Z)
    u_new = Function(Z)

    u_old = Function(Z)
    data0 = get_data(t0)
    for i, comp in enumerate(data0["ufl_u0"]):
        u_old.sub(i).interpolate(comp)

    # create timestep solver
    solver = create_timestep_solver(theta, Z, dsN, u_old, u_new,
                                    make_weak_form, get_data,
                                    bcs=bcs, nullspace=nullspace,
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
    outfile = VTKFile("soln_N.pvd")
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

        print(f"Step {step}, time = {t:.4f}")

    # Done
    green(f"Completed", spaced=True)