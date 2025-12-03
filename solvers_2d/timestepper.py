from firedrake import *
from firedrake.functionspaceimpl import MixedFunctionSpace

from .create_timestep_solver import create_timestep_solver
from .printoff import iter_info_verbose, text, green

def timestepper(theta, Z, dsN, t, T, dt, make_weak_form, function_space_appctx, 
                bcs=None, nullspace=None, solver_parameters=None):
    """
    Perform timestepping using theta-scheme with
    final time T, timestep dt, initial datum u0
    """

    # -------------
    # Setup problem
    # -------------

    # Initialize solution function
    u_old = Function(Z)
    u_new = Function(Z)

    # initial condition
    if isinstance(Z.ufl_element(), MixedElement):
        ufl_v0 = function_space_appctx["ufl_v0"]
        ufl_p0 = function_space_appctx["ufl_p0"]
        u_old.sub(0).interpolate(ufl_v0)
        u_old.sub(1).interpolate(ufl_p0)
    
    else:
        ufl_u0 = function_space_appctx["ufl_u0"]
        u_old.interpolate(ufl_u0)

    # Prepare solver for computing time step
    solver = create_timestep_solver(theta, Z, dsN, u_old, u_new, make_weak_form,
                                    function_space_appctx, bcs, nullspace, solver_parameters)

    # Print table header
    energy = assemble(inner(u_old.sub(0), u_old.sub(0)) * dx)
    iter_info_verbose("INITIAL CONDITIONS", f"energy = {energy}", i=0, spaced=True)

    text(f"*** Beginning solve with step size {dt} ***", spaced=True)

    # --------------------
    # Perform timestepping
    # --------------------

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

        # -------------
        # Write to file
        # -------------
        if isinstance(Z.ufl_element(), MixedElement):
            outfile.write(u_new.sub(0), u_new.sub(1))
        else:
            outfile.write(u_new)

    # Done
    green(f"Completed", spaced=True)
