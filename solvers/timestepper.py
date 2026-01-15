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
    is_mixed = isinstance(Z.ufl_element(), MixedElement)

    # --------
    # Tracking
    # --------

    energy_list = []
    palenstrophy_list = []
    stream_func_list = []
    vorticity_list = []
    enstrophy_list = []

    if is_mixed:
        v_error_list = []
        p_error_list = []
    else:
        u_error_list = []

    # -------------
    # Setup problem
    # -------------

    # old and new solutions
    u_old = Function(Z)
    u = Function(Z)
    u_exact = Function(Z)

    data_new0 = get_data(t0) # get the functions at initial time

    if is_mixed:
        u_old.sub(0).interpolate(data_new0["ufl_v0"])  # velocity
        u_old.sub(1).interpolate(data_new0["ufl_p0"])  # pressure

        # for L2 error
        v_error = 0
        p_error = 0

    else:
        u_old.interpolate(data_new0["ufl_u0"])  # just velocity

        # for L2 error
        u_error = 0

    # create timestep solver
    solver = create_timestep_solver(get_data, theta, Z, dx , dsN, u_old, u,
                                    make_weak_form, is_mixed, bcs=bcs, nullspace=nullspace,
                                    solver_parameters=solver_parameters, appctx=appctx)

    # get energy + report run starting
    if is_mixed:
        energy = assemble(inner(u_old.sub(0), u_old.sub(0)) * dx)
    else:
        energy = assemble(inner(u_old, u_old) * dx)
    
    iter_info_verbose("INITIAL CONDITIONS", f"energy = {energy}", i=0, spaced=True)
    text(f"*** Beginning solve with step size {dt} ***", spaced=True)

    # ------------------
    # Setup timestepping
    # ------------------

    # initialize
    t = t0
    step = 0

    # initialize VTK
    outfile = VTKFile(f"{vtkfile_name}.pvd", comm=Z.mesh().comm)

    # rename
    if is_mixed:
        u.sub(0).rename("velocity")
        u.sub(1).rename("pressure")

        u.sub(0).assign(u_old.sub(0))
        u.sub(1).assign(u_old.sub(1))

        outfile.write(u.sub(0), u.sub(1), time=t)
    else:
        u.rename("temperature")
        u.assign(u_old)

        outfile.write(u, time=t)

    # --------------------
    # Perform timestepping
    # --------------------

    while t < T:

        # Perform time step
        solver(t, dt)
        t += dt
        u_old.assign(u)

        # count steps to print
        step += 1

        # --------- energy ---------
        energy_list.append(assemble(inner(u_old.sub(0), u_old.sub(0)) * dx))

        # --------- stream ---------
        stream_func_list.append("")

        # --------- vorticity = curl(v) ---------
        omega = curl(u_old.sub(0))
        vorticity_list.append(omega)

        # --------- palenstrophy ---------
        palinstrophy_list.append(assemble(0.5 * (div(grad(omega)))**2 * dx))

        # --------- enstrophy ---------
        enstrophy_list.append(assemble(0.5 * omega**2 * dx))

        # --------- error ---------
        if is_mixed:
            u_exact.sub(0).interpolate(data_new["ufl_v0"])  # velocity
            u_exact.sub(1).interpolate(data_new["ufl_p0"])  # pressure

            v_error_list.append(assemble(inner(u_exact.sub(0) - u.sub(0), u_exact.sub(0) - u.sub(0))*dx)*dt) 
            p_error_list.append(assemble(inner(grad(u_exact.sub(0)) - grad(u.sub(0)), 
                                grad(u_exact.sub(0)) - grad(u.sub(0)))*dx)*dt)

        else:
            u_exact.interpolate(data_new["ufl_u0"])  # just velocity

            u_error_list.append(assemble(inner(u_exact - u, u_exact - u)*dx)*dt)

        # --------- solution ---------
        if step % 2 == 0:
            if is_mixed:
                outfile.write(u.sub(0), u.sub(1), time=t)
            else:
                outfile.write(u, time=t)

        iter_info_verbose("TIME STEP COMPLETED", f"energy = {energy}", i=step, n=num_steps)

        # get data at current time
        data_new = get_data(t)

    # ----------------------------------
    # Report done; find and return error
    # ----------------------------------

    # report completed
    print(f"\n")
    green(f"Completed", spaced=True)

    # Write error to file
    if is_mixed:
        return(v_error_list, p_error_list, palenstrophy_list, stream_func_list, vorticity_list, enstrophy_list)

    else:
        return(u_error_list, palenstrophy_list, stream_func_list, vorticity_list, enstrophy_list)