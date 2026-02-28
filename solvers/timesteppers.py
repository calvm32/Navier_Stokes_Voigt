from firedrake import *
from mpi4py import MPI

from .create_timestep_solvers import *
from .printoff import iter_info_verbose, text, green

from solvers.statistics.pdf_sampler import pdf_sampler
from solvers.statistics.structure_funcs import structure_funcs
from solvers.statistics.energy_spectra import energy_spectra

def timestepper_CN(get_data, Z, dx , dsN, t0, T, dt, make_weak_form, sample_length=40, sample_height=10,
                bcs=None, nullspace=None, solver_parameters=None, appctx=None, vtkfile_name="Soln"):
    """
    Crank-Nicolson theta-scheme timestepper for velocity or velocity x pressure function spaces
    """

    num_steps = int((float(T)-float(t0)) / float(dt)) 
    is_mixed = isinstance(Z.ufl_element(), MixedElement)
    compute_every = 5
    start_sampling = 10

    # --------
    # Tracking
    # --------

    energy_list = []
    palinstrophy_list = []
    stream_func_list = []
    enstrophy_list = []
    every_time_list = []
    all_time_list = []
    energy_spec_list = []
    energy_spec_probe = []

    if is_mixed:
        v_error_list = []
        p_error_list = []
    else:
        u_error_list = []

    # -----------
    # Setup problem
    # -----------

    # old and new solutions
    u_old = Function(Z)
    u = Function(Z)
    u_exact = Function(Z)

    mesh = Z.mesh()
    pdfs = pdf_sampler(mesh)
    struct_func = structure_funcs(
        u_old.sub(0),
        mesh,
        r_max=0.25*min(sample_length,sample_height),
        nbins=30
    )

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
    solver = create_timestep_solver_CN(get_data, Z, dx , dsN, u_old, u,
                                    make_weak_form, is_mixed, bcs=bcs, nullspace=nullspace,
                                    solver_parameters=solver_parameters, appctx=appctx)

    # get energy + report run starting
    energy = sqrt(assemble(inner(u_old.sub(0), u_old.sub(0)) * dx))

    # energy spectra object
    energy_spec = energy_spectra(u_old.sub(0), mesh, nbins=40)

    # target physical location
    target = np.array(energy_spec_target)

    # DOF coordinates (parallel safe)
    V = Z.sub(0).collapse()
    coords = V.tabulate_dof_coordinates().reshape(-1, 2)

    local_min_dist = 1e20
    local_index = -1

    for i, xy in enumerate(coords):
        d = np.linalg.norm(xy - target)
        if d < local_min_dist:
            local_min_dist = d
            local_index = i

    global_min_dist = comm.allreduce(local_min_dist, op=MPI.MIN)

    if abs(local_min_dist - global_min_dist) < 1e-14:
        probe_dof = local_index

    else:
        probe_dof = -1

    print("Using probe DOF:", probe_dof)
    print("Probe location:", coords[probe_dof])

    iter_info_verbose("INITIAL CONDITIONS", f"energy = {energy}", i=0, spaced=True)
    text(f"*** Beginning solve with step size {dt} ***", spaced=True)
    
    # -------------------
    # setup stream function
    # -------------------

    if is_mixed:
        domain = mesh
        Vpsi = FunctionSpace(domain, "CG", 1)
        psi = Function(Vpsi)
        phi = TestFunction(Vpsi)
        psi_trial = TrialFunction(Vpsi)

        omega_f = Function(Vpsi, name="vorticity")

        a_psi = inner(grad(psi_trial), grad(phi)) * dx
        L_psi = omega_f * phi * dx

        bcs_psi = DirichletBC(Vpsi, 0.0, "on_boundary")

        problem_psi = LinearVariationalProblem(a_psi, L_psi, psi, bcs=bcs_psi)

        solver_psi = LinearVariationalSolver(
            problem_psi,
            solver_parameters={
                "ksp_type": "cg",
                "pc_type": "hypre",
                "ksp_rtol": 1e-8,
                "ksp_max_it": 50,
            },
        )

    # ----------------
    # Setup timestepping
    # ----------------

    # initialize
    t = t0
    step = 0

    all_time_list.append(t0)
    energy_list.append(energy)

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

    # ------------------
    # Perform timestepping
    # ------------------

    while t < T:

        # Perform time step
        solver(t, dt)
        t += dt
        u_old.assign(u)

        # count steps to print
        step += 1

        # -------
        # logging
        # -------

        # -------- energy --------
        energy = sqrt(assemble(inner(u_old.sub(0), u_old.sub(0)) * dx))
        energy_list.append(energy)
        all_time_list.append(t)

        iter_info_verbose("TIME STEP COMPLETED", f"energy = {energy}", i=step, n=num_steps)

        if (step % compute_every == 0) and (step >= start_sampling):
            every_time_list.append(t)
            
            if is_mixed:

                # -------- vorticity = curl(v) --------
                omega = curl(u_old.sub(0))
                omega_f.interpolate(omega)

                # -------- stream --------
                solver_psi.solve()
                stream_func_list.append(sqrt(assemble(inner(psi, psi) * dx)))

                # -------- palinstrophy --------
                palinstrophy_list.append(sqrt(assemble(inner(grad(omega_f), grad(omega_f)) * dx)))

                # -------- enstrophy --------
                enstrophy_list.append(sqrt(assemble(inner(omega_f, omega_f) * dx)))

                # -------- compute stats!!! --------
                pdfs.sample_velocity_x(u_old.sub(0))
                pdfs.sample_velocity_y(u_old.sub(0))
                pdfs.sample_vorticity(omega_f)
                
                struct_func.sample(nsamples_per_bin=20)

                # -------- energy spectrum --------
                k_vals, E_vals = energy_spec.compute()
                energy_spec_list.append((k_vals, E_vals))

                # -------- parallel safe --------
                comm = u.sub(0).function_space().mesh().comm
                local_val = np.zeros(2)

                if probe_dof != -1:
                    local_val[:] = u.sub(0).dat.data_ro[probe_dof]

                global_val = comm.allreduce(local_val, op=MPI.SUM)
                ux, uy = global_val

                energy_spec_probe.append([ux, uy])

            # -------- error --------
            # get data at current time
            data_new = get_data(t)
            
            if is_mixed:
                u_exact.sub(0).interpolate(data_new["ufl_v0"])  # velocity
                u_exact.sub(1).interpolate(data_new["ufl_p0"])  # pressure

                v_error_list.append(assemble(inner(u_exact.sub(0) - u.sub(0), u_exact.sub(0) - u.sub(0))*dx)*dt) 
                p_error_list.append(assemble(inner(grad(u_exact.sub(1)) - grad(u.sub(1)), 
                                    grad(u_exact.sub(1)) - grad(u.sub(1)))*dx)*dt)

            else:
                u_exact.interpolate(data_new["ufl_u0"])  # just velocity

                u_error_list.append(assemble(inner(u_exact - u, u_exact - u)*dx)*dt)

            # -------- solution --------
            if is_mixed:
                outfile.write(u.sub(0), u.sub(1), time=t)
            else:
                outfile.write(u, time=t)

    # --------------------
    # finish computing stats
    # --------------------

    velocity_x_vals, velocity_y_vals, omega_vals = pdfs.finalize()
    r_vals, S2 = struct_func.compute()

    # --------------------------------
    # Report done; find and return error
    # --------------------------------

    # report completed
    print(f"\n")
    green(f"Completed", spaced=True)

    # Return everything
    if is_mixed:
        return(v_error_list, p_error_list, palinstrophy_list, stream_func_list, 
        enstrophy_list, every_time_list, energy_list, all_time_list, velocity_x_vals,
        velocity_y_vals, omega_vals, r_vals, S2, energy_spec_list, energy_spec_probe)

    else:
        return(u_error_list, energy_list, all_time_list)


# ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========

# ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========

# ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========

# ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========

# ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========

# ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========


def timestepper_BDF2(get_data, Z, dx , dsN, t0, T, dt, make_weak_form_BDF2, make_weak_form_CN, sample_length, sample_height,
                bcs=None, nullspace=None, solver_parameters=None, appctx=None, vtkfile_name="Soln", energy_spec_target=[6.5,2.0]):
    """
    BDF2 timestepper for velocity or velocity x pressure function spaces
    """

    num_steps = int((float(T)-float(t0)) / float(dt)) 
    is_mixed = isinstance(Z.ufl_element(), MixedElement)
    compute_every = 5
    start_sampling = 10

    # --------
    # Tracking
    # --------

    energy_list = []
    palinstrophy_list = []
    stream_func_list = []
    enstrophy_list = []
    every_time_list = []
    all_time_list = []
    energy_spec_list = []
    energy_spec_probe = []

    if is_mixed:
        v_error_list = []
        p_error_list = []
    else:
        u_error_list = []

    # -----------
    # Setup problem
    # -----------

    # old and new solutions
    u_older = Function(Z)
    u_old = Function(Z)
    u = Function(Z)
    u_exact = Function(Z)

    mesh = Z.mesh()
    pdfs = pdf_sampler(mesh)
    struct_func = structure_funcs(
        u_old.sub(0),
        mesh,
        r_max=0.25*min(sample_length,sample_height),
        nbins=30
    )

    data_new0 = get_data(t0) # get the functions at initial time

    if is_mixed:
        u_old.sub(0).interpolate(data_new0["ufl_v0"])  # velocity
        u_old.sub(1).interpolate(data_new0["ufl_p0"])  # pressure
        u_older.assign(u_old)

        # for L2 error
        v_error = 0
        p_error = 0

    else:
        u_old.interpolate(data_new0["ufl_u0"])  # just velocity
        u_older.assign(u_old)

        # for L2 error
        u_error = 0

    # create timestep solvers
    solver_CN = create_timestep_solver_CN(get_data, Z, dx , dsN, u_old, u,
                                make_weak_form_CN, is_mixed, bcs=bcs, nullspace=nullspace,
                                solver_parameters=solver_parameters, appctx=appctx)
    solver = create_timestep_solver_BDF2(get_data, Z, dx , dsN, u_older, u_old, u,
                                make_weak_form_BDF2, is_mixed, bcs=bcs, nullspace=nullspace,
                                solver_parameters=solver_parameters, appctx=appctx)

    # get energy + report run starting
    energy = sqrt(assemble(inner(u_old.sub(0), u_old.sub(0)) * dx))

    # energy spectra object
    energy_spec = energy_spectra(u_old.sub(0), mesh, nbins=40)

    # target physical location
    target = np.array(energy_spec_target)

    # DOF coordinates (parallel safe)
    V = Z.sub(0).collapse()
    coords = V.tabulate_dof_coordinates().reshape(-1, 2)

    local_min_dist = 1e20
    local_index = -1

    for i, xy in enumerate(coords):
        d = np.linalg.norm(xy - target)
        if d < local_min_dist:
            local_min_dist = d
            local_index = i

    global_min_dist = comm.allreduce(local_min_dist, op=MPI.MIN)

    if abs(local_min_dist - global_min_dist) < 1e-14:
        probe_dof = local_index
    else:
        probe_dof = -1

    print("Using probe DOF:", probe_dof)
    print("Probe location:", coords[probe_dof])

    iter_info_verbose("INITIAL CONDITIONS", f"energy = {energy}", i=0, spaced=True)
    text(f"*** Beginning solve with step size {dt} ***", spaced=True)
    
    # -------------------
    # setup stream function
    # -------------------

    if is_mixed:
        domain = mesh
        Vpsi = FunctionSpace(domain, "CG", 1)
        psi = Function(Vpsi)
        phi = TestFunction(Vpsi)
        psi_trial = TrialFunction(Vpsi)

        omega_f = Function(Vpsi, name="vorticity")

        a_psi = inner(grad(psi_trial), grad(phi)) * dx
        L_psi = omega_f * phi * dx

        bcs_psi = DirichletBC(Vpsi, 0.0, "on_boundary")

        problem_psi = LinearVariationalProblem(a_psi, L_psi, psi, bcs=bcs_psi)

        solver_psi = LinearVariationalSolver(
            problem_psi,
            solver_parameters={
                "ksp_type": "cg",
                "pc_type": "hypre",
                "ksp_rtol": 1e-8,
                "ksp_max_it": 50,
            },
        )

    # ----------------
    # Setup timestepping
    # ----------------

    # initialize
    t = t0
    step = 0

    all_time_list.append(t0)
    energy_list.append(energy)

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

    # ------------------
    # Perform timestepping
    # ------------------

    while t < T:

        # Perform time step
        if step == 0:
            solver_CN(t, dt)
        else:
            solver(t,dt)
        t += dt
        u_older.assign(u_old)
        u_old.assign(u)

        # count steps to print
        step += 1

        # -------
        # logging
        # -------

        # -------- energy --------
        energy = sqrt(assemble(inner(u_old.sub(0), u_old.sub(0)) * dx))
        energy_list.append(energy)
        all_time_list.append(t)

        iter_info_verbose("TIME STEP COMPLETED", f"energy = {energy}", i=step, n=num_steps)

        if (step % compute_every == 0) and (step >= start_sampling):
            every_time_list.append(t)
            
            if is_mixed:

                # -------- vorticity = curl(v) --------
                omega = curl(u_old.sub(0))
                omega_f.interpolate(omega)

                # -------- stream --------
                solver_psi.solve()
                stream_func_list.append(sqrt(assemble(inner(psi, psi) * dx)))

                # -------- palinstrophy --------
                palinstrophy_list.append(sqrt(assemble(inner(grad(omega_f), grad(omega_f)) * dx)))

                # -------- enstrophy --------
                enstrophy_list.append(sqrt(assemble(inner(omega_f, omega_f) * dx)))

                # -------- compute stats!!! --------
                pdfs.sample_velocity_x(u_old.sub(0))
                pdfs.sample_velocity_y(u_old.sub(0))
                pdfs.sample_vorticity(omega_f)
                
                struct_func.sample(nsamples_per_bin=20)

                # -------- energy spectrum --------
                k_vals, E_vals = energy_spec.compute()
                energy_spec_list.append((k_vals, E_vals))

                # -------- parallel safe --------
                comm = u.sub(0).function_space().mesh().comm

                local_val = np.zeros(2)

                if probe_dof != -1:
                    local_val[:] = u.sub(0).dat.data_ro[probe_dof]

                global_val = comm.allreduce(local_val, op=MPI.SUM)
                ux, uy = global_val

                energy_spec_probe.append([ux, uy])
                
            # -------- error --------
            # get data at current time
            data_new = get_data(t)
            
            if is_mixed:
                u_exact.sub(0).interpolate(data_new["ufl_v0"])  # velocity
                u_exact.sub(1).interpolate(data_new["ufl_p0"])  # pressure

                v_error_list.append(assemble(inner(u_exact.sub(0) - u.sub(0), u_exact.sub(0) - u.sub(0))*dx)*dt) 
                p_error_list.append(assemble(inner(grad(u_exact.sub(1)) - grad(u.sub(1)), 
                                    grad(u_exact.sub(1)) - grad(u.sub(1)))*dx)*dt)

            else:
                u_exact.interpolate(data_new["ufl_u0"])  # just velocity

                u_error_list.append(assemble(inner(u_exact - u, u_exact - u)*dx)*dt)

            # -------- solution --------
            if is_mixed:
                outfile.write(u.sub(0), u.sub(1), time=t)
            else:
                outfile.write(u, time=t)

    # --------------------
    # finish computing stats
    # --------------------

    velocity_x_vals, velocity_y_vals, omega_vals = pdfs.finalize()
    r_vals, S2 = struct_func.compute()

    # --------------------------------
    # Report done; find and return error
    # --------------------------------

    # report completed
    print(f"\n")
    green(f"Completed", spaced=True)

    # Return everything
    if is_mixed:
        return(v_error_list, p_error_list, palinstrophy_list, stream_func_list, 
        enstrophy_list, every_time_list, energy_list, all_time_list, velocity_x_vals,
        velocity_y_vals, omega_vals, r_vals, S2, energy_spec_list, energy_spec_probe)

    else:
        return(u_error_list, energy_list, all_time_list)