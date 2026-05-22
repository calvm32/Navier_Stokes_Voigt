from firedrake import *
from mpi4py import MPI
import time
from pathlib import Path
import numpy as np

from .create_timestep_solvers import *
from modules.processing.printoff import iter_info_verbose, text, green
from modules.processing.statistics.pdf_sampler_2d import pdf_sampler_2d
from modules.processing.statistics.structure_funcs_2d import structure_funcs_2d
from modules.processing.statistics.pdf_sampler_3d import pdf_sampler_3d
from modules.processing.statistics.structure_funcs_3d import structure_funcs_3d
from modules.processing.post_processing import *

def timestepper_CN(get_data, Z, dx , dsN, t0, T, dt, make_weak_form, theta, sample_xmax, sample_ymax, sample_zmax=None, gamma=None, Re=None, alpha=None,
                bcs=None, nullspace=None, solver_parameters=None, appctx=None, vtkfile_name="Soln", energy_spec_target=[6.2,4,0]):
    """
    Crank-Nicolson theta-scheme timestepper for velocity or velocity x pressure function spaces
    """

    num_steps = int(np.rint((T-t0)/dt))
    is_mixed = isinstance(Z.ufl_element(), MixedElement)
    compute_every = 5
    start_sampling = 10
    write_every = compute_every*20

    if num_steps <= start_sampling:
        start_sampling = 0

    # --------
    # Tracking
    # --------

    energy_list = []
    palinstrophy_list = []
    stream_func_list = []
    enstrophy_list = []
    every_time_list = []
    all_time_list = []
    energy_spec_probe = []
    cpu_time = 0

    if is_mixed:
        v_error_list = []
        p_error_list = []
        div_list = []
    else:
        u_error_list = []
    
    output_file1 = Path("plot_data.npz")
    output_file2 = Path("plot_final_data.npz")

    # -------------
    # Setup problem
    # -------------

    # old and new solutions
    u_old = Function(Z)
    u = Function(Z)
    u_exact = Function(Z)

    mesh = Z.mesh()
    cell = mesh.ufl_cell()
    dim = cell.topological_dimension() if callable(cell.topological_dimension) else cell.topological_dimension
    if dim == 2:
        pdfs = pdf_sampler_2d(mesh)
        struct_func = structure_funcs_2d(
            u_old.sub(0),
            mesh,
            r_max=0.25*min(sample_xmax,sample_ymax),
            nbins=30
        )
    elif dim == 3:
        pdfs = pdf_sampler_3d(mesh)
        struct_func = structure_funcs_3d(
            u_old.sub(0),
            mesh,
            r_max=0.25*min(sample_xmax,sample_ymax),
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
                                    make_weak_form, is_mixed, theta, gamma, Re, bcs=bcs, nullspace=nullspace,
                                    solver_parameters=solver_parameters, appctx=appctx)

    # get energy + report run starting
    energy = sqrt(assemble(inner(u_old.sub(0), u_old.sub(0)) * dx))

    # ------------------------------
    # Probe DOF selection (MPI safe)
    # ------------------------------

    # velocity function
    u_vel = u.sub(0)
    mesh = u_vel.function_space().mesh()
    comm = mesh.comm

    # mesh coordinates (nodal positions)
    if hasattr(mesh, 'coordinates'):
        coords = mesh.coordinates.dat.data_ro
    elif hasattr(mesh, 'meshes'):
        coords = mesh.meshes[0].coordinates.dat.data_ro
    else:
        raise AttributeError(f"Cannot extract coordinates from mesh of type {type(mesh)}").copy()  # shape (num_local_nodes, 2)

    # target location for probe
    if dim == 2:
        target = np.array([energy_spec_target[0], energy_spec_target[1]])
    elif dim ==3:
        target = np.array(energy_spec_target)

    # compute distance locally
    local_distances = np.linalg.norm(coords - target, axis=1)
    local_min_index = np.argmin(local_distances)
    local_min_dist = local_distances[local_min_index]

    # find global minimum across all ranks
    global_min_dist = comm.allreduce(local_min_dist, op=MPI.MIN)

    # determine which rank has the closest node
    if abs(local_min_dist - global_min_dist) < 1e-14:
        probe_dof = local_min_index
    else:
        probe_dof = -1

    # print probe info from the owning rank
    # if probe_dof != -1:
    #     print("Using probe DOF:", probe_dof)
    #     print("Probe location:", coords[probe_dof], "\n")

    # ---------------------
    # setup stream function
    # ---------------------

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

    # ------------------
    # Setup timestepping
    # ------------------

    # initialize
    t = t0
    step = 0

    all_time_list.append(t0)
    energy_list.append(energy)
    if is_mixed:
        div_list.append(sqrt(assemble(div(u.sub(0))**2 * dx)))

    # initialize VTK
    visfile = VTKFile(f"vis/{vtkfile_name}.pvd", comm=Z.mesh().comm)

    # rename
    if is_mixed:
        u.sub(0).rename("velocity")
        u.sub(1).rename("pressure")

        u.sub(0).assign(u_old.sub(0))
        u.sub(1).assign(u_old.sub(1))

        visfile.write(u.sub(0), u.sub(1), time=t)
    else:
        u.rename("temperature")
        u.assign(u_old)

        visfile.write(u, time=t)

    # --------------------
    # Perform timestepping
    # --------------------

    iter_info_verbose("INITIAL CONDITIONS", f"energy = {energy:.16f}", i=0, spaced=True)
    text(f"*** Beginning solve with step size {dt:.4f} ***", spaced=True)
    start = time.process_time()

    while step < num_steps:

        # Perform time-step
        solver(t, dt)
        t += dt
        u_old.assign(u)

        if is_mixed:
            div_list.append(sqrt(assemble(div(u.sub(0))**2 * dx)))

        # count steps to print
        step += 1

        # -------
        # logging
        # -------

        # -------- energy --------
        energy = sqrt(assemble(inner(u_old.sub(0), u_old.sub(0)) * dx))
        energy_list.append(energy)
        all_time_list.append(t)

        iter_info_verbose("TIME STEP COMPLETED", f"energy = {energy}", i=step, n=(num_steps))

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
                pdfs.sample_velocity(u_old.sub(0))
                pdfs.sample_vorticity(omega_f)
                
                struct_func.sample(nsamples_per_bin=20)

                # -------- energy spec probe --------
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

                v_error_list.append(assemble(inner(u_exact.sub(0) - u.sub(0), u_exact.sub(0) - u.sub(0))*dx)) 
                p_error_list.append(assemble(inner(grad(u_exact.sub(1)) - grad(u.sub(1)), 
                                    grad(u_exact.sub(1)) - grad(u.sub(1)))*dx))

            else:
                u_exact.interpolate(data_new["ufl_u0"])  # just velocity

                u_error_list.append(assemble(inner(u_exact - u, u_exact - u)*dx))

            # -------- solution --------
            if is_mixed:
                visfile.write(u.sub(0), u.sub(1), time=t)
            else:
                visfile.write(u, time=t)
            
        if (step % write_every == 0) and step > 0:
            if mesh.comm.rank == 0 and is_mixed:
                np.savez(
                    output_file1,

                    # time series
                    all_time=np.array(all_time_list),
                    energy=np.array(energy_list),
                    divergence=np.array(div_list),

                    every_time=np.array(every_time_list),
                    palinstrophy=np.array(palinstrophy_list),
                    stream_func=np.array(stream_func_list),
                    enstrophy=np.array(enstrophy_list),

                    # probe
                    probe=np.array(energy_spec_probe)
                )

            elif mesh.comm.rank == 0:
                np.savez(
                    output_file1,

                    # time series
                    all_time=np.array(all_time_list),
                    energy=np.array(energy_list),

                    every_time=np.array(every_time_list),
                    palinstrophy=np.array(palinstrophy_list),
                    stream_func=np.array(stream_func_list),
                    enstrophy=np.array(enstrophy_list),

                    # probe
                    probe=np.array(energy_spec_probe)
                )

    # ---------------
    # plot everything
    # ---------------

    if is_mixed:
        plot_ns()
    else:
        plot_heat()

    # ----------------------------------
    # Report done; find and return error
    # ----------------------------------

    end = time.process_time()
    cpu_time = (end - start) / 60

    # synchronize for finalization process
    comm = mesh.comm
    comm.Barrier()

    if mesh.comm.rank == 0 and is_mixed:
        print("donedonedone")
        if dim == 2:
            velocity_x_vals, velocity_y_vals, omega_vals = pdfs.finalize()
        elif dim == 3:
            velocity_x_vals, velocity_y_vals, velocity_z_vals, omega_vals = pdfs.finalize()
        r_vals, S2 = struct_func.compute()
        
        np.savez(
            output_file2,

            # time series
            all_time=np.array(all_time_list),
            energy=np.array(energy_list),
            divergence=np.array(div_list),

            every_time=np.array(every_time_list),
            palinstrophy=np.array(palinstrophy_list),
            stream_func=np.array(stream_func_list),
            enstrophy=np.array(enstrophy_list),

            # probe
            probe=np.array(energy_spec_probe),

            # rest of stats
            velocity_x=np.array(velocity_x_vals),
            velocity_y=np.array(velocity_y_vals),
            omega=np.array(omega_vals),
            r_vals=np.array(r_vals),
            S2=np.array(S2),
        )

    elif mesh.comm.rank == 0 and not mixed:
        np.savez(
            output_file2,

            # time series
            all_time=np.array(all_time_list),
            energy=np.array(energy_list),

            every_time=np.array(every_time_list),
            palinstrophy=np.array(palinstrophy_list),
            stream_func=np.array(stream_func_list),
            enstrophy=np.array(enstrophy_list),

            # probe
            probe=np.array(energy_spec_probe),
        )

    # report completed
    green(f"\nCompleted after {cpu_time} minutes", spaced=True)

    if is_mixed:
        return v_error_list, p_error_list
    else:
        return u_error_list


# ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========

# ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========

# ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========

# ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========

# ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========

# ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========


def timestepper_BDF2(get_data, Z, dx , dsN, t0, T, dt, make_weak_form_BDF2, make_weak_form_CN, sample_xmax, sample_ymax, sample_zmax=None, gamma=None, Re=None,
                alpha=None, bcs=None, nullspace=None, solver_parameters=None, appctx=None, vtkfile_name="Soln", energy_spec_target=[6.2,4,0]):
    """
    BDF2 timestepper for velocity or velocity x pressure function spaces
    """

    num_steps = int(np.rint((T-t0)/dt))
    is_mixed = isinstance(Z.ufl_element(), MixedElement)
    compute_every = 5
    start_sampling = 10
    write_every = compute_every*20
    plot_data = {}

    if num_steps <= start_sampling:
        start_sampling = 0

    # --------
    # Tracking
    # --------

    energy_list = []
    palinstrophy_list = []
    stream_func_list = []
    enstrophy_list = []
    every_time_list = []
    all_time_list = []
    energy_spec_probe = []
    cpu_time = 0

    if is_mixed:
        v_error_list = []
        p_error_list = []
        div_list = []
    else:
        u_error_list = []
    
    output_file1 = Path("plot_data.npz")
    output_file2 = Path("plot_final_data.npz")

    # -------------
    # Setup problem
    # -------------

    # old and new solutions
    u_older = Function(Z)
    u_old = Function(Z)
    u = Function(Z)
    u_exact = Function(Z)

    mesh = Z.mesh()
    cell = mesh.ufl_cell()
    dim = cell.topological_dimension() if callable(cell.topological_dimension) else cell.topological_dimension
    if dim == 2:
        pdfs = pdf_sampler_2d(mesh)
        struct_func = structure_funcs_2d(
            u_old.sub(0),
            mesh,
            r_max=0.25*min(sample_xmax,sample_ymax),
            nbins=30
        )
    elif dim == 3:
        pdfs = pdf_sampler_3d(mesh)
        struct_func = structure_funcs_3d(
            u_old.sub(0),
            mesh,
            r_max=0.25*min(sample_xmax,sample_ymax),
            nbins=30
        )

    data_old = get_data(t0)
    data_older = get_data(t0 - dt)
    if is_mixed:
        u_old.sub(0).interpolate(data_old["ufl_v0"])  # velocity
        u_old.sub(1).interpolate(data_old["ufl_p0"])  # pressure

        u_older.sub(0).interpolate(data_older["ufl_v0"])  # velocity
        u_older.sub(1).interpolate(data_older["ufl_p0"])  # pressure

        # for L2 error
        v_error = 0
        p_error = 0

    else:
        u_old.interpolate(data_old["ufl_u0"])  # just velocity
        u_older.assign(u_old)

        # for L2 error
        u_error = 0

    # create timestep solvers
    solver_CN = create_timestep_solver_CN(get_data, Z, dx , dsN, u_old, u,
                                make_weak_form_CN, is_mixed, 0.5, gamma, Re, alpha, bcs=bcs, nullspace=nullspace,
                                solver_parameters=solver_parameters, appctx=appctx)
    solver = create_timestep_solver_BDF2(get_data, Z, dx , dsN, u_older, u_old, u,
                                make_weak_form_BDF2, is_mixed, gamma, Re, alpha, bcs=bcs, nullspace=nullspace,
                                solver_parameters=solver_parameters, appctx=appctx)

    # get energy + report run starting
    energy = sqrt(assemble(inner(u_old.sub(0), u_old.sub(0)) * dx))

    # ------------------------------
    # Probe DOF selection (MPI safe)
    # ------------------------------

    # velocity function
    u_vel = u.sub(0)
    mesh = u_vel.function_space().mesh()
    comm = mesh.comm

    # mesh coordinates (nodal positions)
    if hasattr(mesh, 'coordinates'):
        coords = mesh.coordinates.dat.data_ro
    elif hasattr(mesh, 'meshes'):
        coords = mesh.meshes[0].coordinates.dat.data_ro
    else:
        raise AttributeError(f"Cannot extract coordinates from mesh of type {type(mesh)}").copy()  # shape (num_local_nodes, 2)

    # target location for probe
    if dim == 2:
        target = np.array([energy_spec_target[0], energy_spec_target[1]])
    elif dim ==3:
        target = np.array(energy_spec_target)

    # compute distance locally
    local_distances = np.linalg.norm(coords - target, axis=1)
    local_min_index = np.argmin(local_distances)
    local_min_dist = local_distances[local_min_index]

    # find global minimum across all ranks
    global_min_dist = comm.allreduce(local_min_dist, op=MPI.MIN)

    # determine which rank has the closest node
    if abs(local_min_dist - global_min_dist) < 1e-14:
        probe_dof = local_min_index
    else:
        probe_dof = -1

    # print probe info from the owning rank
    # if probe_dof != -1:
    #     print("Using probe DOF:", probe_dof)
    #     print("Probe location:", coords[probe_dof], "\n")
    
    # ---------------------
    # setup stream function
    # ---------------------

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

    # ------------------
    # Setup timestepping
    # ------------------

    # initialize
    t = t0
    step = 0

    all_time_list.append(t0)
    energy_list.append(energy)
    if is_mixed:
        div_list.append(sqrt(assemble(div(u.sub(0))**2 * dx)))

    # initialize VTK
    visfile = VTKFile(f"vis/{vtkfile_name}.pvd", comm=Z.mesh().comm)

    # rename
    if is_mixed:
        u.sub(0).rename("velocity")
        u.sub(1).rename("pressure")

        u.sub(0).assign(u_old.sub(0))
        u.sub(1).assign(u_old.sub(1))

        visfile.write(u.sub(0), u.sub(1), time=t)
    else:
        u.rename("temperature")
        u.assign(u_old)

        visfile.write(u, time=t)

    iter_info_verbose("INITIAL CONDITIONS", f"energy = {energy}", i=0, spaced=True)
    text(f"*** Beginning solve with step size {dt:.4f} ***", spaced=True)
    start = time.process_time()

    # --------------------
    # Perform timestepping
    # --------------------

    while step < num_steps:

        # Perform time-step
        if step == 0:
            solver_CN(t, dt)
        else:
            solver(t,dt)
        t += dt
        u_older.assign(u_old)
        u_old.assign(u)

        if is_mixed:
            div_list.append(sqrt(assemble(div(u.sub(0))**2 * dx)))

        # count steps to print
        step += 1

        # -------
        # logging
        # -------

        # -------- energy --------
        energy = sqrt(assemble(inner(u_old.sub(0), u_old.sub(0)) * dx))
        energy_list.append(energy)
        all_time_list.append(t)

        iter_info_verbose("TIME STEP COMPLETED", f"energy = {energy}", i=step, n=(num_steps))

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
                pdfs.sample_velocity(u_old.sub(0))
                pdfs.sample_vorticity(omega_f)
                
                struct_func.sample(nsamples_per_bin=20)

                # -------- energy spec probe --------
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

                v_error_list.append(assemble(inner(u_exact.sub(0) - u.sub(0), u_exact.sub(0) - u.sub(0))*dx)) 
                p_error_list.append(assemble(inner(grad(u_exact.sub(1)) - grad(u.sub(1)), 
                                    grad(u_exact.sub(1)) - grad(u.sub(1)))*dx))

            else:
                u_exact.interpolate(data_new["ufl_u0"])  # just velocity

                u_error_list.append(assemble(inner(u_exact - u, u_exact - u)*dx))

            # -------- solution --------
            if is_mixed:
                visfile.write(u.sub(0), u.sub(1), time=t)
            else:
                visfile.write(u, time=t)

        if (step % write_every == 0) and step > 0:
            if mesh.comm.rank == 0 and is_mixed:
                np.savez(
                    output_file1,

                    # time series
                    all_time=np.array(all_time_list),
                    energy=np.array(energy_list),
                    divergence=np.array(div_list),

                    every_time=np.array(every_time_list),
                    palinstrophy=np.array(palinstrophy_list),
                    stream_func=np.array(stream_func_list),
                    enstrophy=np.array(enstrophy_list),

                    # probe
                    probe=np.array(energy_spec_probe)
                )

            elif mesh.comm.rank == 0 and not mixed:
                np.savez(
                    output_file1,

                    # time series
                    all_time=np.array(all_time_list),
                    energy=np.array(energy_list),

                    every_time=np.array(every_time_list),
                    palinstrophy=np.array(palinstrophy_list),
                    stream_func=np.array(stream_func_list),
                    enstrophy=np.array(enstrophy_list),

                    # probe
                    probe=np.array(energy_spec_probe)
                )

    # ----------------------------------
    # Report done; find and return error
    # ----------------------------------
    end = time.process_time()
    cpu_time = (end - start) / 60

    # synchronize for finalization process
    comm = mesh.comm
    comm.Barrier()

    if mesh.comm.rank == 0 and is_mixed:

        if dim == 2:
            velocity_x_vals, velocity_y_vals, omega_vals = pdfs.finalize()
        elif dim == 3:
            velocity_x_vals, velocity_y_vals, velocity_z_vals, omega_vals = pdfs.finalize()
        r_vals, S2 = struct_func.compute()
        print(velocity_x_vals.shape)
        print(velocity_y_vals.shape)
        print(omega_vals.shape)
        
        np.savez(
            output_file2,

            # time series
            all_time=np.array(all_time_list),
            energy=np.array(energy_list),
            divergence=np.array(div_list),

            every_time=np.array(every_time_list),
            palinstrophy=np.array(palinstrophy_list),
            stream_func=np.array(stream_func_list),
            enstrophy=np.array(enstrophy_list),

            # probe
            probe=np.array(energy_spec_probe),

            # rest of stats
            velocity_x=np.array(velocity_x_vals),
            velocity_y=np.array(velocity_y_vals),
            omega=np.array(omega_vals),
            r_vals=np.array(r_vals),
            S2=np.array(S2),
        )

        print("done saving to file")

    elif mesh.comm.rank == 0 and not mixed:
        np.savez(
            output_file2,

            # time series
            all_time=np.array(all_time_list),
            energy=np.array(energy_list),

            every_time=np.array(every_time_list),
            palinstrophy=np.array(palinstrophy_list),
            stream_func=np.array(stream_func_list),
            enstrophy=np.array(enstrophy_list),

            # probe
            probe=np.array(energy_spec_probe),
        )

    # report completed
    green(f"\nCompleted after {cpu_time} minutes", spaced=True)

    if is_mixed:
        return v_error_list, p_error_list
    else:
        return u_error_list
        

# ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========

# ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========

# ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========

# ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========

# ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========

# ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========


def timestepper_BDF2_compare(get_data, Z, dx , dsN, t0, T, dt, make_weak_form_NSE_BDF2, make_weak_form_NSV_BDF2, 
                make_weak_form_NSE_CN, make_weak_form_NSV_CN, sample_xmax, sample_ymax, sample_zmax=None, gamma=None, Re=None,
                alpha=None, bcs=None, nullspace=None, solver_parameters=None, appctx=None, vtkfile_name="Soln"):
    """
    BDF2 timestepper for velocity or velocity x pressure function spaces
    """

    num_steps = int(np.rint((T-t0)/dt))
    is_mixed = isinstance(Z.ufl_element(), MixedElement)
    compute_every = 5
    start_sampling = 10
    write_every = compute_every*20
    plot_data = {}

    if num_steps <= start_sampling:
        start_sampling = 0

    # --------
    # Tracking
    # --------

    energy_diff_list = []
    palinstrophy_diff_list = []
    stream_func_diff_list = []
    enstrophy_diff_list = []
    every_time_list = []
    all_time_list = []
    cpu_time = 0

    v_diff_list = []
    omega_diff_list = []
    
    output_file1 = Path("plot_data.npz")
    output_file2 = Path("plot_final_data.npz")

    # -------------
    # Setup problem
    # -------------

    # old and new solutions
    u_older_NSE = Function(Z)
    u_old_NSE = Function(Z)
    u_NSE = Function(Z)
    u_exact_NSE = Function(Z)

    u_older_NSV = Function(Z)
    u_old_NSV = Function(Z)
    u_NSV = Function(Z)
    u_exact_NSV = Function(Z)

    mesh = Z.mesh()

    data_old = get_data(t0)
    data_older = get_data(t0 - dt)

    # FIRST NSE
    u_old_NSE.sub(0).interpolate(data_old["ufl_v0"])  # velocity
    u_old_NSE.sub(1).interpolate(data_old["ufl_p0"])  # pressure
    u_older_NSE.sub(0).interpolate(data_older["ufl_v0"])  # velocity
    u_older_NSE.sub(1).interpolate(data_older["ufl_p0"])  # pressure

    # NEXT NSV
    u_old_NSV.sub(0).interpolate(data_old["ufl_v0"])  # velocity
    u_old_NSV.sub(1).interpolate(data_old["ufl_p0"])  # pressure
    u_older_NSV.sub(0).interpolate(data_older["ufl_v0"])  # velocity
    u_older_NSV.sub(1).interpolate(data_older["ufl_p0"])  # pressure

    # create timestep solvers
    solver_NSE_CN = create_timestep_solver_CN(get_data, Z, dx , dsN, u_old_NSE, u_NSE,
                                make_weak_form_NSE_CN, is_mixed, 1.0, gamma, Re, alpha, bcs=bcs, nullspace=nullspace,
                                solver_parameters=solver_parameters, appctx=appctx)
    solver_NSE = create_timestep_solver_BDF2(get_data, Z, dx , dsN, u_older_NSE, u_old_NSE, u_NSE,
                                make_weak_form_NSE_BDF2, is_mixed, gamma, Re, alpha, bcs=bcs, nullspace=nullspace,
                                solver_parameters=solver_parameters, appctx=appctx)
    solver_NSV_CN = create_timestep_solver_CN(get_data, Z, dx , dsN, u_old_NSV, u_NSV,
                                make_weak_form_NSV_CN, is_mixed, 1.0, gamma, Re, alpha, bcs=bcs, nullspace=nullspace,
                                solver_parameters=solver_parameters, appctx=appctx)
    solver_NSV = create_timestep_solver_BDF2(get_data, Z, dx , dsN, u_older_NSV, u_old_NSV, u_NSV,
                                make_weak_form_NSV_BDF2, is_mixed, gamma, Re, alpha, bcs=bcs, nullspace=nullspace,
                                solver_parameters=solver_parameters, appctx=appctx)

    # get energy + report run starting
    energy_diff = abs(sqrt(assemble(inner(u_old_NSE.sub(0) - u_old_NSV.sub(0), u_old_NSE.sub(0) - u_old_NSV.sub(0)) * dx)))
    
    # ---------------------
    # setup stream function
    # ---------------------

    domain = mesh

    # FIRST NSE
    Vpsi_NSE = FunctionSpace(domain, "CG", 1)
    psi_NSE = Function(Vpsi_NSE)
    phi_NSE = TestFunction(Vpsi_NSE)
    psi_trial_NSE = TrialFunction(Vpsi_NSE)

    omega_f_NSE = Function(Vpsi_NSE, name="vorticity")

    a_psi_NSE = inner(grad(psi_trial_NSE), grad(phi_NSE)) * dx
    L_psi_NSE = omega_f_NSE * phi_NSE * dx

    bcs_psi_NSE = DirichletBC(Vpsi_NSE, 0.0, "on_boundary")

    problem_psi_NSE = LinearVariationalProblem(a_psi_NSE, L_psi_NSE, psi_NSE, bcs=bcs_psi_NSE)

    solver_psi_NSE = LinearVariationalSolver(
        problem_psi_NSE,
        solver_parameters={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1e-8,
            "ksp_max_it": 50,
        },
    )

    # NEXT NSV
    Vpsi_NSV = FunctionSpace(domain, "CG", 1)
    psi_NSV = Function(Vpsi_NSV)
    phi_NSV = TestFunction(Vpsi_NSV)
    psi_trial_NSV = TrialFunction(Vpsi_NSV)

    omega_f_NSV = Function(Vpsi_NSV, name="vorticity")

    a_psi_NSV = inner(grad(psi_trial_NSV), grad(phi_NSV)) * dx
    L_psi_NSV = omega_f_NSV * phi_NSV * dx

    bcs_psi_NSV = DirichletBC(Vpsi_NSV, 0.0, "on_boundary")

    problem_psi_NSV = LinearVariationalProblem(a_psi_NSV, L_psi_NSV, psi_NSV, bcs=bcs_psi_NSV)

    solver_psi_NSV = LinearVariationalSolver(
        problem_psi_NSV,
        solver_parameters={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1e-8,
            "ksp_max_it": 50,
        },
    )

    # ------------------
    # Setup timestepping
    # ------------------

    # initialize
    t = t0
    step = 0

    all_time_list.append(t0)
    energy_diff_list.append(energy_diff)

    # initialize VTK
    visfile = VTKFile(f"vis/{vtkfile_name}.pvd", comm=Z.mesh().comm)

    # rename FIRST NSE
    u_NSE.sub(0).rename("velocity_NSE")
    u_NSE.sub(1).rename("pressure_NSE")

    u_NSE.sub(0).assign(u_old_NSE.sub(0))
    u_NSE.sub(1).assign(u_old_NSE.sub(1))

    # rename NEXT NSV
    u_NSV.sub(0).rename("velocity_NSV")
    u_NSV.sub(1).rename("pressure_NSV")

    u_NSV.sub(0).assign(u_old_NSV.sub(0))
    u_NSV.sub(1).assign(u_old_NSV.sub(1))

    visfile.write(u_NSE.sub(0), u_NSE.sub(1), u_NSV.sub(0), u_NSV.sub(1), time=t)

    iter_info_verbose("INITIAL CONDITIONS", f"energy diff = {energy_diff}", i=0, spaced=True)
    text(f"*** Beginning solve with step size {dt:.4f} ***", spaced=True)
    start = time.process_time()

    # --------------------
    # Perform timestepping
    # --------------------

    while step < num_steps:

        # Perform time-step
        if step == 0:
            solver_NSE_CN(t, dt)
            solver_NSV_CN(t, dt)
        else:
            solver_NSE(t,dt)
            solver_NSV(t,dt)
        t += dt
        u_older_NSE.assign(u_old_NSE)
        u_old_NSE.assign(u_NSE)
        u_older_NSV.assign(u_old_NSV)
        u_old_NSV.assign(u_NSV)

        # count steps to print
        step += 1

        # -------
        # logging
        # -------

        # -------- energy --------
        energy_diff = abs(sqrt(assemble(inner(u_old_NSE.sub(0) - u_old_NSV.sub(0), u_old_NSE.sub(0) - u_old_NSV.sub(0)) * dx)))
        energy_diff_list.append(energy_diff)
        all_time_list.append(t)

        iter_info_verbose("TIME STEP COMPLETED", f"energy diff = {energy_diff}", i=step, n=(num_steps))

        if (step % compute_every == 0) and (step >= start_sampling):
            every_time_list.append(t)
            
            if is_mixed:

                # -------- vorticity = curl(v) --------
                omega_NSE = curl(u_old_NSE.sub(0))
                omega_f_NSE.interpolate(omega_NSE)
                omega_NSV = curl(u_old_NSV.sub(0))
                omega_f_NSV.interpolate(omega_NSV)
                omega_diff_list.append(sqrt(assemble((omega_NSE - omega_NSV)**2 * dx)))

                # -------- stream --------
                solver_psi_NSE.solve()
                solver_psi_NSV.solve()
                stream_func_diff_list.append(sqrt(assemble(inner(psi_NSE - psi_NSV, psi_NSE - psi_NSV) * dx)))

                # -------- palinstrophy --------
                palinstrophy_diff_list.append(sqrt(assemble(inner(grad(omega_f_NSE) - grad(omega_f_NSV), grad(omega_f_NSE) - grad(omega_f_NSV)) * dx)))

                # -------- enstrophy --------
                enstrophy_diff_list.append(sqrt(assemble(inner(omega_f_NSE - omega_f_NSV, omega_f_NSE - omega_f_NSV) * dx)))

                
            # -------- solution difference --------
            # get data at current time
            data_new = get_data(t)

            v_diff_list.append(sqrt(assemble(inner(u_NSE.sub(0) - u_NSV.sub(0), u_NSE.sub(0) - u_NSV.sub(0))*dx)))

            # -------- solution --------
            visfile.write(u_NSE.sub(0), u_NSE.sub(1), u_NSV.sub(0), u_NSV.sub(1), time=t)

        if (step % write_every == 0) and step > 0:
            if mesh.comm.rank == 0:

                np.savez(
                    output_file1,

                    # time series
                    all_time=np.array(all_time_list),
                    energy=np.array(energy_diff_list),

                    every_time=np.array(every_time_list),
                    palinstrophy=np.array(palinstrophy_diff_list),
                    enstrophy=np.array(enstrophy_diff_list),
                )

    # ----------------------------------
    # Report done; find and return error
    # ----------------------------------
    end = time.process_time()
    cpu_time = (end - start) / 60

    # synchronize for finalization process
    comm = mesh.comm
    comm.Barrier()

    # report completed
    green(f"\nCompleted after {cpu_time} minutes", spaced=True)

    return omega_diff_list, v_diff_list