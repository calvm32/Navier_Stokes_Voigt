import numpy as np
from firedrake import COMM_WORLD

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def perturb_mesh(mesh, eps, comm):
    coords = mesh.coordinates.dat.data

    # global mesh size
    local_min = coords.min(axis=0)
    local_max = coords.max(axis=0)

    global_min = comm.allreduce(local_min, op=MPI.MIN)
    global_max = comm.allreduce(local_max, op=MPI.MAX)

    h = np.linalg.norm(global_max - global_min)

    # Generate random perturbation
    np.random.seed(0)
    perturb = np.random.randn(*coords.shape)

    # normalize perturbations
    norms = np.linalg.norm(perturb, axis=1)
    norms[norms == 0] = 1.0
    perturb = perturb / norms[:, None]

    coords[:] += eps * h * perturb # max displacement = eps * h

    return h

def barycentric_distortion(mesh, comm):
    coords = mesh.coordinates.dat.data
    cell_node_map = mesh.coordinates.cell_node_map().values

    local_max = 0.0

    for cell in cell_node_map:
        verts = coords[cell]

        # centroid
        centroid = np.mean(verts, axis=0)

        # measure distance from centroid
        dists = np.linalg.norm(verts - centroid, axis=1)

        # normalized variation
        if np.max(dists) > 0:
            val = (np.max(dists) - np.min(dists)) / np.max(dists)
            local_max = max(local_max, val)

    global_max = comm.allreduce(local_max, op=MPI.MAX)
    return global_max