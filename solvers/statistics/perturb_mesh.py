import numpy as np
from firedrake import COMM_WORLD

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def find_bary_nodes(mesh, tol=1e-10):
    coords = mesh.coordinates.dat.data_with_halos
    cell_node_map = mesh.coordinates.cell_node_map().values

    bary_nodes = set()

    for cell in cell_node_map:
        verts = coords[cell]

        centroid = np.mean(verts, axis=0)

        # find node closest to centroid
        dists = np.linalg.norm(verts - centroid, axis=1)
        idx = np.argmin(dists)

        if dists[idx] < tol:
            bary_nodes.add(cell[idx])

    return np.array(list(bary_nodes), dtype=int)

def perturb_bary(mesh, eps, comm):
    coords = mesh.coordinates.dat.data_with_halos

    bary_nodes = find_bary_nodes(mesh)

    # compute global length scale h
    local_xmin = coords[:, 0].min()
    local_ymin = coords[:, 1].min()
    local_xmax = coords[:, 0].max()
    local_ymax = coords[:, 1].max()

    global_xmin = comm.allreduce(local_xmin, op=MPI.MIN)
    global_ymin = comm.allreduce(local_ymin, op=MPI.MIN)
    global_xmax = comm.allreduce(local_xmax, op=MPI.MAX)
    global_ymax = comm.allreduce(local_ymax, op=MPI.MAX)

    h = np.sqrt((global_xmax - global_xmin)**2 +
                (global_ymax - global_ymin)**2)

    # perturb only barycentric nodes
    for i in bary_nodes:
        x, y = coords[i]

        dx = np.sin(10 * y)
        dy = np.cos(10 * x)

        norm = np.sqrt(dx**2 + dy**2)
        if norm == 0:
            continue

        dx /= norm
        dy /= norm

        coords[i, 0] += eps * h * dx
        coords[i, 1] += eps * h * dy

    return bary_nodes, h

def perturb_error(mesh, bary_nodes, comm):
    coords = mesh.coordinates.dat.data_with_halos
    cell_node_map = mesh.coordinates.cell_node_map().values

    local_max = 0.0

    for cell in cell_node_map:
        verts = coords[cell]
        centroid = np.mean(verts, axis=0)

        for node in cell:
            if node in bary_nodes:
                dist = np.linalg.norm(coords[node] - centroid)
                local_max = max(local_max, dist)

    global_max = comm.allreduce(local_max, op=MPI.MAX)
    return global_max