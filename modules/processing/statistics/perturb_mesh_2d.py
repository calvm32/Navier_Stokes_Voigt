import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD

# -------------
# length scales
# -------------

def compute_global_h(coords):
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

    return h

def compute_local_h(mesh):
    coords = mesh.coordinates.dat.data_with_halos
    cell_node_map = mesh.coordinates.cell_node_map().values

    edge_lengths = []

    for cell in cell_node_map:
        verts = coords[cell]
        a = np.linalg.norm(verts[1] - verts[0])
        b = np.linalg.norm(verts[2] - verts[1])
        c = np.linalg.norm(verts[0] - verts[2])
        edge_lengths.extend([a, b, c])

    local_mean_h = np.mean(edge_lengths)
    global_mean_h = comm.allreduce(local_mean_h, op=MPI.SUM) / comm.allreduce(1, op=MPI.SUM)

    return global_mean_h

# --------------------------
# Deterministic perturbation
# --------------------------

def deterministic_perturbation(coords, eps, h):
    perturb = np.zeros_like(coords)

    perturb[:, 0] = np.sin(2 * np.pi * coords[:, 1] / h)
    perturb[:, 1] = np.cos(2 * np.pi * coords[:, 0] / h)

    norms = np.linalg.norm(perturb, axis=1)
    norms[norms == 0] = 1.0
    perturb = perturb / norms[:, None]

    return eps*h*perturb


def perturb_mesh(mesh, eps):
    coords = mesh.coordinates.dat.data_with_halos
    h = compute_global_h(coords)

    coords[:] += deterministic_perturbation(coords, eps, h)

    return h

# --------------
# measure change
# --------------

def triangle_distortion(verts):
    centroid = np.mean(verts, axis=0)
    dists = np.linalg.norm(verts - centroid, axis=1)

    if np.mean(dists) == 0:
        return 0.0

    return np.std(dists) / np.mean(dists)


def mesh_distortion(mesh):
    coords = mesh.coordinates.dat.data_with_halos
    cell_node_map = mesh.coordinates.cell_node_map().values

    local_sum = 0.0
    local_max = 0.0
    local_count = 0

    for cell in cell_node_map:
        verts = coords[cell]

        val = triangle_distortion(verts)

        local_sum += val
        local_max = max(local_max, val)
        local_count += 1

    global_sum = comm.allreduce(local_sum, op=MPI.SUM)
    global_max = comm.allreduce(local_max, op=MPI.MAX)
    global_count = comm.allreduce(local_count, op=MPI.SUM)

    mean_dist = global_sum / global_count

    return mean_dist, global_max

# ------------
# Aspect ratio
# ------------

def triangle_aspect_ratio(verts):
    # using formula for area given vertices
    a = np.linalg.norm(verts[1] - verts[0])
    b = np.linalg.norm(verts[2] - verts[1])
    c = np.linalg.norm(verts[0] - verts[2])

    longest = max(a, b, c)
    s = 0.5*(a + b + c)
    area = np.sqrt(max(s*(s - a)*(s - b)*(s - c), 0.0))

    # height = 2*area/length
    min_height = 2*area/longest

    if min_height == 0:
        return np.inf

    return longest/min_height


def mesh_aspect_ratio(mesh):
    coords = mesh.coordinates.dat.data_with_halos
    cell_node_map = mesh.coordinates.cell_node_map().values

    local_max = 0.0
    local_sum = 0.0
    local_count = 0

    for cell in cell_node_map:
        verts = coords[cell]

        val = triangle_aspect_ratio(verts)

        local_max = max(local_max, val)
        local_sum += val
        local_count += 1

    global_max = comm.allreduce(local_max, op=MPI.MAX)
    global_sum = comm.allreduce(local_sum, op=MPI.SUM)
    global_count = comm.allreduce(local_count, op=MPI.SUM)

    mean_ar = global_sum / global_count

    return mean_ar, global_max

# --------------------
# add this code in !!!
# --------------------

if __name__ == "__main__":

    h = perturb_mesh(mesh, eps)

    # measure changes
    mean_dist, max_dist = mesh_distortion(mesh)
    mean_ar, max_ar = mesh_aspect_ratio(mesh)

    if rank == 0:
        print(f"\neps = {eps}")
        print(f"h = {h:.6e}")

        print("\n--- Distortion ---")
        print(f"mean distortion = {mean_dist:.6e}")
        print(f"max distortion  = {max_dist:.6e}")

        print("\n--- Aspect Ratio ---")
        print(f"mean AR = {mean_ar:.6e}")
        print(f"max AR  = {max_ar:.6e}")
