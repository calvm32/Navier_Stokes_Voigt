def create_surface_measure_left(mesh):
    """
    Return surface measure on the left boundary of unit square
    """
    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 0)
    facets = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
    Left().mark(facets, 1)
    ds_left = Measure("ds", mesh, subdomain_data=facets, subdomain_id=1)
    return ds_left

from firedrake import *

def create_surface_measure_left(mesh):
    """
    Return surface measure on the left boundary of a unit square (x = 0)
    """
    # Create a facet function (MeshTags) for boundary marking
    facet_tags = MeshTags(mesh, mesh.topological_dimension() - 1, 0)

    # Define left boundary as x = 0
    left_facets = []
    for f in range(mesh.num_facets()):
        cell = mesh.facet_cell(f)
        coords = mesh.coordinates.dat.data[mesh.cell_node_list[cell]]
        # Check if all facet nodes are near x = 0
        if all(abs(x[0]) < 1e-8 for x in coords):
            left_facets.append(f)

    # Mark left facets with ID 1
    if left_facets:
        facet_tags[left_facets] = 1

    # Create a measure over the marked facets
    ds_left = ds(subdomain_data=facet_tags, subdomain_id=1)

    return ds_left
