def create_surface_measure_left(mesh):
    """
    Return surface measure on the left boundary of unit
    square
    """
    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 0)
    facets = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
    Left().mark(facets, 1)
    ds_left = Measure("ds", mesh, subdomain_data=facets, subdomain_id=1)
    return ds_left