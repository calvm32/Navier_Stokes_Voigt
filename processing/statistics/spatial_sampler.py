import numpy as np
from mpi4py import MPI

class spatial_sampler:
    """
    Samples a Firedrake Function using DOF nodes
    assumes non-adaptive mesh to avoid bias (can do purely spatial, but that's quite slow)
    """

    def __init__(self, mesh):
        self.mesh = mesh

    def sample_function(self, f, npoints=1000):
        """
        Randomly sample DOFs from a Firedrake Function
        Returns numpy array of sampled valuess
        """

        data = f.dat.data_ro  # read-only view

        if data.size == 0:
            return np.array([])

        ndofs = len(data)

        # random indices with replacement
        idx = np.random.randint(0, ndofs, size=npoints)

        return np.array(data[idx])