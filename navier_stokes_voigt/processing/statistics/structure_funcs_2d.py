import numpy as np
from mpi4py import MPI

class structure_funcs_2d:
    """
    2nd-order longitudinal structure function S^2(r) using DOF sampling
    """

    def __init__(self, u, mesh, r_max=None, nbins=30):

        self.u = u
        self.mesh = mesh

        # DOF values (view updates automatically)
        self.values = u.dat.data_ro

        ndofs = self.values.shape[0]

        # Fake geometric scale from mesh bounding box
        # (only needed to define r bins)

        coords = mesh.coordinates.dat.data_ro
        xmin, ymin = coords.min(axis=0)
        xmax, ymax = coords.max(axis=0)

        # we need to avoid the boundaries, so only sample a small radial distance (r) #
        #  this only accounts for small eddies, altho can get an even smaller length scale if wanted
        if r_max is None:
            Lx = xmax - xmin
            Ly = ymax - ymin
            r_max = 0.25 * min(Lx, Ly)

        self.r_edges = np.linspace(0.0, r_max, nbins + 1)
        self.r_centers = 0.5 * (self.r_edges[:-1] + self.r_edges[1:])

        self.S2_accum = np.zeros(nbins) # sum of squared velocity differences for bin i, will be averaged 
        self.counts = np.zeros(nbins, dtype=int) # count successful samples (denominator for avg)

        self.ndofs = ndofs

    def sample_increment(self):
        """
        longitudinal velocity increment
        """

        i = np.random.randint(0, self.ndofs)
        j = np.random.randint(0, self.ndofs)

        u1 = self.values[i]
        u2 = self.values[j]

        diff = u2 - u1

        # random direction (isotropic projection)
        theta = 2.0 * np.pi * np.random.rand()
        r_hat = np.array([np.cos(theta), np.sin(theta)])

        return np.dot(diff, r_hat)

    def sample(self, nsamples_per_bin=50):
        """
        actually sample now
        """

        for i in range(len(self.r_centers)):

            for _ in range(nsamples_per_bin):

                inc = self.sample_increment()

                self.S2_accum[i] += inc ** 2
                self.counts[i] += 1

    def compute(self):

        comm = MPI.COMM_WORLD

        S2_global = comm.allreduce(self.S2_accum, op=MPI.SUM)
        counts_global = comm.allreduce(self.counts, op=MPI.SUM)

        mask = counts_global > 0
        S2 = np.zeros_like(S2_global)
        S2[mask] = S2_global[mask] / counts_global[mask]

        return self.r_centers, S2