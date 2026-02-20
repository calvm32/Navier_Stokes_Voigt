import numpy as np
from solvers.statistics.spatial_sampler import spatial_sampler

class structure_funcs:
    def __init__(self, u, mesh, r_max=None, nbins=30):
        self.u = u
        self.mesh = mesh

        self.sampler = spatial_sampler(mesh)

        xmin, ymin = self.sampler.xmin, self.sampler.ymin
        xmax, ymax = self.sampler.xmax, self.sampler.ymax

        # we need to avoid the boundaries, so only sample a small radial distance (r)
        # this only accounts for small eddies, altho can get an even smaller length scale if wanted
        if r_max is None:
            Lx = xmax - xmin
            Ly = ymax - ymin
            r_max = 0.25 * min(Lx, Ly)

        self.r_edges = np.linspace(0.0, r_max, nbins + 1)
        self.r_centers = 0.5 * (self.r_edges[:-1] + self.r_edges[1:])

        self.S2_accum = np.zeros(nbins) # sum of squared velocity differences for bin i, will be averaged
        self.counts = np.zeros(nbins, dtype=int) # count successful samples (denominator for avg)

    def sample_increment(self, r):
        """
        finds the longitudinal velocity inrement
        along a random vector w/ distance r
        """
        x1 = self.sampler.random_point() # random pt in domain

        theta = 2 * np.pi * np.random.rand() # random direction in radians
        r_vec = r * np.array([np.cos(theta), np.sin(theta)]) # vector of length r in that direction
        x2 = x1 + r_vec # next pt to sample & measure difference

        try:
            u1 = self.u(x1) # velocity at x1
            u2 = self.u(x2) # velocity at x2
        except:
            return None

        diff_u = u2 - u1 # difference
        r_hat = r_vec / np.linalg.norm(r_vec) # unit vector along diff

        return np.dot(diff_u, r_hat) # longitudinal velocity increment

    def sample(self, nsamples_per_bin=50):
        """
        finds samples for all of the bins
        sums them to get increments
        """
        for i, r in enumerate(self.r_centers): # go thru all bins
            success = 0
            attempts = 0

            # keep sampling until desired number of samples
            while success < nsamples_per_bin and attempts < 5 * nsamples_per_bin:
                inc = self.sample_increment(r)
                attempts += 1

                if inc is not None:
                    self.S2_accum[i] += inc**2 # gathers squared longitudinal increments
                    self.counts[i] += 1
                    success += 1

    def compute(self):
        mask = self.counts > 0 # given atl one valid sample
        S2 = np.zeros_like(self.S2_accum)
        S2[mask] = self.S2_accum[mask] / self.counts[mask]
        return self.r_centers, S2