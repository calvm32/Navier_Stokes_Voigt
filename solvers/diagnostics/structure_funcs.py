import numpy as np

class structure_funcs:
    def __init__(self, u, mesh, r_max=None, nbins=30):
        self.u = u
        self.mesh = mesh

        coords = mesh.coordinates.dat.data
        self.xmin, self.ymin = coords.min(axis=0)
        self.xmax, self.ymax = coords.max(axis=0)

        if r_max is None:
            Lx = self.xmax - self.xmin
            Ly = self.ymax - self.ymin
            r_max = 0.25 * min(Lx, Ly)

        self.r_edges = np.linspace(0.0, r_max, nbins + 1)
        self.r_centers = 0.5 * (self.r_edges[:-1] + self.r_edges[1:])

        self.S2_accum = np.zeros(nbins)
        self.counts = np.zeros(nbins, dtype=int)

    def random_point(self):
        for _ in range(20):
            x = np.random.uniform(self.xmin, self.xmax)
            y = np.random.uniform(self.ymin, self.ymax)
            try:
                _ = self.u((x, y))
                return np.array([x, y])
            except:
                pass
        return None

    def sample_increment(self, r):
        x1 = self.random_point()
        if x1 is None:
            return None

        theta = 2*np.pi*np.random.rand()
        rvec = r * np.array([np.cos(theta), np.sin(theta)])
        x2 = x1 + rvec

        try:
            u1 = self.u(x1)
            u2 = self.u(x2)
        except:
            return None

        du = u2 - u1
        rhat = rvec / np.linalg.norm(rvec)
        return np.dot(du, rhat)

    def sample(self, nsamples_per_bin=50):
        """
        Sample structure function increments for all r-bins
        """
        for i, r in enumerate(self.r_centers):
            for _ in range(nsamples_per_bin):
                inc = self.sample_increment(r)
                if inc is not None:
                    self.S2_accum[i] += inc**2
                    self.counts[i] += 1

    def compute(self):
        """
        Return averaged S2(r)
        """
        mask = self.counts > 0
        S2 = np.zeros_like(self.S2_accum)
        S2[mask] = self.S2_accum[mask] / self.counts[mask]
        return self.r_centers, S2
