import numpy as np

class structure_funcs:
    def __init__(self, u, mesh):
        self.u = u
        self.mesh = mesh
        coords = mesh.coordinates.dat.data
        self.xmin, self.ymin = coords.min(axis=0)
        self.xmax, self.ymax = coords.max(axis=0)

    def random_point(self):
        for _ in range(20):
            x = np.random.uniform(self.xmin, self.xmax)
            y = np.random.uniform(self.ymin, self.ymax)
            try:
                _ = self.u((x, y))
                return np.array([x, y])
            except:
                pass
        raise RuntimeError("Sampling failed")

    def sample_increment(self, r):
        x1 = self.random_point()
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

    def compute(self, r, nsamples=1000):
        vals = []
        for _ in range(nsamples):
            inc = self.sample_increment(r)
            if inc is not None:
                vals.append(inc)
        return np.array(vals)
