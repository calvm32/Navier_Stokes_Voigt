from firedrake import PointEvaluator
import numpy as np

class spatial_sampler:
    """
    Samples a Firedrake func at random pts in a mesh
    (rectangular mesh only!! for right now)
    """
    def __init__(self, mesh):
        self.mesh = mesh
        coords = mesh.coordinates.dat.data
        self.xmin, self.ymin = coords.min(axis=0)
        self.xmax, self.ymax = coords.max(axis=0)

    def random_point(self):
        """
        return one random pt
        """
        x = np.random.uniform(self.xmin, self.xmax)
        y = np.random.uniform(self.ymin, self.ymax)
        return np.array([x, y])

    def sample_function(self, f, npoints=1000):
        """
        Sample Firedrake Function at random spatial points 
        returns array of sampled pts
        """
        vals = []

        for _ in range(npoints):
            pt = self.random_point()
            try:
                val = f.at(pt) # currently deprecated, but the alternative doesn't work
                vals.append(val)
            except Exception:
                # skip points outside or near edges
                continue

        if len(vals) == 0:
            return np.array([])

        return np.array(vals)

        