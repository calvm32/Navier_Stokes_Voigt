import numpy as np
from solvers.statistics.spatial_sampler import spatial_sampler

class pdf_sampler:
    def __init__(self, mesh):
        self.sampler = spatial_sampler(mesh)
        self.velocity_samples = []
        self.vorticity_samples = []

    def sample_velocity(self, u, npoints=2000):
        """
        samples random pts and finds their velocity magnitude (u)
        """
        vals = self.sampler.sample_function(u, npoints) # computes velocity

        if len(vals) > 0:
            mags = np.linalg.norm(vals, axis=1)
            self.velocity_samples.append(mags)

    def sample_vorticity(self, omega, npoints=2000):
        """
        samples random pts and finds their vorticity (omega)
        """
        vals = self.sampler.sample_function(omega, npoints) # computes vorticity

        if len(vals) > 0:
            self.vorticity_samples.append(vals)

    def finalize(self):
        vel = np.concatenate(self.velocity_samples) if self.velocity_samples else np.array([])
        vort = np.concatenate(self.vorticity_samples) if self.vorticity_samples else np.array([])
        return vel, vort