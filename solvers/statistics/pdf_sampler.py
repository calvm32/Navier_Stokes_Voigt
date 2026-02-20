import numpy as np
from solvers.statistics.spatial_sampler import spatial_sampler


class pdf_sampler:
    """
    PDF of velocity magnitude and vorticity using DOF sampling
    """

    def __init__(self, mesh):
        self.sampler = spatial_sampler(mesh)

        self.velocity_samples = []
        self.vorticity_samples = []

    def sample_velocity(self, u, npoints=2000):
        """
        samples velocity magnitudes (u)
        """
        vals = self.sampler.sample_function(u, npoints)

        if len(vals) == 0:
            return

        vals = np.array(vals) # velocity
        mags = np.linalg.norm(vals, axis=1) # magnitude

        self.velocity_samples.append(mags)

    def sample_vorticity(self, omega, npoints=2000):
        """
        samples vorticity (omega)
        """
        vals = self.sampler.sample_function(omega, npoints)

        if len(vals) == 0:
            return

        self.vorticity_samples.append(np.array(vals).flatten())

    def finalize(self):

        vel = (
            np.concatenate(self.velocity_samples)
            if self.velocity_samples
            else np.array([])
        )

        vort = (
            np.concatenate(self.vorticity_samples)
            if self.vorticity_samples
            else np.array([])
        )

        return vel, vort