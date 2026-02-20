import numpy as np

class pdf_sampler:
    def __init__(self):
        self.velocity_samples = []
        self.vorticity_samples = []

    def sample_velocity(self, u):
        vals = np.linalg.norm(u.dat.data, axis=1)
        self.velocity_samples.append(vals)

    def sample_vorticity(self, omega):
        self.vorticity_samples.append(omega.dat.data.copy())

    def finalize(self):
        vel = np.concatenate(self.velocity_samples)
        vort = np.concatenate(self.vorticity_samples)
        return vel, vort
