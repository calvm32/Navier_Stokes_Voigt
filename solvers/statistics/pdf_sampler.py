import numpy as np
from solvers.statistics.spatial_sampler import spatial_sampler
from mpi4py import MPI

class pdf_sampler:
    """
    PDF of velocity magnitude and vorticity using DOF sampling
    """

    def __init__(self, mesh):
        self.sampler = spatial_sampler(mesh)

        self.velocity_x_samples = []
        self.velocity_y_samples = []
        self.velocity_z_samples = []
        self.vorticity_samples = []

    def sample_velocity_x(self, u, npoints=2000):
        """
        samples velocity x-component
        """
        vals = self.sampler.sample_function(u, npoints)

        if len(vals) == 0:
            return

        vals = np.array(vals) # velocities
        vals_x = vals[:, 0] # extract x-component

        self.velocity_x_samples.append(vals_x)

    def sample_velocity_y(self, u, npoints=2000):
        """
        samples velocity y-component
        """
        vals = self.sampler.sample_function(u, npoints)

        if len(vals) == 0:
            return

        vals = np.array(vals) # velocities
        vals_y = vals[:, 1] # extract y-component

        self.velocity_y_samples.append(vals_y)

    def sample_velocity_z(self, u, npoints=2000):
        """
        samples velocity z-component
        """
        vals = self.sampler.sample_function(u, npoints)

        if len(vals) == 0:
            return

        vals = np.array(vals) # velocities
        vals_z = vals[:, 2] # extract z-component

        self.velocity_z_samples.append(vals_z)

    def sample_vorticity(self, omega, npoints=2000):
        """
        samples vorticity (omega)
        """
        vals = self.sampler.sample_function(omega, npoints)

        if len(vals) == 0:
            return

        self.vorticity_samples.append(np.array(vals).flatten())

    def finalize(self):

        comm = MPI.COMM_WORLD

        vel_x_local = (
            np.concatenate(self.velocity_x_samples)
            if self.velocity_x_samples else np.array([])
        )

        vel_y_local = (
            np.concatenate(self.velocity_y_samples)
            if self.velocity_y_samples else np.array([])
        )

        vort_local = (
            np.concatenate(self.vorticity_samples)
            if self.vorticity_samples else np.array([])
        )

        vel_x = comm.allgather(vel_x_local)
        vel_y = comm.allgather(vel_y_local)
        vort = comm.allgather(vort_local)

        vel_x = np.concatenate(vel_x)
        vel_y = np.concatenate(vel_y)
        vort = np.concatenate(vort)

        return vel_x, vel_y, vort