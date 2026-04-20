import numpy as np
from solvers.processing.statistics.spatial_sampler import spatial_sampler
from firedrake import COMM_WORLD

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

class pdf_sampler:
    """
    PDF of velocity magnitude and vorticity using DOF sampling
    """

    def __init__(self, mesh):
        self.comm = mesh.comm
        self.nbins = 100
        self.range = (-5, 5) # may need to change? # should b fine
        
        self.sampler = spatial_sampler(mesh)

        self.vel_x_hist = np.zeros(self.nbins)
        self.vel_y_hist = np.zeros(self.nbins)
        self.vort_hist  = np.zeros(self.nbins)

        self.bin_edges = np.linspace(self.range[0], self.range[1], self.nbins + 1)

    def sample_velocity(self, u, npoints=2000):
        """
        samples velocity values, breaks into x and y components
        """
        vals = self.sampler.sample_function(u, npoints)
        if len(vals) == 0:
            return

        vals = np.array(vals)

        hist_x, _ = np.histogram(vals[:, 0], bins=self.bin_edges)
        hist_y, _ = np.histogram(vals[:, 1], bins=self.bin_edges)

        self.vel_x_hist += hist_x
        self.vel_y_hist += hist_y

    def sample_vorticity(self, omega, npoints=2000):
        """
        samples vorticity (omega)
        """
        vals = self.sampler.sample_function(omega, npoints)
        if len(vals) == 0:
            return

        vals = np.array(vals).flatten()
        hist, _ = np.histogram(vals, bins=self.bin_edges)
        self.vort_hist += hist

    def finalize(self):

        # sum histograms across MPI ranks
        global_vel_x = comm.reduce(self.vel_x_hist, op=MPI.SUM, root=0)
        global_vel_y = comm.reduce(self.vel_y_hist, op=MPI.SUM, root=0)
        global_vort  = comm.reduce(self.vort_hist,  op=MPI.SUM, root=0)

        if rank == 0:
            # normalize to PDF
            dx = self.bin_edges[1] - self.bin_edges[0]

            pdf_x = global_vel_x / np.sum(global_vel_x) / dx
            pdf_y = global_vel_y / np.sum(global_vel_y) / dx
            pdf_v = global_vort  / np.sum(global_vort)  / dx

            #centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

            return pdf_x, pdf_y, pdf_v
        else:
            return None, None, None, None