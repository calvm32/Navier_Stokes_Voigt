import numpy as np
from mpi4py import MPI


class pdf_sampler_2d:

    def __init__(
        self,
        mesh,
        nbins=100,
        value_range=(-5, 5),
    ):

        self.comm = mesh.comm
        self.rank = self.comm.rank

        self.nbins = nbins
        self.range = value_range

        self.bin_edges = np.linspace(
            value_range[0],
            value_range[1],
            nbins + 1
        )

        self.vel_x_hist = np.zeros(nbins)
        self.vel_y_hist = np.zeros(nbins)
        self.vort_hist = np.zeros(nbins)

    def sample_velocity(self, u, nsamples=5000):

        vals = u.dat.data_ro

        ndofs = vals.shape[0]

        if ndofs == 0:
            return

        idx = np.random.randint(
            0,
            ndofs,
            size=min(nsamples, ndofs)
        )

        sample = vals[idx]

        hist_x, _ = np.histogram(
            sample[:, 0],
            bins=self.bin_edges
        )

        hist_y, _ = np.histogram(
            sample[:, 1],
            bins=self.bin_edges
        )

        self.vel_x_hist += hist_x
        self.vel_y_hist += hist_y

    def sample_vorticity(self, omega, nsamples=5000):

        vals = omega.dat.data_ro

        ndofs = vals.shape[0]

        if ndofs == 0:
            return

        idx = np.random.randint(
            0,
            ndofs,
            size=min(nsamples, ndofs)
        )

        sample = vals[idx]

        hist, _ = np.histogram(
            sample,
            bins=self.bin_edges
        )

        self.vort_hist += hist

    def finalize(self):

        # ALL ranks participate
        global_vel_x = self.comm.allreduce(
            self.vel_x_hist,
            op=MPI.SUM
        )

        global_vel_y = self.comm.allreduce(
            self.vel_y_hist,
            op=MPI.SUM
        )

        global_vort = self.comm.allreduce(
            self.vort_hist,
            op=MPI.SUM
        )

        dx = (
            self.bin_edges[1]
            - self.bin_edges[0]
        )

        pdf_x = global_vel_x / (
            np.sum(global_vel_x) * dx
        )

        pdf_y = global_vel_y / (
            np.sum(global_vel_y) * dx
        )

        pdf_v = global_vort / (
            np.sum(global_vort) * dx
        )

        return pdf_x, pdf_y, pdf_v