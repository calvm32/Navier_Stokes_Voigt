import numpy as np
from mpi4py import MPI


class pdf_sampler_2d: 
    def __init__(
        self,
        mesh,
        nbins=100,
        value_range=(-5, 5),
        seed=None,
    ):

        self.comm = mesh.comm
        self.rank = self.comm.rank
        self.size = self.comm.size

        self.nbins = nbins
        self.value_range = value_range

        self.bin_edges = np.linspace(
            value_range[0],
            value_range[1],
            nbins + 1
        )

        # local accumulators
        self.vel_x_hist = np.zeros(nbins, dtype=np.int64)
        self.vel_y_hist = np.zeros(nbins, dtype=np.int64)
        self.vort_hist  = np.zeros(nbins, dtype=np.int64)

        self.rng = np.random.default_rng(
            seed + self.rank if seed is not None else None
        )

        # global cached state (always valid)
        self.global_vel_x = np.zeros(nbins, dtype=np.int64)
        self.global_vel_y = np.zeros(nbins, dtype=np.int64)
        self.global_vort  = np.zeros(nbins, dtype=np.int64)

    # --------------
    # local sampling
    # --------------

    def sample_velocity(self, u, nsamples=5000):

        vals = u.dat.data_ro
        ndofs = vals.shape[0]

        if ndofs == 0:
            return

        n = min(nsamples, ndofs)

        idx = self.rng.integers(0, ndofs, size=n)

        sample = vals[idx]

        hist_x, _ = np.histogram(sample[:, 0], bins=self.bin_edges)
        hist_y, _ = np.histogram(sample[:, 1], bins=self.bin_edges)

        self.vel_x_hist += hist_x
        self.vel_y_hist += hist_y

    def sample_vorticity(self, omega, nsamples=5000):

        vals = omega.dat.data_ro
        ndofs = vals.shape[0]

        if ndofs == 0:
            return

        n = min(nsamples, ndofs)

        idx = self.rng.integers(0, ndofs, size=n)

        sample = vals[idx]

        hist, _ = np.histogram(sample, bins=self.bin_edges)

        self.vort_hist += hist


    def sync(self):

        # reduce each field independently but safely
        self.global_vel_x += self.comm.allreduce(self.vel_x_hist, op=MPI.SUM)
        self.global_vel_y += self.comm.allreduce(self.vel_y_hist, op=MPI.SUM)
        self.global_vort  += self.comm.allreduce(self.vort_hist,  op=MPI.SUM)

        # reset local accumulators after sync
        self.vel_x_hist.fill(0)
        self.vel_y_hist.fill(0)
        self.vort_hist.fill(0)


    def finalize(self):

        dx = self.bin_edges[1] - self.bin_edges[0]

        def norm(x):
            s = np.sum(x)
            if s == 0:
                return np.zeros_like(x)
            return x / (s * dx)

        return (
            norm(self.global_vel_x),
            norm(self.global_vel_y),
            norm(self.global_vort),
        )