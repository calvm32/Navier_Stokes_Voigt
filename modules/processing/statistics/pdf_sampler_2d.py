import numpy as np
from mpi4py import MPI


class pdf_sampler_2d:
    """
    MPI-safe DOF-sampled PDFs for velocity components + vorticity.

    Designed for long Firedrake turbulence runs:
    - no per-call collectives
    - single synchronized finalize
    - robust against rank divergence
    """

    def __init__(
        self,
        mesh,
        nbins=100,
        value_range=(-5, 5),
        seed=None,
    ):

        self.comm = mesh.comm
        self.rank = self.comm.rank

        self.nbins = nbins
        self.value_range = value_range

        self.bin_edges = np.linspace(
            value_range[0],
            value_range[1],
            nbins + 1
        )

        # local accumulators only
        self.vel_x_hist = np.zeros(nbins, dtype=np.int64)
        self.vel_y_hist = np.zeros(nbins, dtype=np.int64)
        self.vort_hist  = np.zeros(nbins, dtype=np.int64)

        # deterministic RNG per rank
        self.rng = np.random.default_rng(seed + self.rank if seed is not None else None)

    # -----------------------------
    # sampling (pure local work)
    # -----------------------------

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

    # -----------------------------
    # MPI-safe finalize
    # -----------------------------

    def finalize(self):

        # IMPORTANT: all ranks must enter

        local = np.stack(
            [
                self.vel_x_hist,
                self.vel_y_hist,
                self.vort_hist,
            ],
            axis=0
        )

        global_hist = np.zeros_like(local)

        self.comm.Allreduce(
            local,
            global_hist,
            op=MPI.SUM
        )

        vel_x, vel_y, vort = global_hist

        dx = self.bin_edges[1] - self.bin_edges[0]

        # avoid divide-by-zero
        vel_x_sum = np.sum(vel_x)
        vel_y_sum = np.sum(vel_y)
        vort_sum  = np.sum(vort)

        pdf_x = np.zeros_like(vel_x, dtype=float)
        pdf_y = np.zeros_like(vel_y, dtype=float)
        pdf_v = np.zeros_like(vort, dtype=float)

        if vel_x_sum > 0:
            pdf_x = vel_x / (vel_x_sum * dx)

        if vel_y_sum > 0:
            pdf_y = vel_y / (vel_y_sum * dx)

        if vort_sum > 0:
            pdf_v = vort / (vort_sum * dx)

        if self.rank == 0:
            print("[pdf_sampler] finalize complete", flush=True)

        return pdf_x, pdf_y, pdf_v