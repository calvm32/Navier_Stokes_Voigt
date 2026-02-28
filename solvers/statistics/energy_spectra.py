import numpy as np
from mpi4py import MPI

class energy_spectra:
    """
    kinetic energy spectrum using DOF
    mostly works for rectangles
    """

    def __init__(self, u, mesh, nbins=50):
        self.u = u
        self.mesh = mesh
        self.nbins = nbins

        # DOF coordinates
        self.coords = mesh.coordinates.dat.data_ro.copy()

    def compute(self):

        comm = self.mesh.comm

        uvals = self.u.dat.data_ro

        if uvals.ndim == 1:
            ux = uvals
            uy = np.zeros_like(ux)
        else:
            ux = uvals[:, 0]
            uy = uvals[:, 1]

        # ---- remove global mean (must be global!) ----
        local_sum_x = np.sum(ux)
        local_sum_y = np.sum(uy)
        local_N = len(ux)

        global_sum_x = comm.allreduce(local_sum_x, op=MPI.SUM)
        global_sum_y = comm.allreduce(local_sum_y, op=MPI.SUM)
        global_N = comm.allreduce(local_N, op=MPI.SUM)

        mean_x = global_sum_x / global_N
        mean_y = global_sum_y / global_N

        ux = ux - mean_x
        uy = uy - mean_y

        # ---- local FFT ----
        ux_hat = np.fft.fft(ux)
        uy_hat = np.fft.fft(uy)

        energy_modes = 0.5 * (np.abs(ux_hat)**2 + np.abs(uy_hat)**2)

        # ---- global bounding box ----
        coords = self.mesh.coordinates.dat.data_ro
        xmin_local, ymin_local = coords.min(axis=0)
        xmax_local, ymax_local = coords.max(axis=0)

        xmin = comm.allreduce(xmin_local, op=MPI.MIN)
        ymin = comm.allreduce(ymin_local, op=MPI.MIN)
        xmax = comm.allreduce(xmax_local, op=MPI.MAX)
        ymax = comm.allreduce(ymax_local, op=MPI.MAX)

        Lx = xmax - xmin
        Ly = ymax - ymin

        N = global_N

        k = np.fft.fftfreq(len(energy_modes), d=min(Lx, Ly)/np.sqrt(N))
        k = np.abs(k)

        k_bins = np.linspace(0, k.max(), self.nbins + 1)

        E_local = np.zeros(self.nbins)
        counts_local = np.zeros(self.nbins)

        inds = np.digitize(k, k_bins) - 1

        for i in range(len(energy_modes)):
            b = inds[i]
            if 0 <= b < self.nbins:
                E_local[b] += energy_modes[i]
                counts_local[b] += 1

        # ---- GLOBAL REDUCTION ----
        E = comm.allreduce(E_local, op=MPI.SUM)
        counts = comm.allreduce(counts_local, op=MPI.SUM)

        counts[counts == 0] = 1
        E /= counts

        k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

        return k_centers, E