import numpy as np

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

        uvals = self.u.dat.data_ro

        if uvals.ndim == 1:
            ux = uvals
            uy = np.zeros_like(ux)
        else:
            ux = uvals[:, 0]
            uy = uvals[:, 1]

        N = len(ux)

        # Remove mean (important for spectra)
        ux = ux - np.mean(ux)
        uy = uy - np.mean(uy)

        # FFT of flattened data
        ux_hat = np.fft.fft(ux)
        uy_hat = np.fft.fft(uy)

        energy_modes = 0.5 * (np.abs(ux_hat) ** 2 + np.abs(uy_hat) ** 2)

        # Create pseudo wavenumbers based on spatial spacing
        x = self.coords[:, 0]
        y = self.coords[:, 1]

        Lx = x.max() - x.min()
        Ly = y.max() - y.min()

        k = np.fft.fftfreq(N, d=min(Lx, Ly) / np.sqrt(N))
        k = np.abs(k)

        # Radial binning
        k_bins = np.linspace(0, k.max(), self.nbins + 1)
        E = np.zeros(self.nbins)
        counts = np.zeros(self.nbins)

        inds = np.digitize(k, k_bins) - 1

        for i in range(N):
            b = inds[i]
            if 0 <= b < self.nbins:
                E[b] += energy_modes[i]
                counts[b] += 1

        counts[counts == 0] = 1
        E /= counts

        k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

        return k_centers, E