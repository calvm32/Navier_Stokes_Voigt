import numpy as np

class energy_spectra:
    def __init__(self, u, nx, ny):
        self.u = u
        self.nx = nx
        self.ny = ny

    def compute(self):
        uvals = self.u.dat.data
        ux = uvals[:,0].reshape(self.nx, self.ny)
        uy = uvals[:,1].reshape(self.nx, self.ny)

        ux_hat = np.fft.fftn(ux)
        uy_hat = np.fft.fftn(uy)

        E = 0.5 * (np.abs(ux_hat)**2 + np.abs(uy_hat)**2)
        return np.real(E)
