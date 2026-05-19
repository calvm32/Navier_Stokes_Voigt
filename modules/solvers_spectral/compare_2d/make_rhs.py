import numpy as np

def make_rhs(kx, ky, Re, alpha): 

    def dealias(u_hat):
        Nx, Ny = u_hat.shape
        mask = np.ones((Nx, Ny))
        kx_cut = Nx // 3
        ky_cut = Ny // 3
        mask[kx_cut:-kx_cut, :] = 0
        mask[:, ky_cut:-ky_cut] = 0
        return u_hat * mask

    # setup for Laplacian terms 
    ksq = kx[:,None]**2 + ky[None,:]**2 
    inv_lap = np.zeros_like(ksq) # array of zeroes, then keep 0 node = 0 
    for i in range(ksq.shape[0]): # go through and set stuff, but avoid dividing by 0 
        for j in range(ksq.shape[1]): 
            if ksq[i, j] != 0: 
                inv_lap[i, j] = -1.0 / ksq[i, j] 
                
    def rhs_NSE(psi_hat, f_hat): 
        psi = np.fft.ifftn(psi_hat) 
        
        # laplace in fourier space 
        omega_hat = -ksq * psi_hat 
        omega = np.fft.ifftn(omega_hat) 
        
        # compute derivatives 
        psi_x = np.fft.ifftn(1j * kx[:, None] * psi_hat) 
        psi_y = np.fft.ifftn(1j * ky[None, :] * psi_hat) 
        omega_x = np.fft.ifftn(1j * kx[:, None] * omega_hat) 
        omega_y = np.fft.ifftn(1j * ky[None, :] * omega_hat) 
        
        # back to fourier 
        nonlinear_hat = -inv_lap*dealias(np.fft.fftn(psi_x * omega_y - psi_y * omega_x))
        lap_psi_hat = -ksq * psi_hat 

        forcing_hat = f_hat
        
        return nonlinear_hat + (1/Re)*lap_psi_hat + forcing_hat
        #return (1/Re)*lap_psi_hat + forcing_hat
    

    def rhs_NSV(psi_hat, f_hat): 
        psi = np.fft.ifftn(psi_hat) 
        
        # laplace in fourier space 
        omega_hat = ksq * psi_hat 
        omega = np.fft.ifftn(omega_hat) 
        
        # compute derivatives 
        psi_x = np.fft.ifftn(1j * kx[:, None] * psi_hat) 
        psi_y = np.fft.ifftn(1j * ky[None, :] * psi_hat) 
        omega_x = np.fft.ifftn(1j * kx[:, None] * omega_hat) 
        omega_y = np.fft.ifftn(1j * ky[None, :] * omega_hat) 
        
        # back to fourier 
        nonlinear_hat = -inv_lap*dealias(np.fft.fftn(psi_x * omega_y - psi_y * omega_x))
        lap_psi_hat = -ksq * psi_hat 

        forcing_hat = f_hat
        
        # finally sum everything and divide by voigt b/c (1 + alpha^2k^2) = RHS
        return (nonlinear_hat + (1/Re)*lap_psi_hat + forcing_hat) / (1.0 + alpha**2*ksq)
    
    return rhs_NSE, rhs_NSV