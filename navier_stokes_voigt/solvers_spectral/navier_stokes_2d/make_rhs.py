def make_rhs(kx, ky, Re): 
    # setup for Laplacian terms 
    ksq = kx[:,None]**2 + ky[None,:]**2 
    inv_lap = np.zeros_like(ksq) # array of zeroes, then keep 0 node = 0 
    for i in range(ksq.shape[0]): # go through and set stuff, but avoid dividing by 0 
        for j in range(ksq.shape[1]): 
            if ksq[i, j] != 0: 
                inv_lap[i, j] = -1.0 / ksq[i, j] 
                
    def rhs(psi_hat, f_hat): 
        psi = np.fft.ifft2(psi_hat) 
        
        # laplace in fourier space 
        omega_hat = ksq * psi_hat 
        omega = np.fft.ifft2(omega_hat) 
        
        # compute derivatives 
        psi_x = np.fft.ifft2(1j * kx[:, None] * psi_hat) 
        psi_y = np.fft.ifft2(1j * ky[None, :] * psi_hat) 
        omega_x = np.fft.ifft2(1j * kx[:, None] * omega_hat) 
        omega_y = np.fft.ifft2(1j * ky[None, :] * omega_hat) 
        
        # back to fourier 
        nonlinear_hat = inv_lap*np.fft.fft2(psi_x * omega_y - psi_y * omega_x) 
        lap_psi_hat = -ksq * psi_hat 
        
        return nonlinear_hat + (1/Re)*lap_psi_hat + f_hat 
    
    return rhs