import numpy as np 

def make_rhs(kx, ky, dealias, Re, inv_lap):

    def rhs(psi_hat, f_hat, ksq):

        # laplacian
        lap_psi_hat = -ksq*psi_hat

        # gradients
        psi_x = np.fft.ifftn(1j*kx[:,None]*psi_hat)
        psi_y = np.fft.ifftn(1j*ky[None,:]*psi_hat)

        lap_psi_x = np.fft.ifftn(1j*kx[:,None]*lap_psi_hat)
        lap_psi_y = np.fft.ifftn(1j*ky[None,:]*lap_psi_hat)

        # nonlinear Jacobian
        J = psi_x*lap_psi_y - psi_y*lap_psi_x
        J_hat = np.fft.fftn(J)
        J_hat = dealias(J_hat)

        nonlinear_hat = -inv_lap * J_hat

        # viscous term
        viscous_hat = (1.0/Re)* lap_psi_hat

        # forcing term
        f_x_hat, f_y_hat = f_hat

        curl_f_hat = 1j*kx[:,None]*f_y_hat - 1j*ky[None,:]*f_x_hat
        forcing_hat = inv_lap*curl_f_hat

        return (viscous_hat + nonlinear_hat + forcing_hat) 

    return rhs