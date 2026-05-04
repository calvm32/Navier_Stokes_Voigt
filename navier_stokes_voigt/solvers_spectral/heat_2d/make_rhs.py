import numpy as np 

def make_rhs(kx, ky):

    def rhs(u_hat, f_hat, ksq):
        # Compute derivatives in Fourier space
        u_x_hat=1j*kx[:,None]*u_hat # multiply along columns
        u_y_hat=1j*ky[None,:]*u_hat # multiply along rows
        
        # t is unused here, but kept for compatibility with general PDEs.
        return -c1*u_x_hat -c2*u_y_hat + nuLap*u_hat

    return rhs