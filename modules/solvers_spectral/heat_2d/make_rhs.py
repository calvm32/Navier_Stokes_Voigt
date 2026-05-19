import numpy as np 

def make_rhs(kx, ky):
    ksq = kx[:,None]**2 + ky[None,:]**2

    def rhs(u_hat, f_hat):
        lap_u_hat = -ksq * u_hat
        return lap_u_hat + f_hat

    return rhs
