from firedrake import *
import yaml
from pathlib import Path
import os
import shutil
import csv
import sys

from processing.printoff import blue
from processing.config_setup import *
import matplotlib.pyplot as plt
import numpy as np

def main(save_dir):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    cfg, solver_parameters = load_run_configs(save_dir)

    plot_path = Path(save_dir) / "plots"
    plot_path.mkdir(exist_ok=True)

    # ------------------
    # Configure settings
    # ------------------

    # Extract settings
    t0 = cfg["t0"]
    T = cfg["T"]
    dt = cfg["dt"]
    theta = cfg["theta"]
    Re = cfg["Re"]
    alpha = cfg["alpha"]
    G = cfg["G"]
    P = cfg["P"]
    solver = cfg["solver"]
    elements = cfg["elements"]

    vtkfile_name = "Soln"

    # --------------
    # Configure mesh
    # --------------

    HERE = os.path.dirname(os.path.abspath(__file__))
    MESH_PATH = os.path.join(HERE, "meshes/mms", "channel.msh")

    # ------------------------------------------
    # Setup FEM space for direct mesh comparison
    # ------------------------------------------

    blue(f"\n*** Starting solve ***", spaced=True)

    mesh = Mesh(MESH_PATH)
    x, y = SpatialCoordinate(mesh)

    dx = Measure("dx", domain=mesh)
    ds = Measure("ds", domain=mesh)

    # get height H
    y_coords = mesh.coordinates.dat.data[:, 1]

    local_ymin = y_coords.min()
    local_ymax = y_coords.max()

    global_ymin = comm.allreduce(local_ymin, op=MPI.MIN)
    global_ymax = comm.allreduce(local_ymax, op=MPI.MAX)

    H = global_ymax - global_ymin

    # get length L
    x_coords = mesh.coordinates.dat.data[:, 0]

    local_xmin = x_coords.min()
    local_xmax = x_coords.max()

    global_xmin = comm.allreduce(local_xmin, op=MPI.MIN)
    global_xmax = comm.allreduce(local_xmax, op=MPI.MAX)

    L = global_xmax - global_xmin

    # get approximate DOFs per unit length:
    W = FunctionSpace(mesh, "CG", 1)
    dof = W.dof_count()
    N = sqrt(dof/(H*L)) # number of subdivisions per unit length

    dx = 1/N # length of those subdivisions

    # --------------------
    # Setup spectral space
    # --------------------

    # Grid (periodic, endpoint excluded)
    x = np.linspace(0,L,N*L,endpoint = False)
    y = np.linspace(0,H,N*H,endpoint = False)
    X,Y = np.meshgrid(x,y,indexing = "ij")

    # Standard NumPy wavenumbers:
    # fftfreq gives cycles per unit length; multiply by 2*pi for angular wavenumbers.
    kx = 2.0*np.pi*np.fft.fftfreq(N*L,d = dx)
    ky = 2.0*np.pi*np.fft.fftfreq(N*H,d = dy)
    Laplacian_k = -kx[:,None]**2 - ky[None,:]**2

    # -------------------
    # Configure functions
    # -------------------

    # initialize t for later
    t = Constant(t0)

    namespace = {
        "x": x,
        "y": y,
        "H": H,
        "L": L,
        "G": G,
        "P": P,
        "Re": Re,
        "pi": pi,
        "sin": sin,
        "cos": cos,
        "exp": exp,
        "t": t,
    }

    numpy_cfg = load_run_ufls(save_dir, namespace)

    # -------------------
    # Boundary conditions
    # -------------------

    numpy_inflow = numpy_cfg["numpy_inflow"]

    # ------------------
    # Allocate functions
    # ------------------

    def get_data(t_curr):

        t.assign(t_curr)

        return {
            "numpy_v0": numpy_cfg["numpy_v0"],
            "numpy_p0": numpy_cfg["numpy_p0"],
            "numpy_f": numpy_cfg["numpy_f"],
            "numpy_g": numpy_cfg["numpy_g"]
        }

    # --------
    # make RHS (  inv(lap)( (grad^T psi)* grad(lap psi) ) + 1/Re lap(psi) + inv(lap)grad^T * grad(f)  )
    # --------

    # setup for Laplacian terms
    ksq = kx[:,None]**2 + ky[None,:]**2
    inv_lap = np.zeros_like(ksq) # array of zeroes, then keep 0 node = 0
    inv_lap[ksq != 0] = -1.0 / ksq # [ksq != 0] so that we don't divide by 0

    numpy_f = numpy_cfg["numpy_f"]
    
    def rhs(t, psi_hat):
        # compute gradients in Fourier space
        psi_x_hat=1j*kx[:,None]*psi_hat # multiply along columns
        psi_y_hat=1j*ky[None,:]*psi_hat # multiply along rows

        # ------
        # term 1
        # ------

        # compute laplacian terms
        lap_psi_hat = -ksq*u_hat

        # go back into real space for nonlinear term
        psi_x = np.fft.ifftn(psi_x_hat)
        psi_y = np.fft.ifftn(psi_y_hat)

        # gradient of laplacian
        lap_psi_x_hat = 1j*kx[:, None]*lap_psi_hat
        lap_psi_y_hat = 1j*ky[None, :]*lap_psi_hat

        lap_psi_x = np.fft.ifftn(lap_psi_x_hat)
        lap_psi_y = np.fft.ifftn(lap_psi_y_hat)

        term1 = -psi_x*lap_psi_x + psi_y*lap_psi_y # NEGATIVE first component for grad^Transpose
        term1_hat = inv_lap*np.fft.fftn(term1) # finally apply inverse laplacian to nonlinear term converted back

        # ----------
        # term 2 & 3
        # ----------

        term2_hat = (1.0/Re)*lap_psi_hat

        # finally forcing: inv(lap)grad^T * grad(f)
        f_hat = np.fft.fftn(numpy_f)

        grad_f_x_hat = 1j*kx[:, None]*f_hat
        grad_f_y_hat = 1j*ky[None, :]*f_hat

        grad_f_x = np.fft.ifftn(grad_f_x_hat)
        grad_f_y = np.fft.ifftn(grad_f_y_hat)

        term3 = -grad_f_x*grad_f_x + grad_f_y*grad_f_y # again negative first component for grad^T
        term3_hat = inv_lap*np.fft.fftn(forcing)
    
        return term1_hat + term2_hat + term3_hat

    # initial value
    psi0 = numpy_cfg["numpy_psi0"]
    psi_hat_0 = np.fft.fftn(psi0)

    # ----------
    # Run solver
    # ----------

    psi_hat, times = timestepper_RK4(rhs, psi_hat_0, f_hat, t0, T, dt, ksq, Re, alpha)

    # Plotting
    plt.ion()
    fig, ax = plt.subplots()

    # main time loop
    for n in range(len(t)-1):  
        ax.clear()
        ax.plot(x,np.fft.ifft(psi_hat[n]).real,linewidth=2) # inverse Fourier
        ax.set_xlim(0,L); ax.set_ylim(-2,2) # Fix axes
        ax.set_title(f't = {t[n+1]:1.3f}')
        plt.pause(0.001) # Slow down animation
        
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise RuntimeError("Must provide save_dir as argument")
    main(sys.argv[1])