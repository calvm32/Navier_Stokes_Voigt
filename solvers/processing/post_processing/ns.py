import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_ns():

    # ---------
    # Load data
    # ---------

    data_path = Path("plot_data.npz") 
    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)

    data = np.load(data_path)

    # unpack
    all_time = data["all_time"]
    energy = data["energy"]
    div = data["divergence"]

    every_time = data["every_time"]
    palinstrophy = data["palinstrophy"]
    stream_func = data["stream_func"]
    enstrophy = data["enstrophy"]

    velocity_x = data["velocity_x"]
    velocity_y = data["velocity_y"]
    omega = data["omega"]

    r_vals = data["r_vals"]
    S2 = data["S2"]

    probe = data["probe"]

    # ----------
    # Divergence
    # ----------

    plt.semilogy(all_time, div, "-o")
    plt.xlabel("time")
    plt.ylabel("L2 divergence")
    plt.title("Divergence vs Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "divergence.png", dpi=200)
    plt.close()

    # ------
    # Energy
    # ------

    plt.semilogy(all_time[10:], energy[10:], "-o")
    plt.xlabel("time")
    plt.ylabel("energy")
    plt.title("Energy vs Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "energy.png", dpi=200)
    plt.close()

    # ------------
    # Palinstrophy
    # ------------

    plt.semilogy(every_time, palinstrophy, "-o")
    plt.xlabel("time")
    plt.ylabel("palinstrophy")
    plt.title("Palinstrophy vs Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "palinstrophy.png", dpi=200)
    plt.close()

    # ---------------
    # Stream function
    # ---------------

    plt.semilogy(every_time, stream_func, "-o")
    plt.xlabel("time")
    plt.ylabel("stream function")
    plt.title("Stream Function vs Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "stream_func.png", dpi=200)
    plt.close()

    # ---------
    # Enstrophy
    # ---------

    plt.semilogy(every_time, enstrophy, "-o")
    plt.xlabel("time")
    plt.ylabel("enstrophy")
    plt.title("Enstrophy vs Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "enstrophy.png", dpi=200)
    plt.close()

    # ----
    # PDFs
    # ----

    plt.hist(velocity_x, bins=100, density=True)
    plt.title("Velocity X PDF")
    plt.grid(True)
    plt.savefig(out_dir / "velx_pdf.png", dpi=200)
    plt.close()

    plt.hist(velocity_y, bins=100, density=True)
    plt.title("Velocity Y PDF")
    plt.grid(True)
    plt.savefig(out_dir / "vely_pdf.png", dpi=200)
    plt.close()

    plt.hist(omega, bins=100, density=True)
    plt.title("Vorticity PDF")
    plt.grid(True)
    plt.savefig(out_dir / "omega_pdf.png", dpi=200)
    plt.close()

    # ------------------
    # Structure function
    # ------------------

    plt.plot(r_vals, S2, "-o")
    plt.xlabel("r")
    plt.ylabel("S2")
    plt.title("Structure Function")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "structure.png", dpi=200)
    plt.close()

    # ---------------
    # Energy spectrum
    # ---------------

    time_vals = every_time
    ux = probe[:, 0] - np.mean(probe[:, 0])

    dt = time_vals[1] - time_vals[0]
    N = len(time_vals)

    u_hat = np.fft.fft(ux)
    f = np.fft.fftfreq(N, d=dt)

    mask = f > 0
    f = f[mask]
    u_hat = u_hat[mask]

    E_f = 2 * (dt / N) * np.abs(u_hat)**2

    U_mean = np.mean(probe[:, 0])
    k = 2*np.pi*f / U_mean
    E_k = E_f * U_mean / (2*np.pi)

    mask = (k > 0) & (E_k > 0)
    k = k[mask]
    E_k = E_k[mask]

    plt.loglog(k, E_k, label="E(k)")

    if len(k) > 6:
        C = E_k[5] * k[5]**(5/3)
        plt.loglog(k, C * k**(-5/3), '--', label="k^-5/3")

    plt.xlabel("k")
    plt.ylabel("E(k)")
    plt.title("Energy Spectrum")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "spectrum.png", dpi=200)
    plt.close()