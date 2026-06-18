import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_ns(data_path):

    # ---------
    # Load data
    # ---------

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

    probes = data["probes"]
    values_at_probes = data["values_at_probes"]

    # ----------
    # Divergence
    # ----------

    plt.semilogy(all_time, div, "-o")
    plt.xlabel("time")
    plt.ylabel("log(L2 of Divergence)")
    plt.title("Divergence vs. Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "divergence.png", dpi=200)
    plt.close()

    # ------
    # Energy
    # ------

    plt.semilogy(all_time[10:], energy[10:], "-o")
    plt.xlabel("time")
    plt.ylabel("log(L2 of Energy)")
    plt.title("Energy vs. Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "energy.png", dpi=200)
    plt.close()

    # ------------
    # Palinstrophy
    # ------------

    plt.semilogy(every_time, palinstrophy, "-o")
    plt.xlabel("time")
    plt.ylabel("log(L2 of Palinstrophy)")
    plt.title("Palinstrophy vs. Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "palinstrophy.png", dpi=200)
    plt.close()

    # ---------------
    # Stream function
    # ---------------

    plt.semilogy(every_time, stream_func, "-o")
    plt.xlabel("time")
    plt.ylabel("log(L2 of Stream Function)")
    plt.title("Stream Function vs. Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "stream_func.png", dpi=200)
    plt.close()

    # ---------
    # Enstrophy
    # ---------

    plt.semilogy(every_time, enstrophy, "-o")
    plt.xlabel("time")
    plt.ylabel("log(L2 of Enstrophy)")
    plt.title("Enstrophy vs. Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "enstrophy.png", dpi=200)
    plt.close()

    # ----
    # PDFs
    # ----

    plt.hist(velocity_x, bins=100, density=True)
    plt.title("Velocity-x Probability Density Function")
    plt.grid(True)
    plt.savefig(out_dir / "velx_pdf.png", dpi=200)
    plt.close()

    plt.hist(velocity_y, bins=100, density=True)
    plt.title("Velocity-y Probability Density Function")
    plt.grid(True)
    plt.savefig(out_dir / "vely_pdf.png", dpi=200)
    plt.close()

    plt.hist(omega, bins=100, density=True)
    plt.title("Velocity-z Probability Density Function")
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

    # -------------------
    # Lift and drag funcs
    # -------------------

    plt.plot(every_time, lift_list, "-o")
    plt.xlabel("time")
    plt.ylabel("Lift Coefficient")
    plt.title("Lift Coefficient vs. Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "lift.png", dpi=200)
    plt.close()

    plt.plot(every_time, drag_list, "-o")
    plt.xlabel("time")
    plt.ylabel("Drag Coefficient")
    plt.title("Drag Coefficient vs. Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "drag.png", dpi=200)
    plt.close()

    # ---------------
    # Energy spectrum
    # ---------------

    time_vals = every_time

    if len(values_at_probes) != 0:
        for i in range(len(probes)):

            probe = probes[i]
            print(probe)
            dim = len(probe)
            
            if dim == 2:
                x, y = probe
            elif dim == 3:
                x, y, z = probe

            values_at_probe = values_at_probes[:,i,:]
            U_mean = np.mean(values_at_probe[:, 0])

            ux = values_at_probe[:, 0] - U_mean

            dt = time_vals[1] - time_vals[0]
            N = len(time_vals)

            u_hat = np.fft.fft(ux)
            f = np.fft.fftfreq(N, d=dt)

            mask = f > 0
            f = f[mask]
            u_hat = u_hat[mask]

            E_f = 2*(dt /N) * np.abs(u_hat)**2

            k = 2*np.pi*f / U_mean
            E_k = E_f * U_mean / (2*np.pi)

            mask = (k > 0) & (E_k > 0)
            k = k[mask]
            E_k = E_k[mask]

            plt.loglog(k, E_k, label="E(k)")

            if len(k) > 6:
                C = E_k[5] * k[5]**(5/3)
                plt.loglog(k, C * k**(-5/3), '--', label="k^-5/3")

            plt.xlabel("log(k)")
            plt.ylabel("log(Energy(k))")
            if dim == 2:
                plt.title(f"Energy Spectrum at Point [{x:2d}, {y:2d}]")
            elif dim == 3:
                plt.title(f"Energy Spectrum at Point [{x:2d}, {y:2d}, {z:2d}]")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(out_dir / f"spectrum_probe{i}.png", dpi=200)
            plt.close()

    # k = data["k"]
    # E_k = data["E_k"]

    # plt.loglog(k, E_k)

    # # optional Kolmogorov fit
    # mask = (k > k[int(len(k)*0.2)]) & (k < k[int(len(k)*0.6)])
    # C = np.mean(E_k[mask] * k[mask]**(5/3))
    # plt.loglog(k, C*k**(-5/3), '--')

    # plt.xlabel("k")
    # plt.ylabel("E(k)")
    # plt.title("Energy Spectrum")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(out_dir / "spectrum_spatial.png", dpi=200)
    # plt.close()