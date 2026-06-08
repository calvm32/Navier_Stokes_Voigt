import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_heat():
        
    # ---------
    # Load data
    # ---------

    data_path = Path("plot.npz")  # change as needed
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