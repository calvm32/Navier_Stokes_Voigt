import pyvista as pv
import numpy as np
from glob import glob
from scipy.signal import welch
import matplotlib.pyplot as plt

# ------------------------
# USER SETTINGS
# ------------------------

probe_point = np.array([6.5, 2.0, 0.0])
dt = 0.0001   # your timestep
velocity_name = "u"  # change if needed

# ------------------------
# Load files
# ------------------------

files = sorted(glob("Soln/Soln_*.vtu"))

u_time = []

for f in files:
    mesh = pv.read(f)

    idx = mesh.find_closest_point(probe_point)
    velocity = mesh.point_data[velocity_name]

    u_time.append(velocity[idx])

u_time = np.array(u_time)

# ------------------------
# Compute temporal spectrum
# ------------------------

ux = u_time[:,0]
ux = ux - np.mean(ux)

fs = 1/dt

f, E_f = welch(
    ux,
    fs=fs,
    window="hann",
    nperseg=len(ux)//4,
    scaling="spectrum"
)

# ------------------------
# Taylor conversion
# ------------------------

U_mean = np.mean(u_time[:,0])

k = 2*np.pi*f / U_mean
E_k = E_f / U_mean

# ------------------------
# Plot
# ------------------------

plt.figure()
plt.loglog(k[1:], E_k[1:], 'r-', label="Probe (Taylor)")

C1 = E_k[5] * k[5]**3
C2 = E_k[5] * k[5]**(5/3)

plt.loglog(k, C1*k**(-3), '--', label=r"$k^{-3}$")
plt.loglog(k, C2*k**(-5/3), ':', label=r"$k^{-5/3}$")

plt.xlabel("Wavenumber k")
plt.ylabel("E(k)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("probe_energy_spec.png", dpi=200)
plt.show()