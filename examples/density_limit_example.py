import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from density_limit.metrics import DL26, Greenwald, PlasmaState

filename = "examples/CUTE/plasma_parameters_CUTE.csv"
df = pd.read_csv(filename)  # Replace with actual path to dataframe

times = df["time"]
Ip = df["ip"]
Bt0 = df["BT"]
R0 = df["R0"]
epsilon = df["epsilon"]
a0 = R0 * epsilon
kappa = df["kappa"]
ne = df["n_e"]
ne_edge = df["ne_edge"]
Te_edge = df["Te_edge"]

dl26 = np.zeros_like(times)
f_Gw = np.zeros_like(times)
for i, time in enumerate(times):
    state = PlasmaState(
        R0=R0[i],
        a0=a0[i],
        Bt0=Bt0[i],
        kappa=kappa[i],
        Ip=Ip[i] * 1e6, # Convert from MA to A
        Te_edge=Te_edge[i] * 1e3, # Convert from keV to eV
        ne_edge=ne_edge[i] * 1e20, # Convert from 10^20 m^-3 to m^-3
        ne_mean=ne[i] * 1e20, # Convert from 10^20 m^-3 to m^-3
    )
    dl26_metric = DL26(state, warning_threshold=0.1)
    gw_metric = Greenwald(state)

    dl26[i] = dl26_metric.instability_metric
    f_Gw[i] = gw_metric.f_Gw


fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# Plot DL26 metric and Greenwald fraction over time
ax.plot(times, dl26, label='DL26 Metric', color='tab:blue', linewidth=2)
ax.plot(times, f_Gw, label='Greenwald Fraction', color='tab:orange', linewidth=2)

# Add thresholds 
ax.axhline(1.0, color='tab:green', linestyle='--', label='Greenwald density limit')
ax.axhline(1.0, color='tab:red', linestyle=':', label='DL26 instability threshold')

# Labels and legend
ax.set_xlabel('Time [s]', fontsize=14)
ax.set_ylabel('Normalized Metric', fontsize=14)
ax.set_title('Plasma Stability Metrics Over Time', fontsize=16)
ax.legend()
ax.grid(True)
fig.tight_layout()

os.makedirs("outputs", exist_ok=True)
fig.savefig("outputs/density_limit_metrics.png", dpi=300)