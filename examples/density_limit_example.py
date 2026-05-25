import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from density_limit.metrics import DL26, Greenwald, PlasmaState, calc_q_star, calc_beta_edge, calc_collisionality_edge

# Read plasma data
filename = "examples/CUTE/plasma_parameters_CUTE.csv"
df = pd.read_csv(filename)  # Replace with actual path to dataframe

# Convert units and extract parameters
times = df["time"]
Ip = df["ip"] * 1e6  # Convert from [MA] to [A]
Bt0 = df["BT"]
R0 = df["R0"]
epsilon = df["epsilon"]
a0 = R0 * epsilon
kappa = df["kappa"]
ne = df["n_e"] * 1e20  # Convert from [10^20 m^-3] to [m^-3]
ne_edge = df["ne_edge"] * 1e20  # Convert from [10^20 m^-3] to [m^-3]
Te_edge = df["Te_edge"] * 1e3  # Convert from [keV] to [eV]

# Calculate limits
dl26 = np.zeros_like(times)
f_Gw = np.zeros_like(times)
dens_lim = np.zeros_like(times)
nu = np.zeros_like(times)
beta = np.zeros_like(times)
q_star = np.zeros_like(times)
for i, time in enumerate(times):
    state = PlasmaState(
        R0=R0[i],
        a0=a0[i],
        Bt0=Bt0[i],
        kappa=kappa[i],
        Ip=Ip[i],
        Te_edge=Te_edge[i],
        ne_edge=ne_edge[i],
        ne_mean=ne[i],
    )
    dl26_metric = DL26(state, warning_threshold=0.1)
    gw_metric = Greenwald(state)

    dl26[i] = dl26_metric.instability_metric
    dens_lim[i] = dl26_metric.to_limit_density()
    f_Gw[i] = gw_metric.f_Gw

    q_star[i] = calc_q_star(
        Bt0=state.Bt0,
        R0=state.R0,
        epsilon=state.a0 / state.R0,
        kappa=state.kappa,
        Ip=state.Ip,
    )
    nu[i] = calc_collisionality_edge(
        ne_edge=state.ne_edge,
        Te_edge=state.Te_edge,
        q_star=q_star[i],
        R0=state.R0,
        epsilon=state.a0 / state.R0,
    )
    beta[i] = calc_beta_edge(
        ne_edge=state.ne_edge,
        Te_edge=state.Te_edge,
        Bt0=state.Bt0,
        R0=state.R0,
        a0=state.a0,
        Ip=state.Ip,
    )

# PLOT
fig, ax = plt.subplots(2, 1, figsize=(12, 8))

# Plot DL26 metric and Greenwald fraction over time
ax[0].plot(times, dl26, label='DL26 Metric', color='tab:blue', linewidth=2)
ax[0].plot(times, f_Gw, label='Greenwald Fraction', color='tab:orange', linewidth=2)

# Add thresholds 
ax[0].axhline(1.0, color='tab:green', linestyle='--', label='Greenwald density limit')
ax[0].axhline(1.0, color='tab:red', linestyle=':', label='DL26 instability threshold')

# Labels and legend
ax[0].set_xlabel('Time [s]', fontsize=14)
ax[0].set_ylabel('Normalized Metric', fontsize=14)
ax[0].set_title('Plasma Stability Metrics Over Time', fontsize=16)
ax[0].legend()
ax[0].grid(True)

ax[1].plot(times, dens_lim * 1e-20, label='DL26 Density Limit [10^20 m^-3]', color='tab:purple', linewidth=2)
ax[1].plot(times, ne * 1e-20, label='Mean Density [10^20 m^-3]', color='tab:cyan', linewidth=2)
ax[1].plot(times, ne_edge * 1e-20, label='Edge Density [10^20 m^-3]', color='tab:gray', linewidth=2)
ax[1].set_xlabel('Time [s]', fontsize=14)
ax[1].set_ylabel('Density [10^20 m^-3]', fontsize=14)
ax[1].set_title('Plasma Density Over Time', fontsize=16)
ax[1].legend()
ax[1].grid(True)

fig.tight_layout()

os.makedirs("outputs", exist_ok=True)
fig.savefig("outputs/density_limit_metrics.png", dpi=300)
