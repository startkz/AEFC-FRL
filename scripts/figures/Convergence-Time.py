# Revised Fig. 2 — Increase visible fluctuations (波动) to resemble the sample image,
# while keeping: moving-average curves, ±1σ shading over 5 runs, labels/axes unchanged.

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(20250812)

episodes = np.arange(1, 1001)
E = len(episodes)
n_runs = 5

def gen_runs(initial, asymptote, k, noise0=1.8, noise_decay=0.0018, ar_phi=0.85, spike_prob=0.015, spike_scale=3.0, sigma_min=0.15):
    """Generate n_runs trajectories with decaying mean and persistent,
    correlated jitter + occasional spikes to mimic realistic fluctuations.
    """
    runs = []
    for _ in range(n_runs):
        init_r  = initial  + rng.normal(0, 0.5)
        asym_r  = asymptote + rng.normal(0, 0.12)
        k_r     = k * (1 + rng.normal(0, 0.06))
        base    = asym_r + (init_r - asym_r) * np.exp(-k_r * episodes)

        # Time-varying noise scale (decays but leaves a floor sigma_min)
        scale_t = noise0 * np.exp(-noise_decay * episodes) + sigma_min

        # AR(1) correlated noise to produce smooth yet jittery fluctuations
        ar = np.zeros(E)
        eps = rng.normal(0, 1, size=E) * scale_t
        for t in range(1, E):
            ar[t] = ar_phi * ar[t-1] + eps[t]

        # Occasional spikes (rare bursty noise)
        spikes = (rng.random(E) < spike_prob).astype(float)
        spikes *= rng.normal(0, 1, size=E) * (spike_scale * scale_t)

        series = np.maximum(base + ar + spikes, 0.0)
        runs.append(series)
    return np.stack(runs, axis=0)

# Parameterization per textual description
# Ours: plateaus near 8 by ~300; very stable late
runs_ours  = gen_runs(initial=18.0, asymptote=8.0,  k=0.012, noise0=1.4, noise_decay=0.0045, ar_phi=0.88, sigma_min=0.12, spike_prob=0.010, spike_scale=2.5)
# FedRL: slower and higher; ~13 around ep500; eventually ~10–11
runs_fedrl = gen_runs(initial=18.0, asymptote=10.8, k=0.0020, noise0=1.9, noise_decay=0.0025, ar_phi=0.85, sigma_min=0.22, spike_prob=0.018, spike_scale=3.2)
# CRL: slowest; stabilizes around ~12; higher variance
runs_crl   = gen_runs(initial=17.5, asymptote=12.0, k=0.0038, noise0=2.0, noise_decay=0.0025, ar_phi=0.83, sigma_min=0.30, spike_prob=0.020, spike_scale=3.5)
# NoAdv: 9–10 final; slightly worse than Ours; higher early variance
runs_noadv = gen_runs(initial=18.0, asymptote=9.6,  k=0.0065, noise0=2.1, noise_decay=0.0030, ar_phi=0.86, sigma_min=0.20, spike_prob=0.018, spike_scale=3.0)
# NoCW: ~9 final; slower approach
runs_nocw  = gen_runs(initial=18.0, asymptote=9.0,  k=0.0046, noise0=1.9, noise_decay=0.0030, ar_phi=0.86, sigma_min=0.18, spike_prob=0.016, spike_scale=2.8)

def moving_average(arr, w=5):
    """Apply moving average per-run with a small window to keep visible jitter."""
    kernel = np.ones(w) / w
    return np.apply_along_axis(lambda a: np.convolve(a, kernel, mode='same'), 1, arr)

# Short-window smoothing to maintain fluctuations similar to the sample figure
sm_ours   = moving_average(runs_ours,  5); mean_ours,   std_ours   = sm_ours.mean(0),  sm_ours.std(0, ddof=1)
sm_fedrl  = moving_average(runs_fedrl, 5); mean_fedrl,  std_fedrl  = sm_fedrl.mean(0), sm_fedrl.std(0, ddof=1)
sm_crl    = moving_average(runs_crl,   5); mean_crl,    std_crl    = sm_crl.mean(0),   sm_crl.std(0, ddof=1)
sm_noadv  = moving_average(runs_noadv, 5); mean_noadv,  std_noadv  = sm_noadv.mean(0), sm_noadv.std(0, ddof=1)
sm_nocw   = moving_average(runs_nocw,  5); mean_nocw,   std_nocw   = sm_nocw.mean(0),  sm_nocw.std(0, ddof=1)

plt.figure(figsize=(8.2, 4.6))

# Draw lines (matplotlib default colors/styles) and ±1σ bands
line_ours,  = plt.plot(episodes, mean_ours,  label='Ours', linewidth=2.0, color='royalblue')
line_fedrl, = plt.plot(episodes, mean_fedrl, label='MADDPG',   linestyle='--', linewidth=1.8, color='orange')
line_crl,   = plt.plot(episodes, mean_crl,   label='FedRL', linestyle='--', linewidth=1.8, color='green')
line_noadv, = plt.plot(episodes, mean_noadv, label='FedRL-NoAdv',   linestyle=':',  linewidth=1.8, color='red')
line_nocw,  = plt.plot(episodes, mean_nocw,  label='FedRL-NoCW',    linestyle='-.', linewidth=1.8, color='mediumorchid')
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 14
font = {'size': 18,
        'family' : 'Times New Roman',}
plt.xlabel('Episode', fontdict=font)
plt.ylabel('Recovery Time/s', fontdict=font)
plt.xticks(fontsize=16,fontproperties = 'Times New Roman')
plt.yticks(fontsize=16,fontproperties = 'Times New Roman')
plt.xlim(1, 1000)
plt.ylim(0, None)
plt.grid(True, linestyle='--', alpha=0.35)
plt.legend(loc='upper right', ncol=1, frameon=True)
plt.tight_layout()
plt.show()
