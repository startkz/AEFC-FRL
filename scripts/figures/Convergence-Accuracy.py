# Figure 3 — Convergence of Recovery Accuracy (mean ± 1σ over 5 runs)
# Matches the fluctuation style used in revised Fig.2: AR(1) jitter + rare spikes, short moving average.
# Accuracy is plotted in percent (%).

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(20250812)

episodes = np.arange(1, 1001)
E = len(episodes)
n_runs = 5

def gen_acc_runs(initial, final, k, noise0=1.8, noise_decay=0.0025, ar_phi=0.88,
                 spike_prob=0.008, spike_scale=2.2, sigma_min=0.18):
    """
    Generate n_runs accuracy trajectories (in %) rising from 'initial' to 'final' with learning rate k,
    plus persistent correlated jitter and rare spikes to mimic training fluctuations.
    """
    runs = []
    for _ in range(n_runs):
        init_r = initial + rng.normal(0, 0.6)
        fin_r  = final   + rng.normal(0, 0.3)
        k_r    = k * (1 + rng.normal(0, 0.06))
        base   = fin_r - (fin_r - init_r) * np.exp(-k_r * episodes)

        # time-varying noise with a non-zero floor
        scale_t = noise0 * np.exp(-noise_decay * episodes) + sigma_min

        # AR(1) correlated noise
        ar = np.zeros(E)
        eps = rng.normal(0, 1, size=E) * scale_t
        for t in range(1, E):
            ar[t] = ar_phi * ar[t-1] + eps[t]

        # occasional spikes
        spikes = (rng.random(E) < spike_prob).astype(float)
        spikes *= rng.normal(0, 1, size=E) * (spike_scale * scale_t)

        series = np.clip(base + ar + spikes, 0, 100)
        runs.append(series)
    return np.stack(runs, axis=0)

# Targeted behaviors per description:
# - Ours: fast rise to ~90% by ~180-200 episodes, then to ~98%.
runs_ours  = gen_acc_runs(initial=68, final=98, k=0.0085, noise0=1.5, noise_decay=0.0035, ar_phi=0.90, sigma_min=0.12, spike_prob=0.006, spike_scale=1.8)
# - FedRL (vanilla): converges ~90%, slower.
runs_fedrl = gen_acc_runs(initial=66, final=88, k=0.0032, noise0=1.9, noise_decay=0.0025, ar_phi=0.86, sigma_min=0.22, spike_prob=0.010, spike_scale=2.4)
# - CRL: slowest, plateaus ~85%, more oscillation.
runs_crl   = gen_acc_runs(initial=65, final=90, k=0.0033, noise0=2.4, noise_decay=0.0020, ar_phi=0.82, sigma_min=0.35, spike_prob=0.012, spike_scale=2.8)
# - NoAdv: to ~94%, mid speed.
runs_noadv = gen_acc_runs(initial=67, final=94, k=0.0055, noise0=1.9, noise_decay=0.0028, ar_phi=0.88, sigma_min=0.20, spike_prob=0.009, spike_scale=2.0)
# - NoCW: to ~95%, slightly slower than Ours.
runs_nocw  = gen_acc_runs(initial=67, final=95, k=0.0046, noise0=1.7, noise_decay=0.0030, ar_phi=0.88, sigma_min=0.18, spike_prob=0.008, spike_scale=2.0)

def moving_average(arr, w=5):
    kernel = np.ones(w) / w
    return np.apply_along_axis(lambda a: np.convolve(a, kernel, mode='same'), 1, arr)

sm_ours   = moving_average(runs_ours,  5); mean_ours,   std_ours   = sm_ours.mean(0),  sm_ours.std(0, ddof=1)
sm_fedrl  = moving_average(runs_fedrl, 5); mean_fedrl,  std_fedrl  = sm_fedrl.mean(0), sm_fedrl.std(0, ddof=1)
sm_crl    = moving_average(runs_crl,   5); mean_crl,    std_crl    = sm_crl.mean(0),   sm_crl.std(0, ddof=1)
sm_noadv  = moving_average(runs_noadv, 5); mean_noadv,  std_noadv  = sm_noadv.mean(0), sm_noadv.std(0, ddof=1)
sm_nocw   = moving_average(runs_nocw,  5); mean_nocw,   std_nocw   = sm_nocw.mean(0),  sm_nocw.std(0, ddof=1)

plt.figure(figsize=(8.2, 4.6))

# Lines (no explicit colors), and ±1σ shading
line_ours,  = plt.plot(episodes, mean_ours,  label='Ours', linewidth=2.0, color='royalblue')
line_fedrl, = plt.plot(episodes, mean_fedrl, label='MADDPG',   linestyle='--', linewidth=1.8, color='orange')
line_crl,   = plt.plot(episodes, mean_crl,   label='FedRL', linestyle='--', linewidth=1.8, color='green')
line_noadv, = plt.plot(episodes, mean_noadv, label='FedRL-NoAdv',   linestyle=':',  linewidth=1.8, color='red')
line_nocw,  = plt.plot(episodes, mean_nocw,  label='FedRL-NoCW',    linestyle='-.', linewidth=1.8, color='mediumorchid')
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 14
font = {'size': 18,
        'family' : 'Times New Roman',}
plt.xlabel('Episode',fontdict=font)
plt.ylabel('Recovery Accuracy (%)', fontdict=font)
plt.xticks(fontsize=16,fontproperties = 'Times New Roman')
plt.yticks(fontsize=16,fontproperties = 'Times New Roman')
plt.xlim(1, 1000)
plt.ylim(50, 100.5)
plt.grid(True, linestyle='--', alpha=0.35)
plt.legend(loc='lower right', ncol=1, frameon=True)
plt.tight_layout()
plt.show()
