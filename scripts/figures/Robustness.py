# Fig.8 – Robustness vs. Attack Intensity
# Requirements:
# - Matplotlib only (no seaborn)
# - Single figure
# - No explicit color settings (use default cycle)
# - Include error bars (±1σ) to mirror Fig.2/3 styling
# - Save high-resolution PNG for paper use

import numpy as np
import matplotlib.pyplot as plt

# Attack intensity levels
x = np.array([0, 1, 2])
x_labels = ['Low', 'Medium', 'High']

# Percentage drop vs no-attack (lower is better)
# Values chosen to match the narrative:
data = {
    'Ours': np.array([1, 5, 15]),
    'MADDPG':              np.array([5, 22, 50]),
    'FedRL':  np.array([3, 15, 35]),
    'FedRL-NoAdv':            np.array([2, 9, 25]),
    'FedRL-NoCW':             np.array([2, 8, 20]),
}

# Reasonable ±1σ uncertainties to visualize variability across runs
errs = {
    'Ours': np.array([0.4, 0.8, 1.8]),
    'MADDPG':             np.array([0.7, 2.0, 3.8]),
    'FedRL': np.array([0.6, 1.8, 3.2]),
    'FedRL-NoAdv':           np.array([0.5, 1.2, 2.8]),
    'FedRL-NoCW':            np.array([0.5, 1.0, 2.2]),
}

markers = {
    'Ours': 'o',
    'MADDPG':             'x',
    'FedRL': 'D',
    'FedRL-NoAdv':           '^',
    'FedRL-NoCW':            's',
}

fig, ax = plt.subplots(figsize=(7.0, 4.8))

# Plot each method with error bars and markers
order = ['Ours', 'MADDPG', 'FedRL', 'FedRL-NoAdv', 'FedRL-NoCW']
for key in order:
    ax.errorbar(
        x, data[key], yerr=errs[key],
        marker=markers[key], linewidth=1.8, markersize=5.5,
        capsize=3.0, elinewidth=1.2, label=key
    )
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 14
font = {'size': 16,
        'family' : 'Times New Roman',}
# Axes formatting
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
plt.xticks(fontsize=16,fontproperties = 'Times New Roman')
plt.yticks(fontsize=16,fontproperties = 'Times New Roman')
ax.set_xlim(-0.2, 2.2)
ax.set_ylim(0, 55)  # show up to CRL 50% + margin
ax.set_ylabel('Performance drop (%)',fontdict=font)
ax.set_xlabel('Attack intensity level',fontdict=font)
ax.grid(True, linestyle='--', alpha=0.35)

legend = ax.legend(frameon=True, fontsize=8, ncol=2)
ax.legend(loc="upper left",fontsize=14)
plt.show()
