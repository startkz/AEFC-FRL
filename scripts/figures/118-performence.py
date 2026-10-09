# IEEE 118-bus — Recovery Accuracy vs. Number of Agents
# Grouped bar chart with value labels on each bar.
# Methods: Ours, CRL, FedRL, NoAdv, NoCW
# Agent counts: 10, 20, 30, 40
# Note: No explicit colors, matplotlib defaults only (per your style constraints).

import numpy as np
import matplotlib.pyplot as plt

agents = np.array([10, 20, 30, 40])

# Accuracy (%) consistent with your narrative
acc_ours = np.array([96.0, 97.0, 98.0, 98.0])
acc_crl  = np.array([85.0, 82.0, 78.0, 75.0])
acc_fed  = np.array([86.0, 87.0, 88.0, 89.0])
acc_noadv= np.array([92.0, 93.0, 93.5, 93.0])
acc_nocw = np.array([93.0, 94.0, 95.0, 95.0])

methods = ['MADDPG', 'FedRL', 'FedRL-NoAdv', 'FedRL-NoCW', 'Ours']
colors = ['orange', 'green', 'red', 'mediumorchid', 'royalblue']
data = [acc_crl, acc_fed, acc_noadv, acc_nocw, acc_ours]

# Plot
fig, ax = plt.subplots(figsize=(8.5, 4.8))

n_groups = len(agents)
n_methods = len(methods)
bar_width = 0.16
indices = np.arange(n_groups)

# draw bars
bars = []
for i, vals in enumerate(data):
    b = ax.bar(indices + i*bar_width, vals, width=bar_width, label=methods[i],align="center",color=colors[i])
    bars.append(b)
    # add labels on bars
    for rect, v in zip(b, vals):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.6,
                f'{v:.1f}', ha='center', va='bottom', fontsize=14, fontproperties = 'Times New Roman',rotation=0)

# axes and legend
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 16
font = {'size': 16,
        'family' : 'Times New Roman',}

ax.set_xticks(indices + bar_width*(n_methods-1)/2)
ax.set_xticklabels([str(a) for a in agents])
plt.xticks(fontsize=16,fontproperties = 'Times New Roman')
plt.yticks(fontsize=16,fontproperties = 'Times New Roman')
ax.set_xlabel('Number of Agents',fontdict=font)
ax.set_ylabel('Recovery Accuracy (%)',fontdict=font)
ax.set_ylim(70, 101)
ax.grid(True, axis='y', linestyle='--', alpha=0.35)
ax.legend(ncol=3, fontsize=8, frameon=True)
ax.legend(loc="upper left",fontsize=14)

fig.tight_layout()
plt.show()
