# IEEE 118-bus – Two grouped bar charts with value labels
# (1) Convergence speed (episodes to reach 90% accuracy; lower is better)
# (2) Communication overhead per round (KB)
# Methods: Ours, CRL, FedRL, NoAdv, NoCW
# Agent counts: 10, 20, 30, 40
# Notes: matplotlib only, one figure per chart, default colors, labels on each bar.

import numpy as np
import matplotlib.pyplot as plt

agents = np.array([10, 20, 30, 40])
x = np.arange(len(agents))

methods = ['MADDPG', 'FedRL', 'FedRL-NoAdv', 'FedRL-NoCW', 'Ours']
colors = ['orange', 'green', 'red', 'mediumorchid', 'royalblue']

# (1) Convergence speed (episodes to hit 90% accuracy)
# Use 1000 to denote "not reached (NR)" within budget.
conv_crl   = np.array([1000, 1200, 1600, 2000])  # NR
conv_fedrl = np.array([600, 650, 680, 700])  # NR
conv_noadv = np.array([320, 360, 400, 450])
conv_nocw  = np.array([300, 340, 380, 420])
conv_ours  = np.array([220, 230, 240, 250])

conv_data = [conv_crl, conv_fedrl, conv_noadv, conv_nocw, conv_ours]

bar_w = 0.16

fig1, ax1 = plt.subplots(figsize=(8.6, 4.8))
bars_list = []
for i, vals in enumerate(conv_data):
    bars = ax1.bar(x + i*bar_w, vals, width=bar_w, label=methods[i],color=colors[i])
    bars_list.append(bars)
    for rect, v in zip(bars, vals):
        label = f'{int(v)}'
        ax1.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 10,
                 label, ha='center', va='bottom', fontsize=12, fontproperties = 'Times New Roman')

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 16
font = {'size': 16,
        'family' : 'Times New Roman',}

ax1.set_xticks(x + bar_w*(len(methods)-1)/2)
ax1.set_xticklabels([str(a) for a in agents])
plt.xticks(fontsize=16,fontproperties = 'Times New Roman')
plt.yticks(fontsize=16,fontproperties = 'Times New Roman')
ax1.set_xlabel('Number of Agents',fontdict=font)
ax1.set_ylabel('Episodes',fontdict=font)
ax1.set_ylim(0, 2200)
ax1.grid(True, axis='y', linestyle='--', alpha=0.35)
ax1.legend(ncol=3, fontsize=8, frameon=True)
fig1.tight_layout()
ax1.legend(loc="upper left",fontsize=14)
plt.show()