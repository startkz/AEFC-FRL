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

# (2) Communication overhead (KB per round)
comm_fedrl = np.array([500, 1000, 1500, 2000])
comm_noadv = np.array([490, 960, 1440, 1920])
comm_nocw  = np.array([265, 530, 795, 1060])
comm_ours  = np.array([270, 540, 810, 1080])
methods = ['FedRL', 'FedRL-NoAdv', 'FedRL-NoCW', 'Ours']
colors = ['green', 'red', 'mediumorchid', 'royalblue']
comm_data = [comm_fedrl, comm_noadv, comm_nocw, comm_ours]

fig2, ax2 = plt.subplots(figsize=(8.6, 4.8))
bars_list2 = []
for i, vals in enumerate(comm_data):
    bars = ax2.bar(x + i*bar_w, vals, width=bar_w, label=methods[i],color=colors[i])
    bars_list2.append(bars)
    for rect, v in zip(bars, vals):
        ax2.text(rect.get_x() + rect.get_width()/2, rect.get_height() + (200 if v>3000 else 25),
                 f'{int(v)}', ha='center', va='bottom', fontproperties = 'Times New Roman')
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 16
font = {'size': 16,
        'family' : 'Times New Roman',}
ax2.set_xticks(x + bar_w*(len(methods)-1)/2)
ax2.set_xticklabels([str(a) for a in agents])
plt.xticks(fontsize=16,fontproperties = 'Times New Roman')
plt.yticks(fontsize=16,fontproperties = 'Times New Roman')
ax2.set_xlabel('Number of Agents',fontdict=font)
ax2.set_ylabel('Communication per Round/KB',fontdict=font)
ax2.set_ylim(0, 2500)
ax2.grid(True, axis='y', linestyle='--', alpha=0.35)
ax2.legend(ncol=3, fontsize=8, frameon=True)
ax2.legend(loc="upper left",fontsize=14)
fig2.tight_layout()
plt.show()
