# Figure 9(d) — Membership Inference Attack AUC (lower is better)
# Style consistent with prior bar charts: fixed color order, value labels, reference line at 0.5.

import numpy as np
import matplotlib.pyplot as plt

# Methods in unified order
methods = ['Ours', 'MADDPG', 'FedRL', 'FedRL-NoAdv', 'FedRL-NoCW']

# AUC values per description
auc_vals = np.array([0.50, 0.90, 0.88, 0.87, 0.70])

# Color scheme consistent with earlier figures
colors = {
    'Ours': 'royalblue',    # 蓝
    'MADDPG': 'orange',     # 橙
    'FedRL': 'green',   # 绿
    'FedRL-NoAdv': 'red',   # 红
    'FedRL-NoCW': 'mediumorchid'     # 紫
}

x = np.arange(len(methods))
bar_w = 0.55

fig, ax = plt.subplots(figsize=(7.2, 4.4))

bars = ax.bar(x, auc_vals, width=bar_w, color=[colors[m] for m in methods])

# Value labels on bars
for rect, v in zip(bars, auc_vals):
    ax.text(rect.get_x() + rect.get_width()/2, v + 0.02, f'{v:.2f}',
            ha='center', va='bottom', fontsize=14)



# Axes & title
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 16
font = {'size': 16,
        'family' : 'Times New Roman',}
ax.set_ylim(0.45, 1.0)
ax.set_ylabel('AUC',fontdict=font)
ax.set_xticks(x)
ax.set_xticklabels(methods)
plt.xticks(fontsize=12,fontproperties = 'Times New Roman')
plt.yticks(fontsize=14,fontproperties = 'Times New Roman')
ax.grid(True, axis='y', linestyle='--', alpha=0.35)

fig.tight_layout()
plt.show()
