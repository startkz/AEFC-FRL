import matplotlib.pyplot as plt
import numpy as np

# 数据（IEEE 39-bus恢复精度百分比）
labels = ['0%', '10%', '20%', '30%']  # 拜占庭代理比例
ours = [98, 92, 89, 83]
crl = [85, 71, 63, 43]
fedrl = [88, 75, 73, 65]
noadv = [93, 87, 83, 75]
nocw = [94, 90, 86, 80]

# 颜色方案（与扩展性实验统一）
colors = {
    'Ours': 'royalblue',    # 蓝
    'MADDPG': 'orange',     # 橙
    'FedRL': 'green',   # 绿
    'FedRL-NoAdv': 'red',   # 红
    'FedRL-NoCW': 'mediumorchid'     # 紫
}

x = np.arange(len(labels))
bar_width = 0.15

fig, ax = plt.subplots(figsize=(8, 5))

# 绘制柱状图
rects1 = ax.bar(x - 2*bar_width, ours, bar_width, label='Ours', color=colors['Ours'])
rects2 = ax.bar(x - bar_width, crl, bar_width, label='MADDPG', color=colors['MADDPG'])
rects3 = ax.bar(x, fedrl, bar_width, label='FedRL', color=colors['FedRL'])
rects4 = ax.bar(x + bar_width, noadv, bar_width, label='FedRL-NoAdv', color=colors['FedRL-NoAdv'])
rects5 = ax.bar(x + 2*bar_width, nocw, bar_width, label='FedRL-NoCW', color=colors['FedRL-NoCW'])

# 数值标签函数
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

for rect_set in [rects1, rects2, rects3, rects4, rects5]:
    autolabel(rect_set)


plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 16
font = {'size': 16,
        'family' : 'Times New Roman',}
# 坐标轴与标题
ax.set_xlabel('Byzantine Agent Proportion',fontdict=font)
ax.set_ylabel('Recovery Accuracy/%',fontdict=font)
ax.set_xticks(x)
ax.set_xticklabels(labels)
plt.xticks(fontsize=16,fontproperties = 'Times New Roman')
plt.yticks(fontsize=16,fontproperties = 'Times New Roman')
ax.grid(True, axis='y', linestyle='--', alpha=0.35)
ax.set_ylim(0, 110)  # 留出顶部空间
ax.legend()
ax.legend(loc="upper right",fontsize=14)
plt.tight_layout()
plt.show()
