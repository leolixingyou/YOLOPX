#!/usr/bin/env python3
"""
可视化TCI实验结果
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# TCI数据
models = ['YOLOPv1', 'YOLOPv2', 'YOLOPv3']
final_tci = [0.2151, 0.2129, 0.2241]

# 各epoch的TCI
tci_history = {
    'YOLOPv1': [0.1987, 0.2395, 0.2020, 0.2319, 0.2036],
    'YOLOPv2': [0.2525, 0.2131, 0.1918, 0.2176, 0.1894],
    'YOLOPv3': [0.2222, 0.2596, 0.2056, 0.1916, 0.2415]
}

# 创建图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 图1: 最终TCI比较
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
bars = ax1.bar(models, final_tci, color=colors, alpha=0.8, edgecolor='black')
ax1.set_ylabel('Task Conflict Intensity (TCI)', fontsize=12)
ax1.set_title('Final TCI Comparison Across YOLOP Models', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 0.3)
ax1.grid(axis='y', alpha=0.3)

# 在柱状图上添加数值
for bar, value in zip(bars, final_tci):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{value:.4f}', ha='center', va='bottom', fontsize=10)

# 图2: TCI随训练变化
epochs = list(range(1, 6))
for model, history, color in zip(models, tci_history.values(), colors):
    ax2.plot(epochs, history, marker='o', label=model, color=color, linewidth=2, markersize=6)

ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Task Conflict Intensity (TCI)', fontsize=12)
ax2.set_title('TCI Evolution During Training', fontsize=14, fontweight='bold')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(epochs)

# 调整布局
plt.tight_layout()

# 保存图片
output_dir = Path(__file__).parent / 'results'
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / 'tci_comparison.png', dpi=300, bbox_inches='tight')
print(f"图表已保存到: {output_dir / 'tci_comparison.png'}")

# 创建详细的TCI分析图
fig2, ax3 = plt.subplots(1, 1, figsize=(10, 6))

# 箱线图显示TCI分布
tci_data = [list(history) for history in tci_history.values()]
box_plot = ax3.boxplot(tci_data, labels=models, patch_artist=True)

# 设置箱线图颜色
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax3.set_ylabel('Task Conflict Intensity (TCI)', fontsize=12)
ax3.set_title('TCI Distribution Across Training Epochs', fontsize=14, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# 添加均值线
means = [np.mean(history) for history in tci_history.values()]
ax3.scatter(range(1, len(models)+1), means, color='red', s=100, zorder=5, label='Mean TCI')
ax3.legend()

plt.tight_layout()
plt.savefig(output_dir / 'tci_distribution.png', dpi=300, bbox_inches='tight')
print(f"分布图已保存到: {output_dir / 'tci_distribution.png'}")

# 打印统计信息
print("\n任务冲突强度(TCI)统计:")
print("="*50)
for model, history in tci_history.items():
    print(f"\n{model}:")
    print(f"  平均值: {np.mean(history):.4f}")
    print(f"  标准差: {np.std(history):.4f}")
    print(f"  最小值: {np.min(history):.4f}")
    print(f"  最大值: {np.max(history):.4f}")
print("="*50)