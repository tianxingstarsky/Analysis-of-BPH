# -*- coding: utf-8 -*-
"""Generate the summary figures used in README (ablation bars, MedSAM2 comparison, pipeline)."""
import os

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')

# ---------------- 1. ablation 3D Dice bar chart ----------------
configs = [
    ('无增强\n无SE', 0.8725, '#2e7d32'),
    ('geo增强\n+SE', 0.8139, '#1565c0'),
    ('bg增强\n+SE', 0.8057, '#1565c0'),
    ('full增强\n无SE', 0.8049, '#9e9e9e'),
    ('full增强\n+SE (v3)', 0.7662, '#1565c0'),
    ('旧基线\nv2.1', 0.7090, '#757575'),
    ('warp增强\n+SE', 0.3613, '#c62828'),
    ('无增强\n+SE', 0.1481, '#c62828'),
]
fig, ax = plt.subplots(figsize=(9, 4.6))
bars = ax.bar([c[0] for c in configs], [c[1] for c in configs], color=[c[2] for c in configs], width=0.62)
for b, (_, v, _) in zip(bars, configs):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f'{v:.3f}', ha='center', fontsize=10)
ax.axhline(0.9356, color='#e65100', ls='--', lw=1.4)
ax.text(6.9, 0.945, 'MedSAM2 零样本\n(每层框提示) 0.936', color='#e65100', fontsize=9, ha='center')
ax.set_ylim(0, 1.02)
ax.set_ylabel('逐病例 3D 体积 Dice')
ax.set_title('增强方案 × 通道注意力 消融对比（MSD Task05 前列腺，按病例验证集）')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig_ablation_3d.png'), dpi=160)
plt.close()

# ---------------- 2. MedSAM2 per-case comparison ----------------
cases = ['p06', 'p18', 'p24', 'p40']
unet_best = [0.8242, 0.9011, 0.8379, 0.8790]   # none+noSE per-case 3D
medsam2_box = [0.9341, 0.9459, 0.9374, 0.9248]
medsam2_prop = [0.6290, 0.6682, 0.5862, 0.6637]
import numpy as np
x = np.arange(4)
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.bar(x - 0.26, unet_best, 0.24, label='U-Net 自训练最优', color='#2e7d32')
ax.bar(x, medsam2_box, 0.24, label='MedSAM2 每层框提示', color='#e65100')
ax.bar(x + 0.26, medsam2_prop, 0.24, label='MedSAM2 单框传播', color='#90a4ae')
for xi, v in zip(x - 0.26, unet_best):
    ax.text(xi, v + 0.01, f'{v:.2f}', ha='center', fontsize=8)
for xi, v in zip(x, medsam2_box):
    ax.text(xi, v + 0.01, f'{v:.2f}', ha='center', fontsize=8)
ax.set_xticks(x, cases)
ax.set_ylim(0, 1.02)
ax.set_ylabel('逐病例 3D 体积 Dice')
ax.set_title('U-Net（全自动） vs MedSAM2（框提示交互式）分病例对比')
ax.legend(fontsize=9, loc='lower right')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig_medsam2_per_case.png'), dpi=160)
plt.close()

# ---------------- 3. pipeline diagram ----------------
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.axis('off')
boxes = [
    (0.02, 0.55, '多序列 MRI\n(T2WI / ADC)\nDICOM / NIfTI', '#e3f2fd'),
    (0.21, 0.55, '数据预处理\n病例级 z-score\n中心裁剪 176px', '#e8f5e9'),
    (0.40, 0.72, 'Phase 1a 分割\nU-Net(+SE)\n全自动 3D Dice 0.873', '#fff3e0'),
    (0.40, 0.30, 'Phase 1b 分割\nMedSAM2 框提示\n零样本 3D Dice 0.936', '#fff3e0'),
    (0.62, 0.51, 'Phase 2 量化\n质地/形态特征\n+ 临床量表', '#f3e5f5'),
    (0.83, 0.51, 'Phase 3 预测\n手术必要性\n(组学+临床融合)', '#ffebee'),
]
for x, y, t, c in boxes:
    ax.add_patch(plt.Rectangle((x, y), 0.15, 0.30, facecolor=c, edgecolor='#555', lw=1.2))
    ax.text(x + 0.075, y + 0.15, t, ha='center', va='center', fontsize=10)
arr = dict(arrowstyle='->', lw=1.6, color='#444')
ax.annotate('', xy=(0.208, 0.70), xytext=(0.17, 0.70), arrowprops=arr)
ax.annotate('', xy=(0.208, 0.45), xytext=(0.17, 0.45), arrowprops=arr)
ax.annotate('', xy=(0.398, 0.70), xytext=(0.36, 0.70), arrowprops=arr)
ax.annotate('', xy=(0.398, 0.45), xytext=(0.36, 0.45), arrowprops=arr)
ax.annotate('', xy=(0.618, 0.66), xytext=(0.55, 0.66), arrowprops=arr)
ax.annotate('', xy=(0.618, 0.45), xytext=(0.55, 0.45), arrowprops=arr)
ax.annotate('', xy=(0.828, 0.66), xytext=(0.77, 0.66), arrowprops=arr)
ax.annotate('', xy=(0.828, 0.45), xytext=(0.77, 0.45), arrowprops=arr)
ax.text(0.55, 0.14, '本研究完成 Phase 1a / 1b / 2；Phase 3 需医院数据（手术结局标签）到位后开展',
        ha='center', fontsize=10, color='#666')
ax.set_title('技术路线：分割 → 量化 → 手术必要性预测', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig_pipeline.png'), dpi=160)
plt.close()
print('figures saved:', [f for f in os.listdir(FIG) if f.startswith('fig_')])
