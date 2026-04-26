"""生成论文中所有公式的高清 PNG 图片"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUT_DIR = 'images'
os.makedirs(OUT_DIR, exist_ok=True)

# 公式定义: (文件名, LaTeX公式, 标签)
formulas = [
    # 公式 (1.1) GCN 传播规则
    (
        'formula_1_1_gcn.png',
        r'$H^{(l+1)} = \sigma\!\left(\tilde{D}^{-\frac{1}{2}}\,\tilde{A}\,\tilde{D}^{-\frac{1}{2}}\,H^{(l)}\,W^{(l)}\right)$',
        '公式 (1.1) GCN'
    ),
    # 公式 (1.2) GAT 注意力 - 分两行展示
    (
        'formula_1_2_gat.png',
        r'$\alpha_{ij} = \frac{\exp\left(\mathrm{LeakyReLU}\left(\vec{a}^{T}[Wh_i \| Wh_j]\right)\right)}{\sum_{k \in \mathcal{N}_i} \exp\left(\mathrm{LeakyReLU}\left(\vec{a}^{T}[Wh_i \| Wh_k]\right)\right)}$',
        '公式 (1.2) GAT'
    ),
    # 公式 (3.1) R-GCN 更新 (修正版，含 h_i 和 h_j)
    (
        'formula_3_1_rgcn.png',
        r'$h_i^{(l+1)} = \sigma\!\left(W_0^{(l)}\,h_i^{(l)} + \sum_{r \in \mathcal{R}}\;\sum_{j \in \mathcal{N}_i^r} \frac{1}{c_{i,r}}\,W_r^{(l)}\,h_j^{(l)}\right)$',
        '公式 (3.1) R-GCN (修正)'
    ),
    # 公式 (3.2) GroupEnhance
    (
        'formula_3_2_group.png',
        r"$h_i' = h_i + \alpha \cdot \mathrm{proj}\!\left(\frac{1}{|\mathcal{N}_i|}\sum_{j \in \mathcal{N}_i} h_j\right)$",
        '公式 (3.2) GroupEnhance'
    ),
    # 公式 (3.3) BCEWithLogitsLoss
    (
        'formula_3_3_loss.png',
        r'$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\left[w_+\!\cdot\! y_i \log\!\left(\hat{y}_i\right) + \left(1-y_i\right)\log\!\left(1-\hat{y}_i\right)\right]$',
        '公式 (3.3) Loss'
    ),
    # 公式 (4.1) Accuracy
    (
        'formula_4_1_acc.png',
        r'$\mathrm{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$',
        '公式 (4.1) Accuracy'
    ),
    # 公式 (4.2) Precision
    (
        'formula_4_2_prec.png',
        r'$\mathrm{Precision} = \frac{TP}{TP + FP}$',
        '公式 (4.2) Precision'
    ),
    # 公式 (4.3) Recall
    (
        'formula_4_3_recall.png',
        r'$\mathrm{Recall} = \frac{TP}{TP + FN}$',
        '公式 (4.3) Recall'
    ),
]

for fname, latex, label in formulas:
    fig, ax = plt.subplots(figsize=(8, 1.2))
    ax.text(0.5, 0.5, latex, fontsize=22, ha='center', va='center',
            transform=ax.transAxes)
    ax.axis('off')
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.15,
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'[OK] {path}  ({label})')

print('\n全部公式图片生成完成！')
