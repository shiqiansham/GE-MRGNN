import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_curves():
    # Set global plotting parameters for academic style
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'lines.linewidth': 2,
        'figure.dpi': 300
    })

    base_dir = Path(r"F:\GranduateDesign\DataGet\results")
    out_dir = Path(r"F:\GranduateDesign\DataGet\images")
    out_dir.mkdir(parents=True, exist_ok=True)

    models = {
        'ge-mrgcn': ('GE-MRGNN (Ours)', '#E63946', '-'),  # Red
        'rgcn': ('R-GCN', '#457B9D', '--'),               # Blue
        'han': ('HAN', '#2A9D8F', '-.'),                  # Teal
        'gcn': ('GCN', '#F4A261', ':')                    # Orange
    }

    # Plot ROC Curves
    plt.figure(figsize=(7, 5.5))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5, label='Random Guess')

    for model_key, (model_name, color, linestyle) in models.items():
        roc_path = base_dir / model_key / 'roc.csv'
        if roc_path.exists():
            df = pd.read_csv(roc_path)
            # Assuming columns are threshold, fpr, tpr
            fpr = df['fpr']
            tpr = df['tpr']
            plt.plot(fpr, tpr, label=model_name, color=color, linestyle=linestyle)

    plt.title('ROC Curve Comparison')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right', frameon=True, shadow=False, edgecolor='black')
    plt.tight_layout()
    plt.savefig(out_dir / 'roc_academic.png', format='png', dpi=300)
    plt.savefig(out_dir / 'roc_academic.svg', format='svg')
    plt.close()

    # Plot PR Curves
    plt.figure(figsize=(7, 5.5))
    plt.grid(True, linestyle='--', alpha=0.6)

    for model_key, (model_name, color, linestyle) in models.items():
        pr_path = base_dir / model_key / 'pr.csv'
        if pr_path.exists():
            df = pd.read_csv(pr_path)
            # Assuming columns are threshold, precision, recall
            precision = df['precision']
            recall = df['recall']
            plt.plot(recall, precision, label=model_name, color=color, linestyle=linestyle)

    plt.title('Precision-Recall (PR) Curve Comparison')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower left', frameon=True, shadow=False, edgecolor='black')
    plt.tight_layout()
    plt.savefig(out_dir / 'pr_academic.png', format='png', dpi=300)
    plt.savefig(out_dir / 'pr_academic.svg', format='svg')
    plt.close()

    print("Academic plots generated successfully at", out_dir)

if __name__ == '__main__':
    plot_curves()