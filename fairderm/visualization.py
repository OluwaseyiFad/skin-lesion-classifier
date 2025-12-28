# plotting functions

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_history(history, title='Training History', save_path=None):
    # loss and accuracy curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    ax1 = axes[0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_fairness_comparison(results_df, save_path=None):
    # compare fairness across models
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    models = results_df['model'].tolist() if 'model' in results_df else results_df['name'].tolist()
    x = np.arange(len(models))

    ax1 = axes[0, 0]
    ax1.bar(x, results_df['balanced_accuracy'], color='steelblue', alpha=0.8)
    ax1.set_ylabel('Balanced Accuracy')
    ax1.set_title('Overall Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_ylim(0, 1)
    for i, v in enumerate(results_df['balanced_accuracy']):
        ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)

    ax2 = axes[0, 1]
    width = 0.25
    ax2.bar(x - width, results_df['tpr_light'], width, label='Light', color='#2ecc71')
    ax2.bar(x, results_df['tpr_medium'], width, label='Medium', color='#f39c12')
    ax2.bar(x + width, results_df['tpr_dark'], width, label='Dark', color='#9b59b6')
    ax2.set_ylabel('True Positive Rate (Sensitivity)')
    ax2.set_title('Per-Group TPR')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend()
    ax2.set_ylim(0, 1)

    ax3 = axes[1, 0]
    colors = ['#e74c3c' if v > 0.15 else '#f39c12' if v > 0.10 else '#2ecc71'
              for v in results_df['equal_opportunity_gap']]
    ax3.bar(x, results_df['equal_opportunity_gap'], color=colors, alpha=0.8)
    ax3.set_ylabel('Equal Opportunity Gap (lower = fairer)')
    ax3.set_title('Fairness Gap Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.axhline(y=0.10, color='green', linestyle='--', alpha=0.5,
                label='Target (<10%)')
    ax3.legend()
    for i, v in enumerate(results_df['equal_opportunity_gap']):
        ax3.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

    ax4 = axes[1, 1]
    scatter = ax4.scatter(
        results_df['balanced_accuracy'],
        results_df['equal_opportunity_gap'],
        s=150,
        c=results_df['tpr_dark'],
        cmap='RdYlGn',
        edgecolors='black',
        linewidth=1
    )
    for i, model in enumerate(models):
        ax4.annotate(
            model,
            (results_df['balanced_accuracy'].iloc[i],
             results_df['equal_opportunity_gap'].iloc[i]),
            xytext=(5, 5), textcoords='offset points', fontsize=8
        )
    ax4.set_xlabel('Balanced Accuracy (higher = better)')
    ax4.set_ylabel('Equal Opportunity Gap (lower = fairer)')
    ax4.set_title('Accuracy vs Fairness Trade-off')
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Dark Skin TPR')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_group_comparison(group_metrics, metric='accuracy', save_path=None):
    # compare metric across skin tone groups
    models = list(group_metrics.keys())
    groups = ['Light', 'Medium', 'Dark']
    colors = {'Light': '#2ecc71', 'Medium': '#f39c12', 'Dark': '#9b59b6'}

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, group in enumerate(groups):
        values = [group_metrics[m][group].get(metric, 0) for m in models]
        offset = (i - 1) * width
        ax.bar(x + offset, values, width, label=group, color=colors[group])

    ax.set_xlabel('Model')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'Per-Group {metric.replace("_", " ").title()} Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrices(confusion_matrices, model_names, save_path=None):
    # confusion matrices side by side
    n_models = len(confusion_matrices)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4))

    if n_models == 1:
        axes = [axes]

    for ax, cm, name in zip(axes, confusion_matrices, model_names):
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant']
        )
        ax.set_title(name)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
