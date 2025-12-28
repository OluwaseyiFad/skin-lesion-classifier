# metrics for evaluating model performance and fairness
# the goal is to measure both accuracy AND fairness across skin tones

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, confusion_matrix
)


def compute_metrics(y_true, y_pred, y_prob=None):
    # standard classification metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }

    # add roc-auc if we have probability predictions
    if y_prob is not None:
        if len(np.unique(y_true)) > 1:  # need both classes
            # handle both (n,2) and (n,) shaped probs
            if len(y_prob.shape) > 1 and y_prob.shape[1] == 2:
                y_prob_pos = y_prob[:, 1]
            else:
                y_prob_pos = y_prob
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob_pos)

    return metrics


def compute_group_metrics(y_true, y_pred, groups, y_prob=None,
                          group_names=['Light', 'Medium', 'Dark']):
    # same metrics but broken down by skin tone group
    # useful to see if model performs differently on different groups
    group_metrics = {}

    for idx, group_name in enumerate(group_names):
        mask = groups == idx
        if mask.sum() > 0:
            group_y_true = y_true[mask]
            group_y_pred = y_pred[mask]
            group_y_prob = y_prob[mask] if y_prob is not None else None

            group_metrics[group_name] = compute_metrics(
                group_y_true, group_y_pred, group_y_prob
            )
            group_metrics[group_name]['n_samples'] = int(mask.sum())

    return group_metrics


def compute_fairness_metrics(y_true, y_pred, groups,
                             group_names=['Light', 'Medium', 'Dark']):
    # key fairness metrics - all measure disparities between groups
    # lower gap = fairer model

    # equal opportunity gap: max TPR - min TPR across groups
    #   (do we detect disease equally well for all skin tones?)
    # equalized odds gap: max of TPR gap and FPR gap
    #   (are error rates similar across groups?)
    # f1 gap: max F1 - min F1 across groups

    results = {
        'tpr_per_group': {},
        'fpr_per_group': {},
        'f1_per_group': {},
        'support_per_group': {}
    }

    for idx, group_name in enumerate(group_names):
        mask = groups == idx
        if mask.sum() > 0:
            group_y_true = y_true[mask]
            group_y_pred = y_pred[mask]

            # get confusion matrix components
            cm = confusion_matrix(group_y_true, group_y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            # TPR = sensitivity = recall = how often we catch positives
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            # FPR = false alarm rate = how often we wrongly flag negatives
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            f1 = f1_score(group_y_true, group_y_pred, average='binary',
                          pos_label=1, zero_division=0)

            results['tpr_per_group'][group_name] = tpr
            results['fpr_per_group'][group_name] = fpr
            results['f1_per_group'][group_name] = f1
            results['support_per_group'][group_name] = int(mask.sum())

    tpr_values = list(results['tpr_per_group'].values())
    fpr_values = list(results['fpr_per_group'].values())
    f1_values = list(results['f1_per_group'].values())

    if not tpr_values or not fpr_values or not f1_values:
        results['equal_opportunity_gap'] = 0.0
        results['equalized_odds_gap'] = 0.0
        results['f1_gap'] = 0.0
        return results

    # compute the gaps
    results['equal_opportunity_gap'] = max(tpr_values) - min(tpr_values)
    results['equalized_odds_gap'] = max(
        max(tpr_values) - min(tpr_values),
        max(fpr_values) - min(fpr_values)
    )
    results['f1_gap'] = max(f1_values) - min(f1_values)

    return results


def print_fairness_report(fairness_metrics):
    # nice formatted printout of fairness results
    print("\n" + "=" * 60)
    print("FAIRNESS EVALUATION REPORT")
    print("=" * 60)

    print("\nPer-Group True Positive Rates (Sensitivity):")
    for group, tpr in fairness_metrics['tpr_per_group'].items():
        n = fairness_metrics['support_per_group'][group]
        print(f"  {group}: {tpr:.4f} (n={n})")
    print(f"  -> Equal Opportunity Gap: "
          f"{fairness_metrics['equal_opportunity_gap']:.4f}")

    print("\nPer-Group False Positive Rates:")
    for group, fpr in fairness_metrics['fpr_per_group'].items():
        print(f"  {group}: {fpr:.4f}")
    print(f"  -> Equalized Odds Gap: "
          f"{fairness_metrics['equalized_odds_gap']:.4f}")

    print("\nPer-Group F1 Scores:")
    for group, f1 in fairness_metrics['f1_per_group'].items():
        print(f"  {group}: {f1:.4f}")
    print(f"  -> Inter-group F1 Gap: {fairness_metrics['f1_gap']:.4f}")

    print("\n" + "=" * 60)
