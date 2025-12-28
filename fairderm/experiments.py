# running experiments and saving results
# handles training multiple configs with multiple seeds

import os
import copy
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .models import create_model
from .samplers import SkinToneAdaptiveSampler
from .training import train_model, validate_epoch
from .evaluation import (
    compute_metrics, compute_group_metrics,
    compute_fairness_metrics, print_fairness_report
)
from .visualization import plot_training_history


def run_experiment(train_dataset, val_dataset, config, seed=42, device=None):
    # train and evaluate a single model with a specific config and seed

    # set all random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'=' * 60}")
    print(f"Experiment: {config['name']} (seed={seed})")
    print(f"{'=' * 60}")

    model = create_model(num_classes=2, pretrained=True, device=device)

    # set up adaptive sampler if this config uses it
    adaptive_sampler = None
    if config.get('use_adaptive', False):
        adaptive_sampler = SkinToneAdaptiveSampler(
            train_dataset,
            groups=['Light', 'Medium', 'Dark'],
            epsilon=0.1,
            tau=1.0,
            ema_decay=0.9,
            min_prob=config.get('min_prob', 0.10),
            size_weight_power=config.get('size_weight_power', 0.7),
            alpha=config.get('alpha', 0.5)
        )

    history, model = train_model(
        model, train_dataset, val_dataset, config,
        adaptive_sampler=adaptive_sampler,
        device=device
    )

    # final evaluation on validation set
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )
    # deepcopy criterion in case it has state we don't want to modify
    val_criterion = copy.deepcopy(config['criterion'])
    if isinstance(val_criterion, nn.Module):
        val_criterion = val_criterion.to(device)

    val_loss, val_acc, preds, labels, groups, probs = validate_epoch(
        model, val_loader, val_criterion, device
    )

    # compute all the metrics
    overall = compute_metrics(labels, preds, probs)
    per_group = compute_group_metrics(labels, preds, groups, probs)
    fairness = compute_fairness_metrics(labels, preds, groups)

    results = {
        'config': config,
        'history': history,
        'overall': overall,
        'per_group': per_group,
        'fairness': fairness,
        'model_state': model.state_dict(),
        'adaptive_sampler': adaptive_sampler
    }

    # print summary
    print(f"\n{config['name']} Results:")
    print(f"  Accuracy: {overall['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {overall['balanced_accuracy']:.4f}")
    if 'roc_auc' in overall:
        print(f"  ROC-AUC: {overall['roc_auc']:.4f}")
    print_fairness_report(fairness)

    return results


def run_all_experiments(train_dataset, val_dataset, train_df, configs, seeds=[42],
                        save_dir=None, device=None):
    # run all configs with all seeds and collect results
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_results = {}
    comparison_data = []

    for config in configs:
        for seed in seeds:
            run_name = f"{config['name']}_seed{seed}"

            results = run_experiment(
                train_dataset, val_dataset, config, seed=seed, device=device
            )

            all_results[run_name] = results

            # collect metrics for comparison table
            comparison_data.append({
                'name': config['name'],
                'seed': seed,
                'accuracy': results['overall']['accuracy'],
                'balanced_accuracy': results['overall']['balanced_accuracy'],
                'roc_auc': results['overall'].get('roc_auc', None),
                'f1_macro': results['overall']['f1_macro'],
                'equal_opportunity_gap': results['fairness']['equal_opportunity_gap'],
                'equalized_odds_gap': results['fairness']['equalized_odds_gap'],
                'f1_gap': results['fairness']['f1_gap'],
                'tpr_light': results['fairness']['tpr_per_group'].get('Light', None),
                'tpr_medium': results['fairness']['tpr_per_group'].get('Medium', None),
                'tpr_dark': results['fairness']['tpr_per_group'].get('Dark', None),
            })

            # save model checkpoint
            if save_dir:
                os.makedirs(os.path.join(save_dir, run_name), exist_ok=True)
                torch.save(
                    results['model_state'],
                    os.path.join(save_dir, run_name, 'model.pt')
                )

            # save training curves
            plot_training_history(
                results['history'],
                title=f'{run_name} Training History',
                save_path=os.path.join(save_dir, run_name, 'training_history.png')
                if save_dir else None
            )

            # free up GPU memory between runs
            del results['model_state']
            torch.cuda.empty_cache()

    comparison_df = pd.DataFrame(comparison_data)

    if save_dir:
        comparison_df.to_csv(
            os.path.join(save_dir, 'experiment_comparison.csv'),
            index=False
        )

    return all_results, comparison_df


def save_experiment_results(results, run_name, save_dir):
    # save everything to disk for later analysis
    run_dir = os.path.join(save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # model weights
    torch.save(results['model_state'], os.path.join(run_dir, 'model.pt'))

    # training history as csv
    history_df = pd.DataFrame(results['history'])
    history_df.to_csv(os.path.join(run_dir, 'training_history.csv'), index=False)

    # metrics as json (excluding confusion matrix which isn't json-serializable)
    metrics = {
        'overall': results['overall'],
        'fairness': results['fairness']
    }
    if 'confusion_matrix' in metrics['overall']:
        del metrics['overall']['confusion_matrix']

    import json
    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"Results saved to {run_dir}")
