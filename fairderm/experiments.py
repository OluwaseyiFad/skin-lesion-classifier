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


def _set_random_seed(seed):
    # set all random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _worst_tpr(fairness):
    tpr_per_group = fairness.get('tpr_per_group', {})
    if not tpr_per_group:
        return float('-inf')
    return float(min(tpr_per_group.values()))


def _run_single_experiment(train_dataset, val_dataset, config, seed=42, device=None):
    # train and evaluate a single model with a specific config and seed
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


def run_experiment(train_dataset, val_dataset, config, seed=42, device=None):
    # train and evaluate a single model with a specific config and seed
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    eta_sweep = config.get('groupdro_eta_sweep')
    use_eta_sweep = bool(config.get('use_groupdro', False) and eta_sweep)

    if not use_eta_sweep:
        _set_random_seed(seed)
        return _run_single_experiment(train_dataset, val_dataset, config, seed=seed, device=device)

    eta_values = [float(eta) for eta in eta_sweep]
    if len(eta_values) != 4:
        raise ValueError(
            f"GroupDRO eta sweep expects exactly 4 values, got {len(eta_values)}: {eta_values}"
        )

    print(f"\nRunning GroupDRO eta sweep with candidates: {eta_values}")
    sweep_runs = []
    sweep_results = {}

    for eta in eta_values:
        eta_config = copy.deepcopy(config)
        eta_config.pop('groupdro_eta_sweep', None)
        eta_config['groupdro_eta'] = eta
        eta_config['name'] = f"{config['name']}_eta{eta:g}"

        _set_random_seed(seed)
        eta_result = _run_single_experiment(
            train_dataset, val_dataset, eta_config, seed=seed, device=device
        )
        eta_worst_tpr = _worst_tpr(eta_result['fairness'])
        eta_bal_acc = float(eta_result['overall']['balanced_accuracy'])

        sweep_results[eta] = eta_result
        sweep_runs.append({
            'eta': eta,
            'worst_tpr': eta_worst_tpr,
            'balanced_accuracy': eta_bal_acc
        })

    # Selection rule: maximize worstTPR first, balanced accuracy second.
    best_run = max(sweep_runs, key=lambda r: (r['worst_tpr'], r['balanced_accuracy']))
    best_eta = best_run['eta']
    best_result = sweep_results[best_eta]

    selected_config = copy.deepcopy(best_result['config'])
    selected_config['name'] = config['name']
    selected_config['groupdro_eta'] = best_eta
    selected_config['groupdro_eta_sweep'] = eta_values
    best_result['config'] = selected_config
    best_result['eta_sweep'] = {
        'runs': sweep_runs,
        'selected_eta': best_eta,
        'selection_rule': 'max(worst_tpr), tie-break by max(balanced_accuracy)'
    }

    print("\nGroupDRO eta sweep summary:")
    for run in sweep_runs:
        print(
            f"  eta={run['eta']:.6f} | worstTPR={run['worst_tpr']:.4f} | "
            f"bal_acc={run['balanced_accuracy']:.4f}"
        )
    print(f"Selected eta: {best_eta:.6f}")

    return best_result


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
                'groupdro_eta': results['config'].get('groupdro_eta', None),
                'accuracy': results['overall']['accuracy'],
                'balanced_accuracy': results['overall']['balanced_accuracy'],
                'roc_auc': results['overall'].get('roc_auc', None),
                'f1_macro': results['overall']['f1_macro'],
                'equal_opportunity_gap': results['fairness']['equal_opportunity_gap'],
                'equalized_odds_gap': results['fairness']['equalized_odds_gap'],
                'f1_gap': results['fairness']['f1_gap'],
                'worst_tpr': _worst_tpr(results['fairness']),
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
                if 'eta_sweep' in results:
                    sweep_df = pd.DataFrame(results['eta_sweep']['runs'])
                    sweep_df.to_csv(
                        os.path.join(save_dir, run_name, 'eta_sweep.csv'),
                        index=False
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
