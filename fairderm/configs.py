# config for each experiment method

import torch.nn as nn
from .losses import FocalLoss


def get_baseline_config():
    # vanilla cross-entropy, no fairness tricks
    return {
        'name': 'Baseline',
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'patience': 10,
        'num_workers': 2,
        'criterion': nn.CrossEntropyLoss(),
        'use_mixup': False,
        'use_adaptive': False
    }


def get_mixup_config():
    # just mixup, no adaptive sampling
    config = get_baseline_config()
    config.update({
        'name': 'Mixup',
        'use_mixup': True,
        'mixup_alpha': 0.4
    })
    return config


def get_reweighted_config(class_weights, device=None):
    # weight minority classes more in the loss
    config = get_baseline_config()

    if device is not None:
        class_weights = class_weights.to(device)

    config.update({
        'name': 'Reweighted',
        'criterion': nn.CrossEntropyLoss(weight=class_weights)
    })
    return config


def get_focal_config():
    # focal loss - focuses on hard examples
    config = get_baseline_config()
    config.update({
        'name': 'FocalLoss',
        'criterion': FocalLoss(gamma=2.0)
    })
    return config


def get_proposed_config():
    # my method: adaptive sampling + mixup
    config = get_baseline_config()
    config.update({
        'name': 'Proposed',
        'use_mixup': True,
        'mixup_alpha': 0.4,
        'use_adaptive': True,
        'min_prob': 0.10,
        'size_weight_power': 0.7,
        'alpha': 0.5  # 0=pure size, 1=pure loss
    })
    return config


def get_groupdro_config():
    # GroupDRO minimax objective over skin-tone groups
    config = get_baseline_config()
    config.update({
        'name': 'GroupDRO',
        'use_groupdro': True,
        'groupdro_eta': 0.01,
        'groupdro_eta_sweep': [0.001, 0.005, 0.01, 0.05],
        'groupdro_groups': ['Light', 'Medium', 'Dark'],
        'groupdro_stratified_batches': True,
        'groupdro_uncovered_group_policy': 'erm'
    })
    return config
