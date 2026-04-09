# config for each experiment method

import torch.nn as nn
from .losses import FocalLoss


def get_baseline_config():
    # vanilla cross-entropy
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
    config = get_baseline_config()
    config.update({
        'name': 'Mixup',
        'use_mixup': True,
        'mixup_alpha': 0.4
    })
    return config


def get_reweighted_config(class_weights, device=None):
    config = get_baseline_config()

    if device is not None:
        class_weights = class_weights.to(device)

    config.update({
        'name': 'Reweighted',
        'criterion': nn.CrossEntropyLoss(weight=class_weights)
    })
    return config


def get_focal_config():
    config = get_baseline_config()
    config.update({
        'name': 'FocalLoss',
        'criterion': FocalLoss(gamma=2.0)
    })
    return config


def get_ldas_config():
    # loss-driven adaptive sampling: alpha=0.7, tau=2.0
    config = get_baseline_config()
    config.update({
        'name': 'LDAS',
        'use_mixup': False,
        'use_adaptive': True,
        'alpha': 0.7,
        'tau': 2.0,
        'ema_decay': 0.9,
        'min_prob': 0.10,
        'epsilon': 0.1,
        'size_weight_power': 0.7,
    })
    return config


def get_groupdro_config():
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
