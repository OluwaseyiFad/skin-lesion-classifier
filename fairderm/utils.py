# utility functions

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms


def mixup_data(x, y, alpha=0.4):
    # blend pairs of images for regularization
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    # blend the losses instead of labels
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def compute_group_losses(criterion, outputs, labels, group_indices,
                         groups=['Light', 'Medium', 'Dark']):
    # loss per skin tone group for adaptive sampler
    group_losses = {}

    for group_idx, group_name in enumerate(groups):
        mask = group_indices == group_idx
        if mask.sum() > 0:
            group_outputs = outputs[mask]
            group_labels = labels[mask]
            group_loss = criterion(group_outputs, group_labels).item()
            group_losses[group_name] = group_loss

    return group_losses


def get_train_transforms(resize_size=260, crop_size=224):
    # training transforms with augmentation
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop((crop_size, crop_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def get_eval_transforms(resize_size=260, crop_size=224):
    # eval transforms - no augmentation
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def get_ddi_eval_transforms(resize_size=260, crop_size=224):
    return get_eval_transforms(resize_size, crop_size)
