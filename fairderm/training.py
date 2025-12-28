# training loop with early stopping, mixup, and adaptive sampling

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from .utils import mixup_data, mixup_criterion, compute_group_losses


class EarlyStopping:
    # stops training when val loss stops improving for `patience` epochs
    # keeps a copy of the best model weights and restores them at the end

    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode  # 'min' = lower is better (for loss)
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        self.best_sampler_state = None  # also checkpoint sampler if using adaptive

    def __call__(self, score, model, sampler=None):
        # flip sign if minimizing so we can always check for "higher is better"
        if self.mode == 'min':
            score = -score

        if self.best_score is None:
            # first epoch - save as best
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            if sampler is not None:
                self.best_sampler_state = sampler.get_state()
        elif score < self.best_score + self.min_delta:
            # no improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # improvement! save new best
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            if sampler is not None:
                self.best_sampler_state = sampler.get_state()
            self.counter = 0

    def load_best_model(self, model, sampler=None):
        model.load_state_dict(self.best_model_state)
        if sampler is not None and self.best_sampler_state is not None:
            sampler.set_state(self.best_sampler_state)


def train_epoch(model, train_loader, criterion, optimizer, device,
                use_mixup=False, mixup_alpha=0.4, adaptive_sampler=None,
                train_dataset=None, batch_size=None):
    # single epoch of training
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # figure out how many steps this epoch
    if adaptive_sampler is not None:
        # with adaptive sampling, we generate batches on-the-fly
        if train_dataset is None or batch_size is None:
            raise ValueError("train_dataset and batch_size are required when using adaptive_sampler.")
        steps_per_epoch = math.ceil(len(train_dataset) / batch_size)
    else:
        steps_per_epoch = len(train_loader)

    # create data iterator
    if adaptive_sampler is not None:
        # generate batches dynamically using current sampling weights
        # this lets the sampler adapt between batches within an epoch
        def batch_generator():
            for _ in range(steps_per_epoch):
                weights = adaptive_sampler.get_sample_weights()
                indices = torch.multinomial(weights, batch_size, replacement=True).tolist()
                batch_data = [train_dataset[i] for i in indices]
                yield default_collate(batch_data)
        data_iterator = batch_generator()
    else:
        data_iterator = iter(train_loader)

    pbar = tqdm(range(steps_per_epoch), desc='Training')

    for batch_idx in pbar:
        images, labels, group_indices = next(data_iterator)

        images = images.to(device)
        labels = labels.to(device)
        group_indices = group_indices.to(device)

        optimizer.zero_grad()

        if use_mixup:
            # mixup: blend pairs of images and compute blended loss
            mixed_images, labels_a, labels_b, lam = mixup_data(
                images, labels, mixup_alpha
            )
            outputs = model(mixed_images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)

            # accuracy is approximate - weight by mixup lambda
            _, predicted = outputs.max(1)
            correct += (lam * predicted.eq(labels_a).sum().item() +
                        (1 - lam) * predicted.eq(labels_b).sum().item())
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

        total += labels.size(0)

        # update adaptive sampler with per-group losses
        if adaptive_sampler is not None:
            if use_mixup:
                # tricky: mixup blends images, but we need clean loss per group
                # so we do an extra forward pass on unmixed images
                # freeze batchnorm to avoid messing up running stats
                was_training = model.training
                model.train()
                bn_modules = []
                for m in model.modules():
                    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                        bn_modules.append(m)
                        m.eval()  # freeze running stats
                with torch.no_grad():
                    clean_outputs = model(images)
                # restore batchnorm to training mode
                for m in bn_modules:
                    m.train()
                if not was_training:
                    model.eval()
            else:
                # no mixup - just reuse the outputs we already computed
                clean_outputs = outputs.detach()

            group_losses = compute_group_losses(
                criterion, clean_outputs, labels, group_indices
            )
            if group_losses:
                adaptive_sampler.update_losses(group_losses)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        pbar.set_postfix({
            'loss': f'{running_loss / (batch_idx + 1):.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    return running_loss / steps_per_epoch, correct / total


def validate_epoch(model, val_loader, criterion, device):
    # run through validation set and collect predictions
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # collect everything for computing metrics later
    all_preds = []
    all_labels = []
    all_groups = []
    all_probs = []

    with torch.no_grad():
        for images, labels, group_indices in tqdm(val_loader, desc='Validating'):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_groups.extend(group_indices.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (
        running_loss / len(val_loader),
        correct / total,
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_groups),
        np.array(all_probs)
    )


def train_model(model, train_dataset, val_dataset, config, adaptive_sampler=None,
                device=None):
    # full training loop with early stopping and lr scheduling
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # val loader is the same every epoch
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 2),
        pin_memory=True
    )

    # using AdamW - like Adam but with decoupled weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.01)
    )

    # cosine annealing: lr starts high, decays smoothly to near zero
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=config['learning_rate'] * 0.01
    )

    # make sure criterion is on the right device
    criterion = config['criterion']
    if hasattr(criterion, 'weight') and criterion.weight is not None:
        criterion.weight = criterion.weight.to(device)
    if isinstance(criterion, nn.Module):
        criterion = criterion.to(device)

    early_stopping = EarlyStopping(patience=config.get('patience', 10))

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'learning_rates': []
    }

    print(f"\nStarting training for {config['epochs']} epochs...")
    print(f"Mixup: {'Yes' if config.get('use_mixup', False) else 'No'}")
    print(f"Adaptive Sampling: {'Yes' if adaptive_sampler else 'No'}")
    print("-" * 60)

    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")

        # with adaptive sampling, batches are generated on-the-fly in train_epoch
        # so we don't need a DataLoader here
        if adaptive_sampler is not None:
            train_loader = None
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=config['batch_size'],
                shuffle=True,
                num_workers=config.get('num_workers', 2),
                pin_memory=True
            )

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            use_mixup=config.get('use_mixup', False),
            mixup_alpha=config.get('mixup_alpha', 0.4),
            adaptive_sampler=adaptive_sampler,
            train_dataset=train_dataset,
            batch_size=config['batch_size']
        )

        val_loss, val_acc, _, _, _, _ = validate_epoch(
            model, val_loader, criterion, device
        )

        scheduler.step()

        # record everything
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # show current sampling probabilities if using adaptive
        if adaptive_sampler:
            stats = adaptive_sampler.get_statistics()
            probs = stats['sampling_probs']
            print(f"Sampling: Light={probs['Light']:.2f}, "
                  f"Medium={probs['Medium']:.2f}, Dark={probs['Dark']:.2f}")

        # check for early stopping
        early_stopping(val_loss, model, sampler=adaptive_sampler)
        if early_stopping.early_stop:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # restore best model from training
    early_stopping.load_best_model(model, sampler=adaptive_sampler)
    print("\nLoaded best model checkpoint")

    return history, model
