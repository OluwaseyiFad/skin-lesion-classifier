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


class GroupDROObjective:
    def __init__(self, groups=('Light', 'Medium', 'Dark'), eta=0.01,
                 group_name_to_idx=None, uncovered_group_policy='erm'):
        self.groups = list(groups)
        self.eta = eta
        if self.eta <= 0:
            raise ValueError(f"groupdro eta must be > 0, got {self.eta}")
        self.q = None
        self.group_name_to_idx = (
            dict(group_name_to_idx) if group_name_to_idx is not None else None
        )
        self.uncovered_group_policy = uncovered_group_policy
        if self.uncovered_group_policy not in {'erm', 'ignore', 'error'}:
            raise ValueError(
                "uncovered_group_policy must be one of {'erm', 'ignore', 'error'}"
            )

        if self.group_name_to_idx is None:
            self._target_group_indices = list(range(len(self.groups)))
        else:
            missing = [g for g in self.groups if g not in self.group_name_to_idx]
            if missing:
                raise ValueError(
                    f"Configured GroupDRO groups missing from group_name_to_idx: {missing}"
                )
            self._target_group_indices = [self.group_name_to_idx[g] for g in self.groups]

    def to(self, device):
        self.q = torch.ones(len(self.groups), device=device) / len(self.groups)
        return self

    def compute_loss(self, per_sample_losses, group_indices):
        if self.q is None:
            self.to(per_sample_losses.device)

        group_losses = torch.zeros(len(self.groups), device=per_sample_losses.device)
        observed_mask = torch.zeros(len(self.groups), dtype=torch.bool, device=per_sample_losses.device)
        covered_mask = torch.zeros_like(group_indices, dtype=torch.bool)
        group_loss_values = {}

        for objective_idx, (group_name, dataset_group_idx) in enumerate(
            zip(self.groups, self._target_group_indices)
        ):
            mask = group_indices == dataset_group_idx
            if mask.any():
                group_loss = per_sample_losses[mask].mean()
                group_losses[objective_idx] = group_loss
                observed_mask[objective_idx] = True
                group_loss_values[group_name] = group_loss.detach().item()
            covered_mask = covered_mask | mask

        uncovered_mask = ~covered_mask
        if uncovered_mask.any():
            uncovered_loss = per_sample_losses[uncovered_mask].mean()
            group_loss_values['Uncovered'] = uncovered_loss.detach().item()
        else:
            uncovered_loss = None

        if not observed_mask.any():
            if self.uncovered_group_policy == 'error':
                raise ValueError(
                    "GroupDRO batch contained no configured groups. "
                    "Check `groupdro_groups` and dataset group mapping."
                )
            return per_sample_losses.mean(), group_loss_values

        with torch.no_grad():
            scaled_losses = torch.clamp(
                self.eta * group_losses[observed_mask].detach(),
                max=50.0
            )
            self.q[observed_mask] = (
                self.q[observed_mask] * torch.exp(scaled_losses)
            )
            self.q = self.q / self.q.sum().clamp_min(1e-12)

        dro_loss = torch.dot(self.q, group_losses)

        if uncovered_loss is not None:
            if self.uncovered_group_policy == 'erm':
                uncovered_frac = uncovered_mask.float().mean()
                dro_loss = (1.0 - uncovered_frac) * dro_loss + uncovered_frac * uncovered_loss
            elif self.uncovered_group_policy == 'error':
                raise ValueError(
                    "Encountered samples from uncovered groups while "
                    "uncovered_group_policy='error'."
                )
        return dro_loss, group_loss_values

    def get_weights(self):
        if self.q is None:
            return {g: 1.0 / len(self.groups) for g in self.groups}
        return {
            group: float(self.q[idx].detach().cpu().item())
            for idx, group in enumerate(self.groups)
        }


def _compute_per_sample_losses(criterion, outputs, labels):
    if not hasattr(criterion, 'reduction'):
        raise ValueError(
            "GroupDRO requires a loss with a `reduction` attribute "
            "(e.g., nn.CrossEntropyLoss or FocalLoss)."
        )

    original_reduction = criterion.reduction
    try:
        criterion.reduction = 'none'
        losses = criterion(outputs, labels)
    finally:
        criterion.reduction = original_reduction

    if losses.ndim > 1:
        losses = losses.mean(dim=tuple(range(1, losses.ndim)))
    return losses


def _resolve_group_sample_indices(dataset, group_names):
    if hasattr(dataset, 'get_group_indices'):
        return {g: list(dataset.get_group_indices(g)) for g in group_names}

    if hasattr(dataset, 'group_indices') and hasattr(dataset, 'group_to_idx'):
        raw_group_indices = np.asarray(dataset.group_indices)
        resolved = {}
        for group_name in group_names:
            if group_name not in dataset.group_to_idx:
                resolved[group_name] = []
                continue
            dataset_idx = dataset.group_to_idx[group_name]
            resolved[group_name] = np.where(raw_group_indices == dataset_idx)[0].tolist()
        return resolved

    raise ValueError(
        "GroupDRO stratified batches require dataset.get_group_indices(...) "
        "or dataset.group_indices + dataset.group_to_idx."
    )


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        self.best_sampler_state = None

    def __call__(self, score, model, sampler=None):
        if self.mode == 'min':
            score = -score

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            if sampler is not None:
                self.best_sampler_state = sampler.get_state()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
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
                group_dro_objective=None, train_dataset=None, batch_size=None,
                groupdro_stratified_batches=False, groupdro_groups=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    if adaptive_sampler is not None or groupdro_stratified_batches:
        if train_dataset is None or batch_size is None:
            raise ValueError(
                "train_dataset and batch_size are required when using "
                "adaptive sampler or GroupDRO stratified batches."
            )
        steps_per_epoch = math.ceil(len(train_dataset) / batch_size)
    else:
        steps_per_epoch = len(train_loader)

    if adaptive_sampler is not None:
        def batch_generator():
            for _ in range(steps_per_epoch):
                weights = adaptive_sampler.get_sample_weights()
                indices = torch.multinomial(weights, batch_size, replacement=True).tolist()
                batch_data = [train_dataset[i] for i in indices]
                yield default_collate(batch_data)
        data_iterator = batch_generator()
    elif groupdro_stratified_batches:
        if group_dro_objective is None:
            raise ValueError(
                "groupdro_stratified_batches=True requires group_dro_objective."
            )
        target_groups = list(groupdro_groups or group_dro_objective.groups)
        group_to_indices = _resolve_group_sample_indices(train_dataset, target_groups)
        present_groups = [g for g in target_groups if len(group_to_indices.get(g, [])) > 0]
        if not present_groups:
            raise ValueError(
                "No samples found for configured GroupDRO groups in stratified mode."
            )
        if batch_size < len(present_groups):
            raise ValueError(
                f"batch_size={batch_size} is smaller than number of present "
                f"GroupDRO groups={len(present_groups)}; cannot guarantee one sample "
                "per group per batch."
            )

        covered_pool = []
        for group_name in present_groups:
            covered_pool.extend(group_to_indices[group_name])

        def batch_generator():
            for _ in range(steps_per_epoch):
                batch_indices = []

                # guarantee at least one sample from each group per batch
                for group_name in present_groups:
                    indices = group_to_indices[group_name]
                    sampled_pos = torch.randint(0, len(indices), (1,)).item()
                    batch_indices.append(indices[sampled_pos])

                remaining = batch_size - len(batch_indices)
                if remaining > 0:
                    sampled_positions = torch.randint(0, len(covered_pool), (remaining,)).tolist()
                    batch_indices.extend([covered_pool[pos] for pos in sampled_positions])

                perm = torch.randperm(len(batch_indices)).tolist()
                batch_data = [train_dataset[batch_indices[i]] for i in perm]
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
        if group_dro_objective is not None:
            outputs = model(images)
            per_sample_losses = _compute_per_sample_losses(criterion, outputs, labels)
            loss, _ = group_dro_objective.compute_loss(per_sample_losses, group_indices)

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
        elif use_mixup:
            mixed_images, labels_a, labels_b, lam = mixup_data(
                images, labels, mixup_alpha
            )
            outputs = model(mixed_images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)

            _, predicted = outputs.max(1)
            correct += (lam * predicted.eq(labels_a).sum().item() +
                        (1 - lam) * predicted.eq(labels_b).sum().item())
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

        total += labels.size(0)

        if adaptive_sampler is not None:
            if use_mixup:
                # need clean (unmixed) outputs for per-group loss tracking
                was_training = model.training
                model.train()
                bn_modules = []
                for m in model.modules():
                    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                        bn_modules.append(m)
                        m.eval()
                with torch.no_grad():
                    clean_outputs = model(images)
                for m in bn_modules:
                    m.train()
                if not was_training:
                    model.eval()
            else:
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


def validate_epoch(model, val_loader, criterion, device,
                   return_group_losses=False, groups=None, group_name_to_idx=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_groups = []
    all_probs = []
    all_per_sample_losses = [] if return_group_losses else None

    with torch.no_grad():
        for images, labels, group_indices in tqdm(val_loader, desc='Validating'):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            if return_group_losses:
                per_sample = _compute_per_sample_losses(criterion, outputs, labels)
                all_per_sample_losses.extend(per_sample.cpu().numpy())
                loss = per_sample.mean()
            else:
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

    base = (
        running_loss / len(val_loader),
        correct / total,
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_groups),
        np.array(all_probs)
    )

    if not return_group_losses:
        return base

    losses_arr = np.array(all_per_sample_losses)
    groups_arr = np.array(all_groups)
    group_losses = {}
    if groups:
        for fallback_idx, g_name in enumerate(groups):
            if group_name_to_idx is not None:
                if g_name not in group_name_to_idx:
                    continue
                g_idx = group_name_to_idx[g_name]
            else:
                g_idx = fallback_idx
            mask = groups_arr == g_idx
            if mask.any():
                group_losses[g_name] = float(losses_arr[mask].mean())
    return base + (group_losses,)


def train_model(model, train_dataset, val_dataset, config, adaptive_sampler=None,
                device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 2),
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.01)
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=config['learning_rate'] * 0.01
    )

    criterion = config['criterion']
    if hasattr(criterion, 'weight') and criterion.weight is not None:
        criterion.weight = criterion.weight.to(device)
    if isinstance(criterion, nn.Module):
        criterion = criterion.to(device)

    use_groupdro = config.get('use_groupdro', False)
    if use_groupdro and (config.get('use_mixup', False) or adaptive_sampler is not None):
        raise ValueError("GroupDRO is configured as a standalone objective. Disable mixup/adaptive sampling.")
    use_groupdro_stratified = bool(use_groupdro and config.get('groupdro_stratified_batches', True))

    groupdro_groups = tuple(config.get('groupdro_groups', ['Light', 'Medium', 'Dark']))
    if hasattr(train_dataset, 'group_to_idx'):
        group_name_to_idx = dict(train_dataset.group_to_idx)
    else:
        group_name_to_idx = {g: idx for idx, g in enumerate(groupdro_groups)}

    missing_groups = [g for g in groupdro_groups if g not in group_name_to_idx]
    if missing_groups:
        raise ValueError(
            f"GroupDRO groups not found in dataset mapping: {missing_groups}. "
            "Update `groupdro_groups` to match dataset group names."
        )

    group_dro_objective = None
    if use_groupdro:
        group_dro_objective = GroupDROObjective(
            groups=groupdro_groups,
            eta=config.get('groupdro_eta', 0.01),
            group_name_to_idx=group_name_to_idx,
            uncovered_group_policy=config.get('groupdro_uncovered_group_policy', 'erm')
        ).to(device)

    early_stopping = EarlyStopping(patience=config.get('patience', 10))

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_worst_group_loss': [],
        'learning_rates': []
    }

    print(f"\nStarting training for {config['epochs']} epochs...")
    print(f"Mixup: {'Yes' if config.get('use_mixup', False) else 'No'}")
    print(f"Adaptive Sampling: {'Yes' if adaptive_sampler else 'No'}")
    print(f"GroupDRO: {'Yes' if use_groupdro else 'No'}")
    if use_groupdro:
        print(f"GroupDRO Stratified Batches: {'Yes' if use_groupdro_stratified else 'No'}")
    print("-" * 60)

    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")

        if adaptive_sampler is not None or use_groupdro_stratified:
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
            group_dro_objective=group_dro_objective,
            train_dataset=train_dataset,
            batch_size=config['batch_size'],
            groupdro_stratified_batches=use_groupdro_stratified,
            groupdro_groups=groupdro_groups
        )

        if use_groupdro:
            val_loss, val_acc, _, _, _, _, val_group_losses = validate_epoch(
                model,
                val_loader,
                criterion,
                device,
                return_group_losses=True,
                groups=groupdro_groups,
                group_name_to_idx=group_name_to_idx
            )
            if val_group_losses:
                val_worst_group_name, val_worst_group_loss = max(
                    val_group_losses.items(), key=lambda kv: kv[1]
                )
            else:
                val_worst_group_loss = None
                val_worst_group_name = None
        else:
            val_loss, val_acc, _, _, _, _ = validate_epoch(
                model, val_loader, criterion, device
            )
            val_worst_group_loss = None
            val_worst_group_name = None

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_worst_group_loss'].append(
            val_worst_group_loss if val_worst_group_loss is not None else np.nan
        )
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        if val_worst_group_loss is not None:
            print(f"Val Worst-Group Loss: {val_worst_group_loss:.4f} ({val_worst_group_name})")

        if adaptive_sampler:
            stats = adaptive_sampler.get_statistics()
            probs = stats['sampling_probs']
            print(f"Sampling: Light={probs['Light']:.2f}, "
                  f"Medium={probs['Medium']:.2f}, Dark={probs['Dark']:.2f}")
        if group_dro_objective is not None:
            q = group_dro_objective.get_weights()
            q_str = ", ".join([f"{g}={w:.2f}" for g, w in q.items()])
            print(f"GroupDRO q: {q_str}")

        early_stopping_score = val_worst_group_loss if val_worst_group_loss is not None else val_loss
        early_stopping(early_stopping_score, model, sampler=adaptive_sampler)
        if early_stopping.early_stop:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    early_stopping.load_best_model(model, sampler=adaptive_sampler)
    print("\nLoaded best model checkpoint")

    return history, model
