# adaptive sampler - adjusts sampling based on per-group loss
# this is the main contribution of the thesis

import torch
from torch.utils.data import WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt


class SkinToneAdaptiveSampler:
    # dynamically adjusts sampling probs based on loss per skin tone group
    # idea: groups the model struggles with (higher loss) get sampled more often
    # this helps the model learn better on underperforming groups

    def __init__(self, dataset, groups=['Light', 'Medium', 'Dark'],
                 epsilon=0.1, tau=1.0, ema_decay=0.9,
                 min_prob=0.10, size_weight_power=0.7, alpha=0.5):
        self.dataset = dataset
        self.groups = groups
        self.epsilon = epsilon  # smoothing to avoid div by zero
        self.tau = tau  # temperature for softmax-like scaling
        self.ema_decay = ema_decay  # how much to weight old loss vs new
        self.min_prob = min_prob  # floor so no group gets ignored completely
        self.alpha = alpha  # 0=pure size weighting, 1=pure loss weighting

        # figure out which samples belong to which group
        self.group_indices = {g: dataset.get_group_indices(g) for g in groups}
        self.group_sizes = {g: len(self.group_indices[g]) for g in groups}
        total_samples = sum(self.group_sizes.values())

        # inverse frequency weights - small groups get boosted
        # e.g. if Dark has 1/10th the samples, it gets ~10x weight
        self.size_weights = {}
        for g in groups:
            if self.group_sizes[g] > 0:
                inv_freq = total_samples / (len(groups) * self.group_sizes[g])
                self.size_weights[g] = inv_freq ** size_weight_power
            else:
                self.size_weights[g] = 1.0

        print(f"Adaptive Sampler initialized")
        print(f"  Group sizes: {self.group_sizes}")
        print(f"  Size weights: {self.size_weights}")
        print(f"  Min prob floor: {min_prob}")
        print(f"  Alpha (loss vs size): {alpha}")

        # don't init losses to arbitrary values - wait for real data
        self.running_losses = {g: None for g in groups}
        self._warmup_complete = False

        # track when each group was last seen (for staleness detection)
        self._update_counts = {g: 0 for g in groups}
        self._total_updates = 0

        # before we see any losses, use size weights for initial sampling
        total_size_weight = sum(self.size_weights.values())
        self.sampling_probs = {
            g: self.size_weights[g] / total_size_weight for g in groups
        }

        # keep history for plotting/debugging
        self.prob_history = {g: [] for g in groups}
        self.loss_history = {g: [] for g in groups}

    def update_losses(self, group_losses):
        # called after each batch with the loss for each group in that batch
        self._total_updates += 1

        # EMA update for groups that appeared in this batch
        for group, loss in group_losses.items():
            if group in self.groups:
                self._update_counts[group] = self._total_updates

                if self.running_losses[group] is None:
                    # first time seeing this group - just use the value directly
                    self.running_losses[group] = loss
                else:
                    # blend old and new: 90% old + 10% new (with ema_decay=0.9)
                    self.running_losses[group] = (
                        self.ema_decay * self.running_losses[group] +
                        (1 - self.ema_decay) * loss
                    )

        # check if we've seen all groups at least once
        if not self._warmup_complete:
            if all(self.running_losses[g] is not None for g in self.groups):
                self._warmup_complete = True
                print("  Adaptive sampler warmup complete - all groups observed")

        # handle stale groups - if a group hasn't been seen in a while,
        # slowly decay its loss toward the mean so it doesn't get stuck
        observed_losses = [l for l in self.running_losses.values() if l is not None]
        if observed_losses:
            mean_loss = sum(observed_losses) / len(observed_losses)

            for group in self.groups:
                if group not in group_losses and self.running_losses[group] is not None:
                    batches_since_seen = self._total_updates - self._update_counts[group]
                    # after 5 batches of not seeing a group, start decaying
                    if batches_since_seen > 5:
                        stale_decay = 0.05  # gentle decay toward mean
                        self.running_losses[group] = (
                            (1 - stale_decay) * self.running_losses[group] +
                            stale_decay * mean_loss
                        )

        # if a group still hasn't been seen, use mean of others as placeholder
        for group in self.groups:
            if self.running_losses[group] is None:
                self.running_losses[group] = mean_loss if observed_losses else 1.0

        # record for plotting
        for group in self.groups:
            self.loss_history[group].append(self.running_losses[group])

        self._update_probabilities()

    def _update_probabilities(self):
        # the key formula: combine loss signal and size signal
        # higher loss = sample more, smaller group = sample more
        losses = np.array([self.running_losses[g] for g in self.groups])
        sizes = np.array([self.size_weights[g] for g in self.groups])

        # temperature-scaled loss (higher loss -> higher prob)
        loss_scaled = (losses + self.epsilon) ** (1.0 / self.tau)

        # normalize each signal to sum to 1
        loss_normalized = loss_scaled / (loss_scaled.sum() + 1e-8)
        size_normalized = sizes / (sizes.sum() + 1e-8)

        # blend: alpha=1 means all loss, alpha=0 means all size
        combined = self.alpha * loss_normalized + (1 - self.alpha) * size_normalized

        base_probs = {g: combined[i] for i, g in enumerate(self.groups)}

        # enforce minimum floor - no group should drop below min_prob
        # this prevents any group from being completely ignored
        clipped = {g: max(self.min_prob, base_probs[g]) for g in self.groups}
        floor_total = self.min_prob * len(self.groups)

        # renormalize to sum to 1 while respecting floors
        if floor_total >= 1.0:
            # edge case: floors use up all probability mass
            norm_factor = sum(clipped.values())
            self.sampling_probs = {g: clipped[g] / norm_factor for g in self.groups}
        else:
            # normal case: redistribute excess probability
            excess = sum(clipped.values()) - 1.0
            if excess > 0:
                adjustable = {g: clipped[g] - self.min_prob for g in self.groups}
                adjustable_sum = sum(adjustable.values())
                for g in self.groups:
                    if adjustable_sum > 0:
                        scaled = adjustable[g] - (adjustable[g] / adjustable_sum) * excess
                        self.sampling_probs[g] = self.min_prob + max(scaled, 0.0)
                    else:
                        self.sampling_probs[g] = self.min_prob
            else:
                self.sampling_probs = clipped

            # final normalization to exactly 1
            norm_factor = sum(self.sampling_probs.values())
            self.sampling_probs = {g: self.sampling_probs[g] / norm_factor for g in self.groups}

        for group in self.groups:
            self.prob_history[group].append(self.sampling_probs[group])

    def get_sample_weights(self):
        # convert group probabilities to per-sample weights for WeightedRandomSampler
        # each sample in a group gets (group_prob / group_size) weight
        weights = torch.zeros(len(self.dataset))

        # handle Unknown group separately - give it small fixed prob
        unknown_indices = self.dataset.get_group_indices('Unknown')
        unknown_prob = 0.0
        if unknown_indices:
            unknown_prob = min(0.05, len(unknown_indices) / len(self.dataset))

        # distribute remaining probability among known groups
        known_total = 1.0 - unknown_prob
        present_groups = [g for g in self.groups if len(self.group_indices[g]) > 0]
        total_present_prob = sum(self.sampling_probs[g] for g in present_groups) if present_groups else 0.0

        if present_groups and total_present_prob > 0:
            for group in present_groups:
                indices = self.group_indices[group]
                # weight = (group's share of probability) / (number of samples in group)
                weights[indices] = (
                    (self.sampling_probs[group] / total_present_prob) * known_total
                ) / len(indices)
        else:
            # fallback to uniform if something went wrong
            if len(self.dataset) > 0:
                weights[:] = 1.0 / len(self.dataset)

        if unknown_indices:
            weights[unknown_indices] = unknown_prob / len(unknown_indices)

        return weights

    def create_sampler(self, num_samples=None):
        # convenience method to create a PyTorch sampler
        weights = self.get_sample_weights()
        num_samples = num_samples or len(self.dataset)
        return WeightedRandomSampler(weights, num_samples, replacement=True)

    def get_statistics(self):
        # for logging during training
        return {
            'running_losses': self.running_losses.copy(),
            'sampling_probs': self.sampling_probs.copy(),
            'size_weights': self.size_weights.copy(),
            'alpha': self.alpha,
            'warmup_complete': self._warmup_complete,
            'total_updates': self._total_updates
        }

    def get_state(self):
        # full state for checkpointing (so we can resume training)
        return {
            'running_losses': self.running_losses.copy(),
            'sampling_probs': self.sampling_probs.copy(),
            'prob_history': {g: list(h) for g, h in self.prob_history.items()},
            'loss_history': {g: list(h) for g, h in self.loss_history.items()},
            'alpha': self.alpha,
            '_warmup_complete': self._warmup_complete,
            '_update_counts': self._update_counts.copy(),
            '_total_updates': self._total_updates
        }

    def set_state(self, state):
        # restore from checkpoint
        self.running_losses = state['running_losses'].copy()
        self.sampling_probs = state['sampling_probs'].copy()
        self.prob_history = {g: list(h) for g, h in state['prob_history'].items()}
        self.loss_history = {g: list(h) for g, h in state['loss_history'].items()}
        if 'alpha' in state:
            self.alpha = state['alpha']
        if '_warmup_complete' in state:
            self._warmup_complete = state['_warmup_complete']
        if '_update_counts' in state:
            self._update_counts = state['_update_counts'].copy()
        if '_total_updates' in state:
            self._total_updates = state['_total_updates']

    def plot_history(self, save_path=None):
        # visualize how sampling evolved during training
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = {'Light': '#2ecc71', 'Medium': '#f39c12', 'Dark': '#9b59b6'}

        # left plot: sampling probabilities over time
        ax1 = axes[0]
        for group in self.groups:
            ax1.plot(self.prob_history[group], label=group,
                     color=colors[group], linewidth=2)
        ax1.axhline(y=1/3, color='gray', linestyle='--', alpha=0.5, label='Uniform')
        ax1.axhline(y=self.min_prob, color='red', linestyle=':',
                    alpha=0.5, label='Min floor')
        ax1.set_xlabel('Update')
        ax1.set_ylabel('Sampling Probability')
        ax1.set_title('Adaptive Sampling Probabilities')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # right plot: running losses over time
        ax2 = axes[1]
        for group in self.groups:
            ax2.plot(self.loss_history[group], label=group,
                     color=colors[group], linewidth=2)
        ax2.set_xlabel('Update')
        ax2.set_ylabel('Running Loss')
        ax2.set_title('Per-Group Losses')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()
