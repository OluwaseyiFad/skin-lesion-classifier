# fairderm - skin lesion classifier with fairness metrics

from .models import SkinLesionClassifier, create_model
from .datasets import SkinLesionDataset, DDIDataset
from .samplers import SkinToneAdaptiveSampler
from .losses import FocalLoss, get_class_weights
from .training import train_epoch, validate_epoch, train_model, EarlyStopping
from .evaluation import (
    compute_metrics,
    compute_group_metrics,
    compute_fairness_metrics,
    print_fairness_report,
    bootstrap_ci,
)
from .utils import (
    mixup_data, mixup_criterion, compute_group_losses,
    get_train_transforms, get_eval_transforms, get_ddi_eval_transforms
)
from .configs import (
    get_baseline_config,
    get_mixup_config,
    get_reweighted_config,
    get_focal_config,
    get_proposed_config,
    get_groupdro_config,
)
from .visualization import (
    plot_training_history, plot_fairness_comparison,
    plot_group_comparison, plot_confusion_matrices
)
from .experiments import run_experiment, run_all_experiments, save_experiment_results

__version__ = "1.0.0"
__author__ = "Oluwaseyi Fadahunsi"
