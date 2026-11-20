"""Experiment presets.

``manuscript`` presets hold settings stated in the manuscript Methods. Fields
marked ASSUMPTION are not stated in the manuscript; their values and the
evidence behind them are listed in docs/IMPLEMENTATION_NOTES.md.

``demo`` presets are small CPU settings for the synthetic suite only. They are
not manuscript settings and must not be used to report results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import numpy as np


# ---------------------------------------------------------------------------
# Atlas transformation
# ---------------------------------------------------------------------------
TRANSFORM_RANK = {"manuscript": 50, "demo": 10}  # Methods: top k = 50


# ---------------------------------------------------------------------------
# FLECHA
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FlechaPreset:
    name: str
    hidden: tuple
    norm: str
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    rounds: int
    local_steps: int
    epochs_min: int = 5            # Methods: Single-site constrained to 5-80 epochs
    epochs_max: int = 80
    n_seeds: int = 10              # Methods: 10 random site partitions
    client_init: str = "shared"    # ASSUMPTION: standard FedAvg (legacy code: "independent")
    client_optimizer: str = "persistent"  # ASSUMPTION (legacy behavior); or "reset" each round
    target_scaling: str = "none"   # ASSUMPTION (legacy behavior); or "zscore" on training labels
    description: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        return d


FLECHA_PRESETS = {
    # Methods, "Federated learning with heterogeneous atlas spaces", UKB paragraph.
    "ukb": FlechaPreset("ukb", (256, 128, 64), "batch", 0.2, 1e-3, 1e-4, 64, 100, 10,
                        description="Manuscript UKB FC/SC age and fluid intelligence"),
    # Same section, HCP-YA / HCP-D paragraph.
    "hcp": FlechaPreset("hcp", (128, 64), "layer", 0.2, 1e-3, 1e-4, 32, 200, 5,
                        description="Manuscript HCP-YA FC and HCP-D SC age"),
    "demo": FlechaPreset("demo", (32, 16), "batch", 0.2, 1e-3, 1e-4, 16, 15, 3, n_seeds=3,
                         description="Synthetic smoke test only (not a manuscript setting)"),
}


# ---------------------------------------------------------------------------
# MECHA
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MechaPreset:
    name: str
    # Meta-training DNN (Methods, "Meta-matching with heterogeneous atlas spaces")
    hidden: tuple
    norm: str
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int                # ASSUMPTION (legacy config EPOCHS = 300)
    patience: int                  # ASSUMPTION; 0 = no patience, restore best epoch
    # Adaptation
    k_list: tuple = (10, 20, 50, 100, 200)
    n_repeats: int = 10
    stack_alphas: tuple = (0.1, 0.3, 1, 3, 10, 30, 100)   # ASSUMPTION (legacy grid)
    # Support-only baselines (ASSUMPTION: settings not stated in the manuscript)
    baseline_ridge_alphas: tuple = tuple(float(a) for a in np.logspace(-1, 5, 7))
    baseline_dnn_hidden: tuple = (128, 64)
    baseline_dnn_dropout: float = 0.5
    baseline_dnn_lr: float = 1e-3
    baseline_dnn_weight_decay: float = 1e-4
    baseline_dnn_batch_size: int = 32
    baseline_dnn_epochs: int = 100
    description: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d


MECHA_PRESETS = {
    "ukb": MechaPreset("ukb", (128, 64), "batch", 0.5, 1e-4, 1e-4, 256, max_epochs=300, patience=50,
                       description="Manuscript UKB MECHA"),
    # Canonical synthetic MECHA example (atlas_interop demo --workflows mecha).
    # Synthetic-specific: same architecture family and stacking/baseline settings
    # as "ukb", but lr 1e-3 and fewer max epochs so the 10,000-participant
    # meta-training set trains on a CPU. Not a manuscript setting.
    "synthetic": MechaPreset("synthetic", (128, 64), "batch", 0.5, 1e-3, 1e-4, 256, max_epochs=150, patience=20,
                             k_list=(10, 20, 40, 100), n_repeats=10,
                             description="Canonical synthetic MECHA example (not a manuscript setting)"),
    # Tiny developer preflight used by the test suite only.
    "preflight": MechaPreset("preflight", (16,), "batch", 0.2, 1e-3, 1e-4, 16, max_epochs=4, patience=0,
                             k_list=(10, 20), n_repeats=2, baseline_dnn_hidden=(8,), baseline_dnn_epochs=3,
                             description="Developer preflight only"),
}

SYNTHETIC_MECHA_DRAW_SEEDS = tuple(range(1001, 1011))   # 10 predetermined support-draw seeds
SYNTHETIC_MECHA_TRAIN_SEED = 7                          # same DNN seed for both conditions


# ---------------------------------------------------------------------------
# Ridge prediction pipeline (Methods, "Evaluation of reconstruction similarity
# and predictive utility after connectome transformation")
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PredictionPreset:
    name: str
    tuning_fraction: float = 0.2
    p_threshold: float = 0.1
    alphas: tuple = tuple(float(a) for a in np.logspace(-1, 3, 9))  # range stated; grid density ASSUMPTION
    inner_folds: int = 3
    outer_folds: int = 10
    n_repeats: int = 10
    description: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        d["alphas"] = list(self.alphas)
        return d


PREDICTION_PRESETS = {
    "manuscript": PredictionPreset("manuscript", description="Manuscript ridge protocol"),
    "demo": PredictionPreset("demo", outer_folds=5, n_repeats=3,
                             description="Synthetic smoke test only (not a manuscript setting)"),
}


def get_preset(table: dict, name: str):
    if name not in table:
        raise ValueError(f"unknown preset '{name}'. Choose from: {sorted(table)}")
    return table[name]


def override(preset, **kwargs):
    """Return a copy of ``preset`` with non-None keyword overrides applied."""
    changes = {k: v for k, v in kwargs.items() if v is not None}
    return replace(preset, **changes) if changes else preset
