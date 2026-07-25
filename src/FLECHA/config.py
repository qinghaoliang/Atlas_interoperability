"""
config.py
=========

Configuration defaults for FLECHA experiments.

This peer-review version assumes the canonical repository data layout:

    data/raw/fc/<atlas>.npz
    data/raw/sc/<atlas>.npz
    data/mapped/fc/<source_atlas>_to_<target_atlas>.npz
    data/mapped/sc/<source_atlas>_to_<target_atlas>.npz

The actual controlled-access data are not included in the repository.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class FlechaConfig:
    """Default configuration for FLECHA experiments."""

    # ---------------------------------------------------------------------
    # Atlas lists used to define simulated clients/sites
    # ---------------------------------------------------------------------
    fc_atlases: List[str] = field(
        default_factory=lambda: [
            "Schaefer_100",
            "Schaefer_200",
            "Schaefer_300",
            "Schaefer_400",
            "Schaefer_500",
            "glasser",
        ]
    )

    sc_atlases: List[str] = field(
        default_factory=lambda: [
            "Schaefer_200",
            "Schaefer_500",
            "glasser",
        ]
    )

    # ---------------------------------------------------------------------
    # Training hyperparameters
    # ---------------------------------------------------------------------
    batch_size: int = 64
    rounds: int = 100
    local_steps: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    epochs_min: int = 5
    epochs_max: int = 80

    # ---------------------------------------------------------------------
    # Model hyperparameters
    # ---------------------------------------------------------------------
    hidden_1: int = 256
    hidden_2: int = 128
    hidden_3: int = 64
    dropout: float = 0.2

    def atlases_for_modality(self, modality: str):
        """Return atlas list for a modality."""
        if modality == "fc":
            return self.fc_atlases

        if modality == "sc":
            return self.sc_atlases

        raise ValueError(f"Unknown modality '{modality}'. Expected 'fc' or 'sc'.")
