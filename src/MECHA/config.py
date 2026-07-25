"""
config.py
=========

Configuration defaults for MECHA experiments.

Controlled-access participant data and phenotype files are not included.
Users may provide local phenotype arrays via command-line arguments.
"""

import os


class config:
    # ---------------------------------------------------------------------
    # Repository paths
    # ---------------------------------------------------------------------
    BASE_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )

    OUT_DIR = os.path.join(BASE_DIR, "results", "MECHA")
    MODEL_DIR = os.path.join(OUT_DIR, "models")
    STACKING_DIR = os.path.join(OUT_DIR, "stacking")

    # ---------------------------------------------------------------------
    # Defaults
    # ---------------------------------------------------------------------
    DATA_KEY = "fc"
    RANDOM_SEED = 42

    BATCH_SIZE = 128
    EPOCHS = 300

    LR = 1e-3
    WEIGHT_DECAY = 1e-3
    DROPOUT = 0.25

    N_LAYER = 4
    N_L1 = 256
    N_L2 = 128
    N_L3 = 64
    N_L4 = 32

    SCHEDULER_DECREASE = 150

    # ---------------------------------------------------------------------
    # Atlas choices used in the manuscript experiments
    # ---------------------------------------------------------------------
    ATLAS_CHOICES = [
        "glasser",
        "Schaefer_100",
        "Schaefer_200",
        "Schaefer_300",
        "Schaefer_400",
        "Schaefer_500",
    ]
