"""
run_flecha.py
=============

Thin wrapper for running FLECHA experiments.

FLECHA uses native and atlas-mapped connectomes loaded through
`src.utils.data_loading`.

Example
-------
Run heterogeneous federated learning on mapped FC:

    python src/scripts/run_flecha.py \
        --modality fc \
        --target_atlas Schaefer_400 \
        --method federated \
        --run_id 1 \
        --target_path /path/to/local_targets.npy

Run uniform upper-bound baseline:

    python src/scripts/run_flecha.py \
        --modality fc \
        --target_atlas Schaefer_400 \
        --method uniform \
        --run_id 1 \
        --target_path /path/to/local_targets.npy

Run single-site baselines:

    python src/scripts/run_flecha.py \
        --modality sc \
        --target_atlas Schaefer_500 \
        --method single_site \
        --run_id 1 \
        --target_path /path/to/local_targets.npy
"""

import os
import sys


# ---------------------------------------------------------------------
# Make project root importable when this script is called directly.
# This allows:
#
#     python src/scripts/run_flecha.py ...
#
# from the repository root.
# ---------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.FLECHA.experiment import main


if __name__ == "__main__":
    main()
