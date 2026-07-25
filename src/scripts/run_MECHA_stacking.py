"""
run_mecha_stacking.py
=====================

Thin wrapper for MECHA stacking/meta-matching.

Example
-------
python src/scripts/run_mecha_stacking.py \
    --source Schaefer_200 \
    --target Schaefer_400 \
    --mod fc \
    --phenotype_path /path/to/phenotypes.npy
"""

import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.MECHA.ukbb_stacking import main, parse_args


if __name__ == "__main__":
    main(parse_args())
