"""
map_connectome.py
==================

Apply a previously learned atlas-to-atlas mapping (fit by map_nodetime.py)
to translate structural (SC) and functional (FC) connectomes from a source
atlas's node space into a target atlas's node space.

No fitting happens here - this script only loads a saved mapping matrix and
applies it. Run map_nodetime.py first for a given atlas pair.

Workflow
--------
1. Load the saved source->target mapping matrix from data/mappings/.
2. Load raw SC and FC for the source atlas from data/raw/{sc,fc}/.
3. Map each subject's connectome into the target atlas's node space via
   spectral embedding + the mapping matrix.
4. If target-atlas SC/FC are available, evaluate mapping quality by
   comparing mapped connectomes to the true target connectomes (Pearson
   correlation of edge weights).
5. Save the mapped SC and FC to data/mapped/{sc,fc}/.

Usage
-----
    python map_connectome.py -s fs86 -t shen268
"""

import os
import sys
import argparse

import numpy as np
from scipy.stats import pearsonr

from src.utils.data_loading import (
    load_raw_sc, load_raw_fc,
    load_mapping,
    save_mapped_sc, save_mapped_fc,
)
from src.utils.mats2edges import com2edges
from spectral_embedding import spectral_embedding


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply a learned atlas-to-atlas mapping to SC and FC connectomes."
    )
    parser.add_argument("-s", "--source", type=str, required=True, help="source atlas name")
    parser.add_argument("-t", "--target", type=str, required=True, help="target atlas name")
    parser.add_argument("-p", type=int, default=50, help="number of spectral embedding components (default: 50)")
    return parser.parse_args()


def apply_mapping(trans, connectome, p):
    """Map a single connectome from the source atlas's node space into the
    target atlas's node space via spectral embedding.

    trans : np.ndarray, shape (dim_s, dim_t)
        Column-stochastic mapping matrix (columns sum to 1), as produced by
        map_nodetime.py.
    connectome : np.ndarray, shape (dim_s, dim_s)
    """
    Xp, Xn = spectral_embedding(connectome, p=p)
    Xtp = trans.T @ Xp
    Xtn = trans.T @ Xn
    return Xtp @ Xtp.T - Xtn @ Xtn.T


def map_all_subjects(trans, data_source, p):
    """Apply `apply_mapping` across all subjects.

    data_source : np.ndarray, shape (node, node, nsub)
    Returns : np.ndarray, shape (dim_t, dim_t, nsub)
    """
    nsub = data_source.shape[2]
    dim_t = trans.shape[1]
    mapped = np.zeros((dim_t, dim_t, nsub))
    for i in range(nsub):
        mapped[:, :, i] = apply_mapping(trans, data_source[:, :, i], p)
    return mapped


def evaluate_mapping(mapped, data_target):
    """Pearson correlation between mapped and true target connectome edges,
    subject by subject. Only meaningful when subjects are paired across
    source and target atlases."""
    nsub = data_target.shape[2]
    scores = np.zeros(nsub)
    for i in range(nsub):
        scores[i] = pearsonr(com2edges(data_target[:, :, i]), com2edges(mapped[:, :, i]))[0]
    return scores


def main():
    args = parse_args()

    # --- Step 1: load the previously learned mapping ---
    mapping = load_mapping(args.source, args.target)
    trans = mapping["mapping_matrix"]

    # --- Step 2: load source-atlas SC and FC, apply the mapping ---
    sc_source = load_raw_sc(args.source)
    fc_source = load_raw_fc(args.source)

    mapped_sc = map_all_subjects(trans, sc_source, p=args.p)
    mapped_fc = map_all_subjects(trans, fc_source, p=args.p)

    # --- Step 3: optionally evaluate against true target connectomes ---
    try:
        sc_target = load_raw_sc(args.target)
        scores_sc = evaluate_mapping(mapped_sc, sc_target)
        print(f"Mean reconstruction correlation (sc): {np.mean(scores_sc):.4f}")
    except FileNotFoundError:
        print("No target-atlas SC available for evaluation, skipping SC evaluation.")

    try:
        fc_target = load_raw_fc(args.target)
        scores_fc = evaluate_mapping(mapped_fc, fc_target)
        print(f"Mean reconstruction correlation (fc): {np.mean(scores_fc):.4f}")
    except FileNotFoundError:
        print("No target-atlas FC available for evaluation, skipping FC evaluation.")

    # --- Step 4: save mapped connectomes for reuse ---
    save_mapped_sc(args.source, args.target, mapped_sc)
    save_mapped_fc(args.source, args.target, mapped_fc)


if __name__ == "__main__":
    main()

