"""
map_nodetime.py
=================

Learn the source-atlas -> target-atlas mapping matrix using optimal
transport applied to paired node-wise BOLD timeseries. This is the only
mapping-fitting method in the pipeline - the resulting matrix is reused to
map both SC and FC (see map_connectome.py).

Paired timeseries data (i.e. the same subjects' fMRI, parcellated with both
the source and target atlas) is required to fit the mapping.

Workflow
--------
1. Load paired node timeseries for both atlases from data/raw/timeseries/.
2. Fit a per-subject optimal-transport plan aligning the source atlas's node
   space to the target atlas's node space, then average across subjects.
3. Evaluate the mapping by comparing spectral-embedding-based
   reconstructions of the target FC against the mapped source FC (Pearson
   correlation of edge weights).
4. Save the mapping matrix (and evaluation scores) to data/mappings/.

Usage
-----
    python map_nodetime.py -s fs86 -t shen268
"""

import os
import sys
import argparse

import numpy as np
from scipy.stats import pearsonr


from src.utils.data_loading import load_raw_timeseries, load_raw_fc, save_mapping
from src.utils.mats2edges import com2edges

from Optimal_transport import Optimal_transport
from spectral_embedding import spectral_embedding


def parse_args():
    parser = argparse.ArgumentParser(
        description="Learn the atlas-to-atlas mapping via optimal transport on node timeseries."
    )
    parser.add_argument("-s", "--source", type=str, required=True, help="source atlas name")
    parser.add_argument("-t", "--target", type=str, required=True, help="target atlas name")
    parser.add_argument("-p", type=int, default=50, help="number of spectral embedding components (default: 50)")
    return parser.parse_args()


def evaluate_mapping(trans, fc_source, fc_target, p):
    """Compare spectral-embedding-based reconstructions of the mapped
    source FC to the true target FC, subject by subject."""
    nsub = fc_source.shape[2]
    scores = np.zeros(nsub)

    for i in range(nsub):
        Xp, Xn = spectral_embedding(fc_target[:, :, i], p=p)
        A_target = Xp @ Xp.T - Xn @ Xn.T

        Xp, Xn = spectral_embedding(fc_source[:, :, i], p=p)
        Xtp = trans.T @ Xp
        Xtn = trans.T @ Xn
        A_mapped = Xtp @ Xtp.T - Xtn @ Xtn.T

        scores[i] = pearsonr(com2edges(A_target), com2edges(A_mapped))[0]

    return scores


def main():
    args = parse_args()

    # --- Step 1: fit the mapping on paired timeseries ---
    ts_source = load_raw_timeseries(args.source)
    ts_target = load_raw_timeseries(args.target)
    T_time = Optimal_transport(ts_source, ts_target)
    trans = np.mean(T_time, axis=2)

    # --- Step 2: evaluate the mapping using FC ---
    fc_source = load_raw_fc(args.source)
    fc_target = load_raw_fc(args.target)
    scores = evaluate_mapping(trans, fc_source, fc_target, p=args.p)
    print(f"Mean reconstruction correlation (fc): {np.mean(scores):.4f}")

    # --- Step 3: save the mapping for reuse ---
    save_mapping(args.source, args.target, trans, eval_scores=scores)


if __name__ == "__main__":
    main()


