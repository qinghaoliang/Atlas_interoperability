"""
dataset.py
==========

Dataset utilities for MECHA.

This module adapts MECHA to the canonical `.npz` connectome layout used in the
peer-review repository. Connectomes are loaded through `src.utils.data_loading`
and converted into edge-vector features for PyTorch models.

Expected connectome shape
-------------------------
The canonical loaders return arrays with shape:

    node × node × subject

MECHA converts these into:

    subject × edge
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.mats2edges import com2edges


class ConnectomeDataset(Dataset):
    """
    In-memory connectome dataset for MECHA.

    Parameters
    ----------
    connectomes : np.ndarray
        Connectome matrices with shape node × node × subject.
    phenotypes : np.ndarray
        Phenotype matrix with shape subject × phenotype.
        A one-dimensional array is converted to subject × 1.
    indices : array-like or None
        Subject indices to include. If None, all subjects are used.
    filter_nan_x : bool
        Whether to remove subjects with non-finite connectome edge values.
    filter_all_nan_y : bool
        Whether to remove subjects whose phenotype vector is entirely NaN.
    """

    def __init__(
        self,
        connectomes,
        phenotypes,
        indices=None,
        filter_nan_x=True,
        filter_all_nan_y=True,
    ):
        if connectomes.ndim != 3:
            raise ValueError("connectomes must have shape node × node × subject.")

        n_subjects = connectomes.shape[2]

        phenotypes = np.asarray(phenotypes, dtype=np.float32)

        if phenotypes.ndim == 1:
            phenotypes = phenotypes.reshape(-1, 1)

        if phenotypes.shape[0] != n_subjects:
            raise ValueError(
                f"phenotypes first dimension ({phenotypes.shape[0]}) must match "
                f"number of subjects in connectomes ({n_subjects})."
            )

        if indices is None:
            indices = np.arange(n_subjects)

        indices = np.asarray(indices, dtype=np.int64).reshape(-1)

        if len(indices) == 0:
            raise ValueError("No subject indices provided.")

        if indices.min() < 0 or indices.max() >= n_subjects:
            raise IndexError(
                f"indices out of range for connectomes with {n_subjects} subjects."
            )

        features = []
        labels = []
        kept_indices = []

        for subject_idx in indices:
            mat = connectomes[:, :, subject_idx]
            edge = com2edges(mat).astype(np.float32)
            y = phenotypes[subject_idx].astype(np.float32)

            if filter_nan_x and not np.isfinite(edge).all():
                continue

            if filter_all_nan_y and np.isnan(y).all():
                continue

            features.append(edge)
            labels.append(y)
            kept_indices.append(subject_idx)

        if len(features) == 0:
            raise ValueError("No valid samples after finite-value filtering.")

        self.x = torch.tensor(np.vstack(features), dtype=torch.float32)
        self.y = torch.tensor(np.vstack(labels), dtype=torch.float32)
        self.indices = np.asarray(kept_indices, dtype=np.int64)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return int(self.x.shape[0])


def load_phenotype_array(path):
    """
    Load a local phenotype array.

    Expected shape:
        subject × phenotype

    A one-dimensional vector is automatically reshaped to subject × 1.
    """
    if path is None:
        raise ValueError(
            "A phenotype file is required for MECHA. Please provide "
            "`--phenotype_path` pointing to a local .npy phenotype array."
        )

    phenotypes = np.load(path)

    if phenotypes.ndim == 1:
        phenotypes = phenotypes.reshape(-1, 1)

    return phenotypes.astype(np.float32)


def load_indices(path, n_subjects, seed=None, n_samples=None):
    """
    Load subject indices or generate a random subset.

    Parameters
    ----------
    path : str or None
        Optional .npy index file.
    n_subjects : int
        Total number of subjects.
    seed : int or None
        Random seed used when path is None.
    n_samples : int or None
        Number of subjects to sample when path is None.
    """
    if path is not None:
        indices = np.load(path).astype(np.int64).reshape(-1)
    else:
        rng = np.random.default_rng(seed)
        indices = np.arange(n_subjects, dtype=np.int64)

        if n_samples is not None:
            if n_samples > n_subjects:
                raise ValueError(
                    f"Requested n_samples={n_samples}, but only "
                    f"{n_subjects} subjects are available."
                )
            indices = rng.choice(indices, size=n_samples, replace=False)

    return indices
