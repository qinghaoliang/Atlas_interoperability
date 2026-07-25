"""
dataset.py
==========

Dataset utilities for FLECHA.

This module converts loaded connectome matrices into edge-feature regression
datasets for federated learning.

Expected connectome shape
-------------------------
The canonical loaders in `src.utils.data_loading` return arrays of shape:

    node × node × subject

FLECHA converts each subject's matrix into an edge vector, yielding:

    subject × edge
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.mats2edges import com2edges


class ConnectomeDataset(Dataset):
    """
    In-memory connectome edge-feature dataset.

    Parameters
    ----------
    connectomes : np.ndarray
        Connectome array with shape node × node × subject.
    targets : np.ndarray
        Target vector with shape subject.
    indices : array-like or None
        Subject indices to include. If None, all subjects are used.
    """

    def __init__(self, connectomes, targets, indices=None):
        if connectomes.ndim != 3:
            raise ValueError(
                "connectomes must have shape node × node × subject."
            )

        n_subjects = connectomes.shape[2]
        targets = np.asarray(targets).reshape(-1)

        if len(targets) != n_subjects:
            raise ValueError(
                f"targets length ({len(targets)}) must match number of "
                f"subjects in connectomes ({n_subjects})."
            )

        if indices is None:
            indices = np.arange(n_subjects)

        indices = np.asarray(indices, dtype=np.int64).reshape(-1)

        if len(indices) == 0:
            raise ValueError("No subject indices were provided.")

        if indices.min() < 0 or indices.max() >= n_subjects:
            raise IndexError(
                f"indices out of range for {n_subjects} subjects."
            )

        # Filter finite targets
        y = targets[indices].astype(np.float32)
        finite_y = np.isfinite(y)
        indices = indices[finite_y]
        y = y[finite_y]

        # Convert connectome matrices to edge features
        features = []
        kept_targets = []

        for idx, target in zip(indices, y):
            mat = connectomes[:, :, idx]
            edge = com2edges(mat).astype(np.float32)

            if np.isfinite(edge).all() and np.isfinite(target):
                features.append(edge)
                kept_targets.append(target)

        if len(features) == 0:
            raise ValueError("No valid subjects after finite-value filtering.")

        self.X = torch.tensor(np.vstack(features), dtype=torch.float32)
        self.y = torch.tensor(np.asarray(kept_targets), dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def make_loader(
    connectomes,
    targets,
    indices,
    batch_size,
    shuffle,
    drop_last,
):
    """Create a PyTorch DataLoader from connectome matrices."""
    dataset = ConnectomeDataset(
        connectomes=connectomes,
        targets=targets,
        indices=indices,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    return dataset, loader


def infer_input_size(loader):
    """Infer feature dimension from the first minibatch."""
    return next(iter(loader))[0].shape[1]


def make_subject_splits(n_subjects, n_splits, seed):
    """
    Create random subject splits for simulated federated clients.

    This peer-review version does not depend on real site labels or phenotype
    files. For controlled-access experiments, users may replace this with a
    site-stratified split using their local metadata.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(n_subjects)
    rng.shuffle(indices)

    splits = np.array_split(indices, n_splits)
    return [np.asarray(split, dtype=np.int64) for split in splits]
