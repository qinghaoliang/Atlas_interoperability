"""
evaluation.py
=============

Evaluation utilities for FLECHA.
"""

import numpy as np
import torch
from scipy.stats import pearsonr


def evaluate_regression_model(model, loader, device):
    """
    Evaluate a regression model using MSE and Pearson correlation.

    Returns
    -------
    mse : float
    r : float
    p : float
    """
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            y_true.append(y.numpy())
            y_pred.append(model(x).cpu().numpy())

    y_true = np.concatenate(y_true).flatten()
    y_pred = np.concatenate(y_pred).flatten()

    mse = float(np.mean((y_true - y_pred) ** 2))

    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        return mse, 0.0, 1.0

    r, p = pearsonr(y_true, y_pred)
    return mse, float(r), float(p)
