"""
mics.py
=======

Utility metrics and logging functions for MECHA.
"""

import os
import numpy as np
from scipy.stats import pearsonr


def nanpearsonr(real, pred):
    """Compute Pearson correlation for each phenotype, omitting NaN values."""
    n = real.shape[1]
    res = np.zeros(n)

    for i in range(n):
        mask = np.logical_not(np.isnan(real[:, i]))

        if mask.sum() < 2 or np.std(real[mask, i]) == 0 or np.std(pred[mask, i]) == 0:
            res[i] = np.nan
        else:
            res[i] = pearsonr(real[mask, i], pred[mask, i])[0]

    return res


def cod_znormed(real, pred):
    """Compute coefficient of determination for z-normalized data."""
    tot = np.sum(real ** 2, axis=-1)
    res = np.sum((real - pred) ** 2, axis=-1)
    return 1 - res / tot


def nancod_znormed(real, pred):
    """Compute COD for z-normalized data, omitting NaN values."""
    tot = np.nansum(real ** 2, axis=-2)
    res = np.nansum((real - pred) ** 2, axis=-2)
    return 1 - np.divide(res, tot)


def split_tra_val_with_y(split, y_test):
    """Split subjects into training and validation subsets."""
    n = split.shape[0]
    n_rng = 1

    while n_rng != 0:
        k = np.where(split == 0)[0]
        m = k.shape[0]
        m_tra = int(m * 0.8)
        k = np.random.permutation(k)

        split_tra = np.zeros(n).astype(bool)
        split_val = np.zeros(n).astype(bool)

        split_tra[k[:m_tra]] = True
        split_val[k[m_tra:]] = True

        y_test_tra = y_test[split_tra]

        if np.unique(y_test_tra).shape[0] > 1:
            n_rng = 0
        else:
            np.random.seed(100 + n_rng)
            n_rng += 1

    return split_tra, split_val


def mics_z_norm(train_y, valid_y, test_y=None):
    """
    Z-normalize train, validation, and optional test labels using training-set
    statistics.
    """
    t_mu = np.nanmean(train_y, axis=0, keepdims=True)

    train_y = train_y - t_mu
    valid_y = valid_y - t_mu

    if test_y is not None:
        test_y = test_y - t_mu

    t_sigma = np.nanstd(train_y, axis=0)
    t_sigma = np.where(t_sigma == 0, 1, t_sigma)

    if train_y.ndim == 2:
        t_sigma_d = t_sigma[np.newaxis, :]
    else:
        t_sigma_d = t_sigma

    train_y = train_y / t_sigma_d
    valid_y = valid_y / t_sigma_d

    if test_y is not None:
        test_y = test_y / t_sigma_d

    return [train_y, valid_y, test_y, t_sigma]


def mics_infer_metric(
    dataloader,
    net,
    criterion,
    device,
    t_sigma=None,
    need_value=False,
    output_size=1,
):
    """Run inference and compute correlation/loss."""
    record_loss = 0.0
    record_real = np.zeros((0, output_size))
    record_pred = np.zeros((0, output_size))

    net.eval()

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        outputs = net(x)
        loss = criterion(outputs, y)

        record_loss += loss.item()
        record_real = np.concatenate((record_real, y.data.cpu().numpy()), axis=0)
        record_pred = np.concatenate((record_pred, outputs.data.cpu().numpy()), axis=0)

    cor = nanpearsonr(record_real, record_pred)
    loss = record_loss / max(1, len(dataloader))

    if need_value:
        return cor, loss, record_real, record_pred

    return cor, loss


def mics_log(model_name, out_path, metric="cor", index=None, **kwargs):
    """Save MECHA training records."""
    if index is None:
        val_record = kwargs["val_" + metric + "_record"]
        temp = np.mean(np.mean(val_record, axis=0), axis=1)
        temp = np.convolve(temp, np.ones(3, dtype=int), "valid") / 3
        index = np.nanargmax(temp) + 1
        print("\nBest validation at index:", index)

    val_cor_record = kwargs["val_cor_record"]

    file_str = model_name + "_base.npz"
    name_str = os.path.join(out_path, file_str)

    os.makedirs(out_path, exist_ok=True)
    np.savez(name_str, **kwargs)

    print("File saved:", name_str)
    print(
        "Average validation COR:",
        np.nanmean(np.nanmean(val_cor_record[:, index, :], axis=0)),
    )


def print_result(tra_cor, val_los, val_cor, epoch):
    """Print training and validation results."""
    print(
        f"Epoch {epoch}: "
        f"Train COR={np.nanmean(tra_cor):.5f}, "
        f"Val Loss={val_los:.5f}, "
        f"Val COR={np.nanmean(val_cor):.5f}"
    )

