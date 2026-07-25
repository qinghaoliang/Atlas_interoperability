"""
ukbb_stacking.py
====================
Run MECHA stacking/meta-matching analysis on UK Biobank connectomes.
This script loads pretrained DNN models and uses their outputs as meta-features for stacking regression. The stacking model is trained on a small query set
and evaluated on a held-out test set. The results are saved in the `results/MECHA/stacking` directory
Examples
--------
Run ridge stacking on mapped FC:

    python -m src.MECHA.ukbb_stacking \
        --source Schaefer_200 \
        --target Schaefer_400 \
        --mod fc \
        --phenotype_path /path/to/phenotypes.npy \
        --query_indices /path/to/query_indices.npy \
        --test_indices /path/to/test_indices.npy
"""

import os
import argparse

import numpy as np
import torch

from torch.utils.data import DataLoader
from scipy.stats import pearsonr

from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

from src.MECHA.config import config
from src.MECHA.dataset import (
    ConnectomeDataset,
    load_phenotype_array,
    load_indices,
)
from src.MECHA.model_pytorch import dnn_4l
from src.MECHA.mics import mics_z_norm

from src.utils.data_loading import (
    load_raw_fc,
    load_raw_sc,
    load_mapped_fc,
    load_mapped_sc,
)


def load_mecha_connectomes(source, target, key):
    """Load native or mapped connectomes using shared data loaders."""
    if key == "fc":
        if source == target:
            return load_raw_fc(target)
        return load_mapped_fc(source, target)

    if key == "sc":
        if source == target:
            return load_raw_sc(target)
        return load_mapped_sc(source, target)

    raise ValueError(f"Unknown key '{key}'. Expected 'fc' or 'sc'.")


def get_cv(k):
    """Choose cross-validation folds based on query sample size."""
    if k <= 20:
        return 2
    if k <= 50:
        return 3
    return 5


def get_stacking_model(name, k):
    """Create stacking model with a small hyperparameter grid."""
    cv = get_cv(k)

    if name == "ridge":
        model = Ridge()
        param_grid = {"alpha": [0.1, 0.3, 1, 3, 10, 30, 100]}

    elif name == "krr":
        model = KernelRidge()
        param_grid = {
            "alpha": [0.01, 0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": [None, 0.01, 0.1, 1],
        }

    elif name == "svr":
        model = SVR()
        param_grid = {
            "C": [0.1, 1, 10, 100],
            "epsilon": [0.01, 0.1, 0.2],
            "kernel": ["linear", "rbf"],
        }

    else:
        raise ValueError(f"Unsupported stacking model: {name}")

    return GridSearchCV(
        model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
        error_score="raise",
    )


def extract_features(net, loader, device):
    """
    Extract DNN outputs used as meta-features for stacking.

    This preserves the original behavior, where the pretrained DNN output is
    used as the representation for downstream meta-matching.
    """
    net.eval()
    outputs = []

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            out = net(x)
            outputs.append(out.cpu().numpy())

    return np.vstack(outputs)


def safe_pearsonr(y_true, y_pred):
    """Pearson correlation with finite-value guards."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    if mask.sum() < 2:
        return np.nan, np.nan

    if np.std(y_true[mask]) <= 1e-8 or np.std(y_pred[mask]) <= 1e-8:
        return 0.0, 1.0

    return pearsonr(y_true[mask], y_pred[mask])


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )

    # ------------------------------------------------------------------
    # Load connectomes and phenotypes
    # ------------------------------------------------------------------
    connectomes = load_mecha_connectomes(args.source, args.target, args.mod)
    phenotypes = load_phenotype_array(args.phenotype_path)

    n_subjects = connectomes.shape[2]

    if phenotypes.shape[0] != n_subjects:
        raise ValueError(
            f"Phenotype rows ({phenotypes.shape[0]}) do not match "
            f"connectome subjects ({n_subjects})."
        )

    output_size = phenotypes.shape[1]

    # ------------------------------------------------------------------
    # Load pretrained DNN
    # ------------------------------------------------------------------
    model_dir = os.path.join(
        config.MODEL_DIR,
        f"model_{args.source}_to_{args.target}_{args.mod}",
    )
    model_path = os.path.join(model_dir, "best_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Pretrained model not found:\n{model_path}\n\n"
            "Run `src.MECHA.ukbb_dnn` first for the same source/target/modality."
        )

    os.makedirs(config.STACKING_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Prepare test indices
    # ------------------------------------------------------------------
    test_indices = load_indices(
        path=args.test_indices,
        n_subjects=n_subjects,
        seed=args.seed + 1000,
        n_samples=args.n_test,
    )

    results = []

    for k in args.k_list:
        print(f"\nEvaluating MECHA stacking with k={k}")

        query_indices = load_indices(
            path=args.query_indices,
            n_subjects=n_subjects,
            seed=args.seed,
            n_samples=k,
        )

        test_dataset = ConnectomeDataset(
            connectomes=connectomes,
            phenotypes=phenotypes,
            indices=test_indices,
        )

        query_dataset = ConnectomeDataset(
            connectomes=connectomes,
            phenotypes=phenotypes,
            indices=query_indices,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )

        query_loader = DataLoader(
            query_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )

        input_dim = test_dataset[0][0].numel()

        net = dnn_4l(
            input_dim,
            args.n_l1,
            args.n_l2,
            args.n_l3,
            dropout=args.dropout,
            output_size=output_size,
        ).to(device)

        net.load_state_dict(torch.load(model_path, map_location=device))

        x_query = extract_features(net, query_loader, device)
        y_query = np.array([y.numpy() for _, y in query_dataset])

        x_test = extract_features(net, test_loader, device)
        y_test = np.array([y.numpy() for _, y in test_dataset])

        meta_cor = np.zeros(output_size)
        meta_p = np.zeros(output_size)

        for i in range(output_size):
            yq = y_query[:, i].reshape(-1, 1)
            yt = y_test[:, i].reshape(-1, 1)

            yq, yt, _, _ = mics_z_norm(yq, yt)

            mask_q = np.isfinite(yq).flatten()

            if mask_q.sum() < 2:
                meta_cor[i] = np.nan
                meta_p[i] = np.nan
                continue

            stack_model = get_stacking_model(args.stack_model, k)
            stack_model.fit(x_query[mask_q], yq[mask_q].ravel())

            print(
                f"[k={k}, phenotype={i}] "
                f"best params: {stack_model.best_params_}"
            )

            y_pred = stack_model.predict(x_test)
            r, p = safe_pearsonr(yt.flatten(), y_pred)

            meta_cor[i] = r
            meta_p[i] = p

        mean_cor = np.nanmean(meta_cor)
        mean_p = np.nanmean(meta_p)

        print(
            f"seed={args.seed}, k={k}: "
            f"mean cor {mean_cor:.5f}, mean p {mean_p:.5f}"
        )

        out_npz = os.path.join(
            config.STACKING_DIR,
            f"stacking_{args.stack_model}_{args.source}_to_"
            f"{args.target}_{args.mod}_ks{k}_{args.seed}.npz",
        )

        np.savez(
            out_npz,
            meta_cor=meta_cor,
            meta_p=meta_p,
            query_indices=query_indices,
            test_indices=test_indices,
        )

        results.append(
            {
                "k": k,
                "mean_cor": mean_cor,
                "mean_p": mean_p,
                "output_file": out_npz,
            }
        )

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MECHA stacking/meta-matching analysis."
    )

    parser.add_argument("--source", type=str, required=True, choices=config.ATLAS_CHOICES)
    parser.add_argument("--target", type=str, required=True, choices=config.ATLAS_CHOICES)
    parser.add_argument("--mod", type=str, choices=["fc", "sc"], required=True)

    parser.add_argument(
        "--phenotype_path",
        type=str,
        required=True,
        help="Local .npy phenotype array with shape subject × phenotype.",
    )

    parser.add_argument(
        "--query_indices",
        type=str,
        default=None,
        help="Optional .npy file containing query subject indices.",
    )

    parser.add_argument(
        "--test_indices",
        type=str,
        default=None,
        help="Optional .npy file containing test subject indices.",
    )

    parser.add_argument(
        "--n_test",
        type=int,
        default=None,
        help="Number of test subjects to sample if --test_indices is omitted.",
    )

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--stack_model",
        type=str,
        choices=["ridge", "krr", "svr"],
        default="ridge",
    )
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--k_list", nargs="+", type=int, default=[10, 20, 50, 100, 200])

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=config.DROPOUT)
    parser.add_argument("--n_l1", type=int, default=config.N_L1)
    parser.add_argument("--n_l2", type=int, default=config.N_L2)
    parser.add_argument("--n_l3", type=int, default=config.N_L3)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())


