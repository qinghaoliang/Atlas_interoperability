"""
experiment.py
=============

Main FLECHA experiment runner.

This module uses the canonical connectome loaders from `src.utils.data_loading`
rather than experiment-specific H5 files. It is designed for the peer-review
repository structure where controlled-access data are not included.

Workflow
--------
1. Load native target-atlas connectomes for testing.
2. Load native or mapped training connectomes depending on the experiment.
3. Convert connectome matrices to edge-feature datasets.
4. Run uniform, heterogeneous federated, or single-site baseline experiments.

Methods
-------
uniform:
    Upper-bound setting. All clients use native target-atlas connectomes.

federated:
    Heterogeneous-atlas setting. Each client uses source-atlas connectomes
    mapped into the target-atlas space.

single_site:
    Compute-matched single-client baseline using mapped source-to-target
    connectomes.
"""

import os
import argparse

import numpy as np
import pandas as pd
import torch

from src.utils.data_loading import (
    load_raw_fc,
    load_raw_sc,
    load_mapped_fc,
    load_mapped_sc,
)

from src.FLECHA.config import FlechaConfig
from src.FLECHA.dataset import (
    make_subject_splits,
    make_loader,
    infer_input_size,
)
from src.FLECHA.model import ConnectomeRegressor4L
from src.FLECHA.train import (
    federated_fedavg,
    train_local,
    compute_matched_epochs,
)
from src.FLECHA.evaluation import evaluate_regression_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run FLECHA experiments on raw or atlas-mapped connectomes."
    )

    parser.add_argument(
        "--modality",
        choices=["fc", "sc"],
        required=True,
        help="Connectome modality.",
    )
    parser.add_argument(
        "--target_atlas",
        required=True,
        help="Target atlas used for testing and mapped feature space.",
    )
    parser.add_argument(
        "--method",
        choices=["uniform", "federated", "single_site"],
        required=True,
        help="Experiment setting.",
    )
    parser.add_argument(
        "--run_id",
        type=int,
        required=True,
        help="Random seed / repeated-run identifier.",
    )
    parser.add_argument(
        "--target_path",
        default=None,
        help="Optional local .npy target vector with one value per subject. "
             "If omitted, synthetic targets are generated for smoke testing.",
    )
    parser.add_argument(
        "--output_dir",
        default="./results/flecha",
        help="Directory for result CSV files.",
    )

    return parser.parse_args()


def load_connectomes(modality, atlas=None, source_atlas=None, target_atlas=None, mapped=False):
    """
    Load native or mapped connectomes using `src.utils.data_loading`.

    Parameters
    ----------
    modality : {'fc', 'sc'}
    atlas : str
        Atlas name for native raw connectomes.
    source_atlas, target_atlas : str
        Atlas pair for mapped connectomes.
    mapped : bool
        Whether to load mapped source-to-target connectomes.
    """
    if modality == "fc":
        if mapped:
            return load_mapped_fc(source_atlas, target_atlas)
        return load_raw_fc(atlas)

    if modality == "sc":
        if mapped:
            return load_mapped_sc(source_atlas, target_atlas)
        return load_raw_sc(atlas)

    raise ValueError(f"Unknown modality '{modality}'. Expected 'fc' or 'sc'.")


def load_or_make_targets(target_path, n_subjects, seed):
    """
    Load a local target vector or create synthetic targets for smoke testing.

    Synthetic targets are only for checking code flow and are not used for
    manuscript results.
    """
    if target_path is not None:
        targets = np.load(target_path).reshape(-1)

        if len(targets) != n_subjects:
            raise ValueError(
                f"Target vector length ({len(targets)}) does not match "
                f"number of subjects ({n_subjects})."
            )

        return targets.astype(np.float32)

    rng = np.random.default_rng(seed)

    print(
        "[WARN] No --target_path provided. Generating synthetic targets for "
        "smoke testing only. These targets are not used in manuscript results."
    )

    return rng.normal(size=n_subjects).astype(np.float32)


def run_uniform(
    cfg,
    modality,
    target_atlas,
    targets,
    train_splits,
    test_loader,
    input_size,
    device,
):
    """
    Run the uniform upper-bound setting.

    All training clients use native target-atlas connectomes, but each client
    receives a different subject split.
    """
    datasets = []
    loaders = []

    target_connectomes = load_connectomes(
        modality=modality,
        atlas=target_atlas,
        mapped=False,
    )

    for split_idx in train_splits:
        dataset, loader = make_loader(
            connectomes=target_connectomes,
            targets=targets,
            indices=split_idx,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
        )

        if len(dataset) < 2:
            continue

        datasets.append(dataset)
        loaders.append(loader)

    if len(loaders) == 0:
        raise ValueError("No valid training clients for uniform experiment.")

    model = federated_fedavg(
        loaders=loaders,
        device=device,
        input_size=input_size,
        rounds=cfg.rounds,
        local_steps=cfg.local_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        hidden_1=cfg.hidden_1,
        hidden_2=cfg.hidden_2,
        hidden_3=cfg.hidden_3,
        dropout=cfg.dropout,
    )

    mse, r, p = evaluate_regression_model(model, test_loader, device)

    return [
        {
            "method": "uniform",
            "mse": mse,
            "r": r,
            "p": p,
            "n_clients": len(loaders),
        }
    ]


def run_federated(
    cfg,
    modality,
    target_atlas,
    train_atlases,
    targets,
    train_splits,
    test_loader,
    input_size,
    device,
):
    """
    Run heterogeneous-atlas FLECHA.

    Each client uses source-atlas connectomes mapped into the target-atlas
    space.
    """
    loaders = []
    used_sources = []

    for split_idx, source_atlas in zip(train_splits, train_atlases):
        mapped_connectomes = load_connectomes(
            modality=modality,
            source_atlas=source_atlas,
            target_atlas=target_atlas,
            mapped=True,
        )

        dataset, loader = make_loader(
            connectomes=mapped_connectomes,
            targets=targets,
            indices=split_idx,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
        )

        if len(dataset) < 2:
            print(
                f"[WARN] Skipping client source_atlas={source_atlas} "
                f"because n_train={len(dataset)} after filtering."
            )
            continue

        loaders.append(loader)
        used_sources.append(source_atlas)

    if len(loaders) == 0:
        raise ValueError("No valid training clients for federated experiment.")

    model = federated_fedavg(
        loaders=loaders,
        device=device,
        input_size=input_size,
        rounds=cfg.rounds,
        local_steps=cfg.local_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        hidden_1=cfg.hidden_1,
        hidden_2=cfg.hidden_2,
        hidden_3=cfg.hidden_3,
        dropout=cfg.dropout,
    )

    mse, r, p = evaluate_regression_model(model, test_loader, device)

    return [
        {
            "method": "federated",
            "mse": mse,
            "r": r,
            "p": p,
            "n_clients": len(loaders),
            "source_atlases": ";".join(used_sources),
        }
    ]


def run_single_site(
    cfg,
    modality,
    target_atlas,
    train_atlases,
    targets,
    train_splits,
    test_loader,
    input_size,
    device,
    run_id,
):
    """
    Run compute-matched single-site baselines.

    Each source atlas is trained independently after mapping into the target
    atlas feature space.
    """
    results = []
    n_clients_total = len(train_splits)

    for split_idx, source_atlas in zip(train_splits, train_atlases):
        mapped_connectomes = load_connectomes(
            modality=modality,
            source_atlas=source_atlas,
            target_atlas=target_atlas,
            mapped=True,
        )

        dataset, loader = make_loader(
            connectomes=mapped_connectomes,
            targets=targets,
            indices=split_idx,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
        )

        if len(dataset) < 2:
            print(
                f"[WARN] Skipping single-site source_atlas={source_atlas} "
                f"because n_train={len(dataset)} after filtering."
            )
            continue

        epochs = compute_matched_epochs(
            n_samples_site=len(dataset),
            n_clients=n_clients_total,
            rounds=cfg.rounds,
            local_steps=cfg.local_steps,
            batch_size=cfg.batch_size,
            epochs_min=cfg.epochs_min,
            epochs_max=cfg.epochs_max,
        )

        print(
            f"[single_site] run={run_id} "
            f"source_atlas={source_atlas} "
            f"n_train={len(dataset)} "
            f"matched_epochs={epochs}"
        )

        model = ConnectomeRegressor4L(
            input_size=input_size,
            n_l1=cfg.hidden_1,
            n_l2=cfg.hidden_2,
            n_l3=cfg.hidden_3,
            dropout=cfg.dropout,
        ).to(device)

        train_local(
            model=model,
            loader=loader,
            device=device,
            epochs=epochs,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        mse, r, p = evaluate_regression_model(model, test_loader, device)

        results.append(
            {
                "method": "single_site",
                "train_src_atlas": source_atlas,
                "matched_epochs": epochs,
                "n_train": int(len(dataset)),
                "mse": mse,
                "r": r,
                "p": p,
            }
        )

    return results


def main():
    args = parse_args()
    cfg = FlechaConfig()

    torch.manual_seed(args.run_id)
    np.random.seed(args.run_id)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    atlases = cfg.atlases_for_modality(args.modality)

    if args.target_atlas not in atlases:
        raise ValueError(
            f"target_atlas '{args.target_atlas}' not in atlas list: {atlases}"
        )

    train_atlases = [atlas for atlas in atlases if atlas != args.target_atlas]

    # ---------------------------------------------------------------------
    # Load target-atlas native connectomes for test set
    # ---------------------------------------------------------------------
    target_connectomes = load_connectomes(
        modality=args.modality,
        atlas=args.target_atlas,
        mapped=False,
    )

    n_subjects = target_connectomes.shape[2]
    targets = load_or_make_targets(
        target_path=args.target_path,
        n_subjects=n_subjects,
        seed=args.run_id,
    )

    # ---------------------------------------------------------------------
    # Subject splits
    # ---------------------------------------------------------------------
    n_splits = len(atlases)
    splits = make_subject_splits(
        n_subjects=n_subjects,
        n_splits=n_splits,
        seed=args.run_id,
    )

    test_split = splits[0]
    train_splits = splits[1:]

    if len(train_splits) != len(train_atlases):
        raise RuntimeError("train_splits and train_atlases must both be N-1.")

    test_dataset, test_loader = make_loader(
        connectomes=target_connectomes,
        targets=targets,
        indices=test_split,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )

    input_size = infer_input_size(test_loader)

    # ---------------------------------------------------------------------
    # Run experiment
    # ---------------------------------------------------------------------
    if args.method == "uniform":
        results = run_uniform(
            cfg=cfg,
            modality=args.modality,
            target_atlas=args.target_atlas,
            targets=targets,
            train_splits=train_splits,
            test_loader=test_loader,
            input_size=input_size,
            device=device,
        )

    elif args.method == "federated":
        results = run_federated(
            cfg=cfg,
            modality=args.modality,
            target_atlas=args.target_atlas,
            train_atlases=train_atlases,
            targets=targets,
            train_splits=train_splits,
            test_loader=test_loader,
            input_size=input_size,
            device=device,
        )

    else:
        results = run_single_site(
            cfg=cfg,
            modality=args.modality,
            target_atlas=args.target_atlas,
            train_atlases=train_atlases,
            targets=targets,
            train_splits=train_splits,
            test_loader=test_loader,
            input_size=input_size,
            device=device,
            run_id=args.run_id,
        )

    out_path = os.path.join(
        args.output_dir,
        f"flecha_{args.modality}_{args.target_atlas}_{args.method}_run{args.run_id}.csv",
    )

    pd.DataFrame(results).to_csv(out_path, index=False)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
