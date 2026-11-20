"""MECHA: meta-matching across heterogeneous atlas spaces.

Manuscript reference: Methods, "Meta-matching with heterogeneous atlas
spaces"; Fig. 4; Supplementary Figs. 2, 8, 9; Supplementary Tables 2-3.

Workflow
--------
1. ``train``: map source-atlas meta-training and validation connectomes into
   the target atlas, fit a multi-output DNN on the meta-training phenotypes
   using only the meta-training participants, and select the epoch with the
   lowest validation loss (early stopping on the separate validation cohort).
2. ``adapt``: feed native target-atlas support and test connectomes to the
   trained DNN. Its phenotype *predictions* (one per meta-training phenotype)
   are the features. For each held-out phenotype and support draw, fit a ridge
   model on the support set and evaluate on the independent test set. The
   same support draws and test participants are used for the support-only
   ridge and DNN baselines, which see only native support connectivity.
3. ``summarize``: benchmark selection, MECHA gain and paired tests.

Test labels are used only to score predictions.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import __version__
from . import data as D
from .models import MLPRegressor, MLPSpec, get_device, nan_mse, predict, seed_everything
from .presets import MechaPreset
from .stats import bh_fdr, fisher_mean_r, fisher_z, pearson, wilcoxon_greater
from .transform import apply_mapping, load_mapping

CHECKPOINT_FORMAT = "atlas_interop.mecha_checkpoint"
CHECKPOINT_VERSION = 1


class MechaInputError(D.DataContractError):
    pass


# ---------------------------------------------------------------------------
# Meta-training
# ---------------------------------------------------------------------------
def _prepare_cohort(conn: D.ConnectomeSet, table: D.PhenotypeTable, columns, mapping, rank, name):
    y = D.phenotypes_for(conn, table, columns)
    keep = ~np.isnan(y).all(axis=1)
    report = dict(cohort=name, n_loaded=int(conn.n_samples), n_all_labels_missing=int((~keep).sum()),
                  n_used=int(keep.sum()),
                  missing_fraction_per_phenotype=dict(zip(columns, np.isnan(y[keep]).mean(0).round(4).tolist())))
    conn = conn.take(np.flatnonzero(keep))
    y = y[keep]
    if mapping is not None:
        conn, tr = apply_mapping(conn, mapping, rank)
        report["transform"] = tr
    return conn.edges().astype(np.float32), y.astype(np.float32), conn, report


def train_dnn(*, train_conn: D.ConnectomeSet, val_conn: D.ConnectomeSet, table: D.PhenotypeTable,
              train_phenotypes: list[str], mapping_path: str | None, rank: int, preset: MechaPreset,
              seed: int, out_dir: str, device: str = "auto") -> dict:
    """Train the meta-training DNN and save a self-describing checkpoint."""
    if len(train_phenotypes) < 1:
        raise MechaInputError("at least one meta-training phenotype is required")
    if len(set(train_phenotypes)) != len(train_phenotypes):
        raise MechaInputError("meta-training phenotype names must be unique")
    D.assert_disjoint_subjects({"meta-training": train_conn.subject_ids, "validation": val_conn.subject_ids})
    mapping = load_mapping(mapping_path) if mapping_path else None
    if mapping is None:
        if train_conn.atlas != val_conn.atlas:
            raise MechaInputError("without a mapping, training and validation must share one atlas")
        target_atlas, n_target = train_conn.atlas, train_conn.n_regions
        target_region_ids = train_conn.region_ids
    else:
        target_atlas, n_target, target_region_ids = mapping.target_atlas, mapping.n_target, mapping.target_region_ids
        for c, nm in ((train_conn, "meta-training"), (val_conn, "validation")):
            if c.atlas is not None and c.atlas != mapping.source_atlas:
                raise MechaInputError(f"{nm} connectomes are '{c.atlas}', mapping source is '{mapping.source_atlas}'")
    Xtr, ytr, ctr, rep_tr = _prepare_cohort(train_conn, table, train_phenotypes, mapping, rank, "meta-training")
    Xva, yva, cva, rep_va = _prepare_cohort(val_conn, table, train_phenotypes, mapping, rank, "validation")
    if len(ytr) < preset.batch_size:
        raise MechaInputError(f"meta-training set has {len(ytr)} usable samples, fewer than batch size {preset.batch_size}")
    # z-score training targets with meta-training statistics only
    mu = np.nanmean(ytr, axis=0)
    sd = np.nanstd(ytr, axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    if np.isnan(mu).any():
        bad = [p for p, m in zip(train_phenotypes, mu) if np.isnan(m)]
        raise MechaInputError(f"meta-training phenotype(s) with no observed values: {bad}")
    ztr, zva = (ytr - mu) / sd, (yva - mu) / sd
    dev = get_device(device)
    seed_everything(seed)
    spec = MLPSpec(Xtr.shape[1], list(preset.hidden), len(train_phenotypes), preset.dropout, preset.norm)
    model = MLPRegressor(spec).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=preset.learning_rate, weight_decay=preset.weight_decay)
    rng = np.random.default_rng(seed)
    Xva_t = torch.as_tensor(Xva, device=dev)
    zva_t = torch.as_tensor(zva, device=dev)
    history, best = [], dict(epoch=-1, val_loss=np.inf, state=None)
    stale = 0
    for epoch in range(preset.max_epochs):
        model.train()
        order = rng.permutation(len(ztr))
        losses = []
        for start in range(0, len(order) - preset.batch_size + 1, preset.batch_size):
            idx = order[start:start + preset.batch_size]
            xb = torch.as_tensor(Xtr[idx], device=dev)
            yb = torch.as_tensor(ztr[idx], device=dev)
            opt.zero_grad()
            loss = nan_mse(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            vl = float(nan_mse(model(Xva_t), zva_t))
        history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), val_loss=vl))
        if vl < best["val_loss"]:
            best = dict(epoch=epoch, val_loss=vl, state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
            if preset.patience and stale >= preset.patience:
                break
    if best["state"] is None:
        raise RuntimeError("training produced no finite validation loss")
    # validation metrics of the selected checkpoint (validation labels select, never train)
    model.load_state_dict(best["state"])
    zpred = predict(model, Xva, dev)
    val_metrics = []
    for j, phen in enumerate(train_phenotypes):
        ok = np.isfinite(zva[:, j])
        r, _ = pearson(zva[ok, j], zpred[ok, j])
        val_metrics.append(dict(phenotype=phen, output_index=j, validation_r=r,
                                validation_mse_z=float(np.mean((zpred[ok, j] - zva[ok, j]) ** 2)),
                                n_validation=int(ok.sum())))
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(val_metrics).to_csv(os.path.join(out_dir, "validation_metrics.csv"), index=False)
    ckpt_path = os.path.join(out_dir, "mecha_dnn.pt")
    torch.save(dict(
        format=CHECKPOINT_FORMAT, format_version=CHECKPOINT_VERSION, toolkit_version=__version__,
        state_dict=best["state"], spec=spec.as_dict(), target_atlas=target_atlas, n_target_regions=n_target,
        target_region_ids=None if target_region_ids is None else [str(x) for x in target_region_ids],
        source_atlas=mapping.source_atlas if mapping else target_atlas,
        vectorization=D.VECTORIZATION, train_phenotypes=list(train_phenotypes),
        target_mean=mu.tolist(), target_sd=sd.tolist(), spectral_rank=rank if mapping else None,
        best_epoch=best["epoch"], best_val_loss=best["val_loss"], preset=preset.as_dict(), seed=seed,
    ), ckpt_path)
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "training_history.csv"), index=False)
    pdir = D.participant_dir(out_dir)
    D.save_id_list(os.path.join(pdir, "meta_training_ids.txt"), ctr.subject_ids)
    D.save_id_list(os.path.join(pdir, "validation_ids.txt"), cva.subject_ids)
    run = dict(workflow="mecha-train", checkpoint=os.path.basename(ckpt_path), target_atlas=target_atlas,
               source_atlas=mapping.source_atlas if mapping else target_atlas, train_phenotypes=list(train_phenotypes),
               epochs_run=len(history), best_epoch=best["epoch"], best_val_loss=best["val_loss"],
               validation_r_mean=float(np.nanmean([m["validation_r"] for m in val_metrics])),
               cohorts=[rep_tr, rep_va], preset=preset.as_dict(), seed=seed, device=str(dev),
               fingerprints=dict(meta_training=D.fingerprint(ctr.subject_ids), validation=D.fingerprint(cva.subject_ids)))
    D.write_json(os.path.join(out_dir, "train_run.json"), run)
    return run


def load_checkpoint(path: str, device: str = "auto"):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ck, dict) or ck.get("format") != CHECKPOINT_FORMAT:
        raise MechaInputError(f"{path}: not an atlas_interop MECHA checkpoint")
    if ck.get("format_version") != CHECKPOINT_VERSION:
        raise MechaInputError(f"{path}: unsupported checkpoint version {ck.get('format_version')}")
    if ck.get("vectorization") != D.VECTORIZATION:
        raise MechaInputError(f"{path}: unsupported feature vectorization {ck.get('vectorization')}")
    dev = get_device(device)
    model = MLPRegressor(MLPSpec(**ck["spec"])).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck, dev


# ---------------------------------------------------------------------------
# Adaptation and baselines
# ---------------------------------------------------------------------------
def _cv_folds(n: int) -> int:
    """Inner CV folds for small support sets (rule from the first release, commit 68ff5ba)."""
    return 2 if n <= 20 else 3 if n <= 50 else 5


def fit_ridge_cv(X, y, alphas, seed, scale: bool) -> object:
    """Ridge with alpha chosen by K-fold CV on the given (support) data only."""
    steps = [StandardScaler()] if scale else []
    pipe = make_pipeline(*steps, Ridge())
    folds = min(_cv_folds(len(y)), len(y))
    if folds < 2:
        raise MechaInputError("ridge needs at least 2 support samples")
    gs = GridSearchCV(pipe, {"ridge__alpha": list(alphas)}, scoring="neg_mean_squared_error",
                      cv=KFold(folds, shuffle=True, random_state=seed))
    gs.fit(X, y)
    return gs


def train_support_dnn(X, Y, preset: MechaPreset, seed: int, device) -> MLPRegressor:
    """Support-only DNN baseline (multi-output over held-out phenotypes).

    ASSUMPTION: trained for a fixed number of epochs with no early stopping,
    because no separate labelled validation data exist at the support-set
    level that are not support labels; test labels are never used.
    Targets are z-scored with support statistics; NaN targets are masked.
    """
    seed_everything(seed)
    mu = np.nanmean(Y, axis=0)
    sd = np.nanstd(Y, axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (Y - mu) / sd
    spec = MLPSpec(X.shape[1], list(preset.baseline_dnn_hidden), Y.shape[1], preset.baseline_dnn_dropout, "batch")
    model = MLPRegressor(spec).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=preset.baseline_dnn_lr, weight_decay=preset.baseline_dnn_weight_decay)
    rng = np.random.default_rng(seed)
    bs = min(preset.baseline_dnn_batch_size, len(Z))
    if bs < 2:
        raise MechaInputError("support-only DNN needs at least 2 support samples")
    steps = 0
    for _ in range(preset.baseline_dnn_epochs):
        model.train()
        order = rng.permutation(len(Z))
        for start in range(0, len(order) - bs + 1, bs):
            idx = order[start:start + bs]
            opt.zero_grad()
            nan_mse(model(torch.as_tensor(X[idx], device=device)), torch.as_tensor(Z[idx], device=device)).backward()
            opt.step()
            steps += 1
    model.steps_taken = steps
    model.target_mean, model.target_sd = mu, sd
    return model


def make_nested_supports(support_ids, k_list, draw_seeds, out_dir: str) -> pd.DataFrame:
    """Write nested support splits: one permutation of the support pool per draw
    seed, with each K taking the first K IDs (so K=10 is a subset of K=20, ...).

    Files: ``support_k{K}_rep{R}.txt`` where R indexes ``draw_seeds``. Returns
    a table of (k, repeat, draw_seed, fingerprint) without IDs.
    """
    ids = np.array(sorted(set(map(str, support_ids))), dtype=str)
    if len(ids) != len(support_ids):
        raise MechaInputError("support pool has duplicate subject IDs")
    k_list = sorted(int(k) for k in k_list)
    if k_list[-1] > len(ids):
        raise MechaInputError(f"largest K ({k_list[-1]}) exceeds the support pool ({len(ids)})")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for rep, s in enumerate(draw_seeds):
        perm = np.random.default_rng(int(s)).permutation(ids)
        for k in k_list:
            D.save_id_list(os.path.join(out_dir, f"support_k{k}_rep{rep}.txt"), perm[:k],
                           header=f"nested support draw: seed {s}, first {k} of one permutation")
            rows.append(dict(k=k, repeat=rep, draw_seed=int(s), support_fingerprint=D.fingerprint(perm[:k])))
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, "splits_manifest.csv"), index=False)
    return table


def load_or_draw_supports(support_ids: np.ndarray, k_list, n_repeats, seed, splits_dir: str | None,
                          available_keys: np.ndarray):
    """Support draws per (k, repeat). Explicit split files must match k exactly.

    File naming in ``splits_dir``: ``support_k{K}_rep{R}.txt`` (one subject ID per line).
    """
    draws = {}
    avail = set(available_keys.tolist())
    for k in k_list:
        if k > len(support_ids):
            raise MechaInputError(f"support size k={k} exceeds the {len(support_ids)} available support samples")
        for rep in range(n_repeats):
            if splits_dir:
                path = os.path.join(splits_dir, f"support_k{k}_rep{rep}.txt")
                ids = D.load_id_list(path)
                if len(ids) != k:
                    raise MechaInputError(f"{path}: contains {len(ids)} IDs but k={k}")
                missing = [i for i in ids if i not in avail]
                if missing:
                    raise MechaInputError(f"{path}: IDs not in the support cohort: {missing[:5]}")
            else:
                rng = np.random.default_rng([seed, k, rep])
                ids = rng.choice(support_ids, size=k, replace=False)
            draws[(k, rep)] = np.asarray(ids, dtype=str)
    return draws


def adapt_and_evaluate(*, checkpoint: str, support_conn: D.ConnectomeSet, test_conn: D.ConnectomeSet,
                       table: D.PhenotypeTable, test_phenotypes: list[str], preset: MechaPreset,
                       seed: int, out_dir: str, splits_dir: str | None = None,
                       baselines=("ridge", "dnn"), device: str = "auto",
                       meta_training_ids: str | None = None, include_mecha: bool = True) -> pd.DataFrame:
    """Adapt MECHA and/or fit support-only baselines on matched support draws.

    ``include_mecha=False`` with ``baselines`` runs the support-only models
    alone (they do not depend on the checkpoint's weights, only on its target
    atlas), so one baseline run can be shared by several MECHA checkpoints
    that use identical support/test data, draws and seeds.
    """
    if not include_mecha and not baselines:
        raise MechaInputError("nothing to run: no MECHA and no baselines")
    model, ck, dev = load_checkpoint(checkpoint, device)
    overlap = set(test_phenotypes) & set(ck["train_phenotypes"])
    if overlap:
        raise MechaInputError(f"held-out phenotypes overlap the meta-training phenotypes: {sorted(overlap)}")
    if len(set(test_phenotypes)) != len(test_phenotypes) or not test_phenotypes:
        raise MechaInputError("held-out phenotype names must be non-empty and unique")
    for c, nm in ((support_conn, "support"), (test_conn, "test")):
        D.check_atlas(c, expected_atlas=ck["target_atlas"], n_regions=ck["n_target_regions"],
                      region_ids=None if ck["target_region_ids"] is None else np.array(ck["target_region_ids"]),
                      context=f"MECHA {nm} (must be native target atlas)")
        if c.metadata.get("transformed_from"):
            raise MechaInputError(f"{c.label()}: support and test connectomes must be native target-atlas "
                                  f"data, but this file was transformed from '{c.metadata['transformed_from']}'.")
    groups = {"support": support_conn.subject_ids, "test": test_conn.subject_ids}
    if meta_training_ids:
        groups["meta-training"] = D.load_id_list(meta_training_ids)
    D.assert_disjoint_subjects(groups)

    Ys = D.phenotypes_for(support_conn, table, test_phenotypes)
    Yt = D.phenotypes_for(test_conn, table, test_phenotypes)
    Xs_edges = support_conn.edges().astype(np.float32)
    Xt_edges = test_conn.edges().astype(np.float32)
    # MECHA features: the DNN's outputs, i.e. its predictions of the meta-training
    # phenotypes (z-scored units), one column per meta-training phenotype in
    # checkpoint order. Hidden-layer activations are not used.
    Fs = predict(model, Xs_edges, dev)
    Ft = predict(model, Xt_edges, dev)
    if Fs.shape[1] != len(ck["train_phenotypes"]):
        raise MechaInputError("DNN output dimension does not match the checkpoint's meta-training phenotypes")
    test_fp = D.fingerprint(test_conn.subject_ids)
    draws = load_or_draw_supports(support_conn.subject_ids, preset.k_list, preset.n_repeats, seed,
                                  splits_dir, support_conn.subject_ids)
    pos = {s: i for i, s in enumerate(support_conn.subject_ids)}
    if len(pos) != len(support_conn.subject_ids):
        raise MechaInputError("support cohort has repeat visits; supply one sample per participant")
    rows, split_rows = [], []
    for (k, rep), ids in draws.items():
        idx = np.array([pos[s] for s in ids])
        split_rows += [dict(k=k, repeat=rep, subject_id=s) for s in ids]
        support_fp = D.fingerprint(ids)
        dnn_base = None
        if "dnn" in baselines:
            dnn_base = train_support_dnn(Xs_edges[idx], Ys[idx], preset, seed + 7919 * rep + k, dev)
            dnn_pred = predict(dnn_base, Xt_edges, dev) * dnn_base.target_sd + dnn_base.target_mean
        for j, phen in enumerate(test_phenotypes):
            ys = Ys[idx, j]
            ok = np.isfinite(ys)
            yt = Yt[:, j]
            tok = np.isfinite(yt)
            base = dict(k=k, repeat=rep, phenotype=phen, n_support_requested=int(k),
                        n_support_used=int(ok.sum()), n_test_used=int(tok.sum()),
                        support_fingerprint=support_fp, test_fingerprint=test_fp)
            if ok.sum() < 3:
                methods = (["MECHA"] if include_mecha else []) + (["Ridge"] if "ridge" in baselines else []) \
                    + (["DNN"] if "dnn" in baselines else [])
                for method in methods:
                    rows.append(dict(base, method=method, r=np.nan, p=np.nan, note="fewer than 3 labelled support samples"))
                continue
            if include_mecha:
                # one adaptation model per held-out phenotype, fit on support data only
                gs = fit_ridge_cv(Fs[idx][ok], ys[ok], preset.stack_alphas, seed, scale=False)
                r, p = pearson(yt[tok], gs.predict(Ft[tok]))
                rows.append(dict(base, method="MECHA", r=r, p=p, alpha=gs.best_params_["ridge__alpha"]))
            if "ridge" in baselines:
                gb = fit_ridge_cv(Xs_edges[idx][ok], ys[ok], preset.baseline_ridge_alphas, seed, scale=True)
                r, p = pearson(yt[tok], gb.predict(Xt_edges[tok]))
                rows.append(dict(base, method="Ridge", r=r, p=p, alpha=gb.best_params_["ridge__alpha"]))
            if dnn_base is not None:
                r, p = pearson(yt[tok], dnn_pred[tok, j])
                rows.append(dict(base, method="DNN", r=r, p=p, update_steps=dnn_base.steps_taken))
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "per_phenotype.csv"), index=False)
    runs = (df.groupby(["method", "k", "repeat"], as_index=False)
              .agg(mean_r=("r", "mean"), n_phenotypes=("r", lambda s: int(np.isfinite(s).sum()))))
    runs.to_csv(os.path.join(out_dir, "runs.csv"), index=False)
    pdir = D.participant_dir(out_dir)
    pd.DataFrame(split_rows).to_csv(os.path.join(pdir, "support_draws.csv"), index=False)
    D.save_id_list(os.path.join(pdir, "test_ids.txt"), test_conn.subject_ids)
    D.write_json(os.path.join(out_dir, "adapt_run.json"), dict(
        workflow="mecha-adapt", checkpoint=os.path.basename(checkpoint), target_atlas=ck["target_atlas"],
        source_atlas=ck["source_atlas"], train_phenotypes=ck["train_phenotypes"], test_phenotypes=list(test_phenotypes),
        k_list=list(preset.k_list), n_repeats=preset.n_repeats, explicit_splits=bool(splits_dir),
        n_support_cohort=int(support_conn.n_samples), n_test_cohort=int(test_conn.n_samples),
        test_fingerprint=test_fp, include_mecha=include_mecha,
        features="DNN phenotype predictions (one per meta-training phenotype), z-scored units",
        baselines=list(baselines), preset=preset.as_dict(), seed=seed, device=str(dev)))
    return runs


def compare_matched(per_phenotype: pd.DataFrame, baselines: pd.DataFrame | None = None) -> dict:
    """Matched comparison of MECHA against support-only Ridge and DNN.

    ``per_phenotype`` is a ``per_phenotype.csv`` table that contains MECHA
    rows (and usually the Ridge/DNN rows). If the baselines were fit in a
    separate ``--baselines-only`` run, pass that table as ``baselines``. Rows
    are matched on (k, repeat, phenotype) *and* on the support and test
    fingerprints, so pairing rows from different participants raises instead
    of being silently compared. Returns per-draw tables and a per-K summary.
    """
    key = ["k", "repeat", "phenotype"]
    fp = ["support_fingerprint", "test_fingerprint"]
    allrows = per_phenotype if baselines is None else pd.concat([per_phenotype, baselines], ignore_index=True)
    allrows = allrows[allrows["method"].isin(["MECHA", "Ridge", "DNN"])]
    missing = {"MECHA", "Ridge", "DNN"} - set(allrows["method"])
    if missing:
        raise MechaInputError(f"comparison needs MECHA, Ridge and DNN rows; missing: {sorted(missing)}")
    if allrows.duplicated(key + ["method"]).any():
        raise MechaInputError("duplicate (k, repeat, phenotype, method) rows")
    nfp = allrows.groupby(key)[fp].nunique()
    if (nfp > 1).any().any():
        raise MechaInputError("support_fingerprint or test_fingerprint differs between methods for the same "
                              "(k, repeat, phenotype); refusing to pair different participants")
    per_draw = allrows.pivot_table(index=key + fp, columns="method", values="r").reset_index()
    if per_draw[["MECHA", "Ridge", "DNN"]].isna().any().any():
        raise MechaInputError("MECHA, Ridge and DNN rows do not cover the same (k, repeat, phenotype)")
    per_draw["MECHA_minus_Ridge"] = per_draw["MECHA"] - per_draw["Ridge"]
    per_draw["MECHA_minus_DNN"] = per_draw["MECHA"] - per_draw["DNN"]
    cols = ["MECHA", "Ridge", "DNN", "MECHA_minus_Ridge", "MECHA_minus_DNN"]
    draw_means = per_draw.groupby(["k", "repeat"], as_index=False)[cols].mean()
    rows = []
    for k, g in draw_means.groupby("k", sort=True):
        row = dict(k=int(k), n_draws=int(len(g)), n_phenotypes=int(per_draw[per_draw.k == k]["phenotype"].nunique()))
        for c in ("MECHA", "Ridge", "DNN"):
            row[f"r_{c}_mean"] = float(g[c].mean())
            row[f"r_{c}_sd"] = float(g[c].std(ddof=1))
        for b in ("Ridge", "DNN"):
            d = g[f"MECHA_minus_{b}"]
            row[f"diff_vs_{b}_mean"] = float(d.mean())
            row[f"diff_vs_{b}_sd"] = float(d.std(ddof=1))
            row[f"draws_MECHA_better_than_{b}"] = int((d > 0).sum())
            row[f"wilcoxon_p_vs_{b}"] = wilcoxon_greater(fisher_z(g["MECHA"]), fisher_z(g[b]))["p"]
        rows.append(row)
    summary = pd.DataFrame(rows)
    for b in ("Ridge", "DNN"):
        summary[f"q_bh_vs_{b}"] = bh_fdr(summary[f"wilcoxon_p_vs_{b}"].to_numpy())
    return dict(per_draw=per_draw, draw_means=draw_means, summary=summary)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """Benchmark selection, MECHA gain and paired one-sided Wilcoxon (Fisher-z).

    For each k, the benchmark is the support-only model with the higher mean
    Fisher-z transformed performance across repeats. Gain = mean raw-r
    difference MECHA - benchmark over matched repeats. BH correction is applied
    across all k in ``runs``; combine runs from several atlas transfers of one
    modality into one table before calling to match the manuscript family.
    """
    group_cols = [c for c in ("modality", "source_atlas", "target_atlas") if c in runs.columns]
    out = []
    for keys, g in runs.groupby(group_cols + ["k"], sort=False) if group_cols else runs.groupby(["k"], sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols + ["k"], keys))
        per = {m: gm.set_index("repeat")["mean_r"] for m, gm in g.groupby("method")}
        if "MECHA" not in per:
            continue
        bench = [m for m in ("Ridge", "DNN") if m in per]
        for m in per:
            row[f"r_{m}_mean"] = float(per[m].mean())
        if bench:
            best = max(bench, key=lambda m: float(np.nanmean(fisher_z(per[m]))))
            j = pd.concat([per["MECHA"], per[best]], axis=1, join="inner").dropna()
            w = wilcoxon_greater(fisher_z(j.iloc[:, 0]), fisher_z(j.iloc[:, 1]))
            row.update(benchmark=best, gain=float((j.iloc[:, 0] - j.iloc[:, 1]).mean()), wilcoxon_p=w["p"],
                       n_matched_repeats=w["n_pairs"], test_note=w["note"])
        out.append(row)
    df = pd.DataFrame(out)
    if "wilcoxon_p" in df:
        df["q_bh"] = bh_fdr(df["wilcoxon_p"].to_numpy())
    return df
