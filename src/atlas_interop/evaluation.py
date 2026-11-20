"""Reconstruction similarity and native vs cross-atlas ridge prediction.

Manuscript reference: Methods, "Evaluation of reconstruction similarity and
predictive utility after connectome transformation"; Fig. 2; Supplementary
Figs. 3-5.

Prediction protocol (per atlas pair and phenotype)
--------------------------------------------------
* Participants with the phenotype and both connectomes are eligible.
* A random tuning subset (20%) is drawn once. Within it: univariate feature
  selection (Pearson p < 0.1 against the phenotype), standardization, and a
  3-fold CV choice of the ridge penalty over 10^-1 .. 10^3. The selected
  penalty is fixed for evaluation. Tuning participants are then excluded.
* Remaining participants: 10-fold CV repeated 10 times. In each fold, feature
  selection and standardization are re-fit on the training fold only, the
  ridge (fixed penalty) is fit on the training fold's *transformed*
  connectomes (or native, for the diagonal), and predictions are made from the
  held-out participants' *native target-atlas* connectomes.
* Performance per repeat = Pearson r between the pooled held-out predictions
  and observations across the 10 folds (ASSUMPTION: pooled rather than
  fold-averaged; exposed as ``--score pooled|fold_mean``).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from . import data as D
from .presets import PredictionPreset
from .stats import pearson, spearman
from .transform import reconstruction_similarity


def univariate_select(X: np.ndarray, y: np.ndarray, p_threshold: float) -> np.ndarray:
    """Boolean mask of features with Pearson p < threshold (two-sided)."""
    n = len(y)
    Xc = X - X.mean(0)
    yc = y - y.mean()
    denom = np.sqrt((Xc ** 2).sum(0) * (yc ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (Xc * yc[:, None]).sum(0) / denom
        t = r * np.sqrt((n - 2) / np.clip(1 - r ** 2, 1e-15, None))
    p = 2 * sps.t.sf(np.abs(t), df=n - 2)
    p = np.where(np.isfinite(p), p, 1.0)
    return p < p_threshold


class SelectScaleRidge:
    """Feature selection -> standardization -> ridge, fit on training data only."""

    def __init__(self, alpha: float, p_threshold: float):
        self.alpha, self.p_threshold = alpha, p_threshold

    def fit(self, X, y):
        self.mask_ = univariate_select(X, y, self.p_threshold)
        if not self.mask_.any():
            self.mask_ = None
            self.const_ = float(np.mean(y))
            return self
        Xs = X[:, self.mask_]
        self.mu_ = Xs.mean(0)
        self.sd_ = Xs.std(0)
        self.sd_[self.sd_ == 0] = 1.0
        self.ridge_ = Ridge(alpha=self.alpha).fit((Xs - self.mu_) / self.sd_, y)
        return self

    def predict(self, X):
        if self.mask_ is None:
            return np.full(len(X), self.const_)
        return self.ridge_.predict((X[:, self.mask_] - self.mu_) / self.sd_)

    @property
    def n_selected(self) -> int:
        return 0 if self.mask_ is None else int(self.mask_.sum())


def tune_alpha(X, y, preset: PredictionPreset, seed: int) -> tuple[float, dict]:
    kf = KFold(preset.inner_folds, shuffle=True, random_state=seed)
    scores = {}
    for a in preset.alphas:
        mse = []
        for tr, va in kf.split(X):
            m = SelectScaleRidge(a, preset.p_threshold).fit(X[tr], y[tr])
            mse.append(np.mean((m.predict(X[va]) - y[va]) ** 2))
        scores[a] = float(np.mean(mse))
    best = min(scores, key=scores.get)
    return float(best), scores


def predict_pair(X_train_space: np.ndarray, X_native: np.ndarray, y: np.ndarray, subject_ids: np.ndarray,
                 preset: PredictionPreset, seed: int, score: str = "pooled") -> tuple[list[dict], dict]:
    """Run the tuning + repeated CV protocol for one atlas pair and phenotype.

    ``X_train_space``: features used for model fitting (transformed source
    connectomes, or native target for the diagonal). ``X_native``: native
    target-atlas features used for held-out predictions. Rows are aligned.
    """
    n = len(y)
    rng = np.random.default_rng(seed)
    subjects = np.array(sorted(set(subject_ids.tolist())), dtype=str)
    n_tune = int(round(preset.tuning_fraction * len(subjects)))
    tune_subj = set(rng.choice(subjects, size=n_tune, replace=False).tolist())
    is_tune = np.array([s in tune_subj for s in subject_ids])
    tune_idx, eval_idx = np.flatnonzero(is_tune), np.flatnonzero(~is_tune)
    alpha, alpha_scores = tune_alpha(X_train_space[tune_idx], y[tune_idx], preset, seed)
    rows = []
    for rep in range(preset.n_repeats):
        kf = KFold(preset.outer_folds, shuffle=True, random_state=seed + 1 + rep)
        pred = np.full(n, np.nan)
        fold_r, n_sel = [], []
        for tr, te in kf.split(eval_idx):
            tr, te = eval_idx[tr], eval_idx[te]
            m = SelectScaleRidge(alpha, preset.p_threshold).fit(X_train_space[tr], y[tr])
            pred[te] = m.predict(X_native[te])
            fold_r.append(pearson(y[te], pred[te])[0])
            n_sel.append(m.n_selected)
        if score == "pooled":
            r, p = pearson(y[eval_idx], pred[eval_idx])
        else:
            r, p = float(np.nanmean(fold_r)), float("nan")
        rows.append(dict(repeat=rep, r=r, p=p, alpha=alpha, n_eval=int(len(eval_idx)),
                         mean_features_selected=float(np.mean(n_sel))))
    info = dict(alpha=alpha, alpha_cv_mse=alpha_scores, n_tune=int(len(tune_idx)), n_eval=int(len(eval_idx)),
                tune_subjects=sorted(tune_subj))
    return rows, info


def run_evaluation(*, native: dict[str, D.ConnectomeSet], mapped: dict[tuple[str, str], D.ConnectomeSet],
                   table: D.PhenotypeTable | None, phenotypes: list[str], preset: PredictionPreset, seed: int,
                   out_dir: str, score: str = "pooled", modality: str = "") -> dict:
    """Reconstruction similarity for every mapped pair; ridge prediction for every
    (source, target) pair including the native diagonal.

    ``native[atlas]``: native connectomes of the evaluation cohort.
    ``mapped[(source, target)]``: the same cohort's source connectomes
    transformed into ``target``.
    """
    os.makedirs(out_dir, exist_ok=True)
    pdir = D.participant_dir(out_dir)
    recon_rows, recon_part = [], []
    for (src, tgt), m in mapped.items():
        if tgt not in native:
            raise D.DataContractError(f"native '{tgt}' connectomes required to evaluate {src}->{tgt}")
        keys, r = reconstruction_similarity(m, native[tgt])
        n_s, n_t = native[src].n_regions if src in native else None, native[tgt].n_regions
        recon_rows.append(dict(modality=modality, source_atlas=src, target_atlas=tgt, n_source=n_s, n_target=n_t,
                               log_ratio=np.log(n_t / n_s) if n_s else np.nan,
                               r_mean=float(np.mean(r)), r_sd=float(np.std(r, ddof=1)) if len(r) > 1 else np.nan,
                               r_median=float(np.median(r)), n=int(len(r))))
        recon_part += [dict(source_atlas=src, target_atlas=tgt, sample=D.describe_key(k), r=float(v))
                       for k, v in zip(keys, r)]
    recon = pd.DataFrame(recon_rows)
    recon.to_csv(os.path.join(out_dir, "reconstruction_summary.csv"), index=False)
    pd.DataFrame(recon_part).to_csv(os.path.join(pdir, "reconstruction_per_participant.csv"), index=False)

    pred_rows, tuning = [], []
    if table is not None and phenotypes:
        pairs = [(a, a) for a in native] + list(mapped)
        for phen in phenotypes:
            for src, tgt in pairs:
                nat = native[tgt]
                train_space = nat if src == tgt else mapped[(src, tgt)]
                if (train_space.session_ids is None) != (nat.session_ids is None):
                    raise D.DataContractError("session ID structure differs between mapped and native files")
                idx = D.align_to(train_space.keys, nat.keys, ref_name=train_space.label(), other_name=nat.label())
                nat_al = nat.take(idx)
                y = D.phenotypes_for(train_space, table, [phen])[:, 0]
                ok = np.isfinite(y)
                rows, info = predict_pair(train_space.edges()[ok], nat_al.edges()[ok], y[ok],
                                          train_space.subject_ids[ok], preset, seed, score)
                for r in rows:
                    r.update(modality=modality, phenotype=phen, source_atlas=src, target_atlas=tgt,
                             native=src == tgt, n_missing_label=int((~ok).sum()), seed=seed)
                pred_rows += rows
                tuning.append(dict(phenotype=phen, source_atlas=src, target_atlas=tgt, alpha=info["alpha"],
                                   n_tune=info["n_tune"], n_eval=info["n_eval"]))
                D.save_id_list(os.path.join(pdir, f"tuning_ids_{phen}_{src}_to_{tgt}.txt"), info["tune_subjects"])
    pred = pd.DataFrame(pred_rows)
    pred.to_csv(os.path.join(out_dir, "prediction_runs.csv"), index=False)
    pd.DataFrame(tuning).to_csv(os.path.join(out_dir, "prediction_tuning.csv"), index=False)
    analysis = analysis_summaries(recon, pred)
    D.write_json(os.path.join(out_dir, "analysis_summary.json"), analysis)
    D.write_json(os.path.join(out_dir, "evaluation_run.json"), dict(
        workflow="evaluate", preset=preset.as_dict(), seed=seed, score=score, modality=modality,
        native_atlases=list(native), mapped_pairs=[f"{s}->{t}" for s, t in mapped], phenotypes=phenotypes))
    return dict(reconstruction=recon, prediction=pred, analysis=analysis)


def analysis_summaries(recon: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """Resolution-reconstruction (Supp. Fig. 4) and reconstruction-prediction
    (Supp. Fig. 5) associations, as Spearman correlations over directed pairs."""
    out = {}
    if len(recon) and recon["log_ratio"].notna().any():
        rho, p = spearman(recon["log_ratio"], recon["r_mean"])
        out["resolution_vs_reconstruction"] = dict(spearman_rho=rho, p=p, n_pairs=int(len(recon)),
                                                   note="Each point is one directed source->target pair.")
    if len(pred):
        cross = pred[~pred["native"]]
        per_pair = cross.groupby(["phenotype", "source_atlas", "target_atlas"], as_index=False)["r"].mean()
        merged = per_pair.merge(recon[["source_atlas", "target_atlas", "r_mean"]], on=["source_atlas", "target_atlas"])
        out["prediction_summary"] = (pred.groupby(["phenotype", "source_atlas", "target_atlas"], as_index=False)
                                     .agg(r_mean=("r", "mean"), r_sd=("r", "std"), n_repeats=("r", "size"))
                                     .to_dict(orient="records"))
        out["reconstruction_vs_prediction"] = {}
        for phen, g in merged.groupby("phenotype"):
            rho, p = spearman(g["r_mean"], g["r"])
            out["reconstruction_vs_prediction"][phen] = dict(spearman_rho=rho, p=p, n_pairs=int(len(g)))
    return out
