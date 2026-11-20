"""Statistical helpers used by the comparison summaries."""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

R_CLIP = 1 - 1e-12


def fisher_z(r) -> np.ndarray:
    return np.arctanh(np.clip(np.asarray(r, dtype=float), -R_CLIP, R_CLIP))


def fisher_mean_r(r) -> float:
    """Average correlations in Fisher-z space and back-transform."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    return float(np.tanh(fisher_z(r).mean())) if r.size else float("nan")


def pearson(y_true, y_pred) -> tuple[float, float]:
    """Pearson r and two-sided p. Returns (nan, nan) when undefined."""
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    if y_true.size < 3 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan"), float("nan")
    r, p = sps.pearsonr(y_true, y_pred)
    return float(r), float(p)


def wilcoxon_greater(a, b) -> dict:
    """One-sided paired Wilcoxon signed-rank test of a > b.

    Returns the statistic, p value, the number of pairs, and a note when the
    test is not informative (fewer than 2 non-zero differences).
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    res = dict(n_pairs=int(ok.sum()), n_nonzero=int((d != 0).sum()), statistic=float("nan"),
               p=float("nan"), note="")
    if res["n_nonzero"] < 2:
        res["note"] = "too few non-zero paired differences for a Wilcoxon test"
        return res
    w = sps.wilcoxon(a[ok], b[ok], alternative="greater", zero_method="wilcox")
    res.update(statistic=float(w.statistic), p=float(w.pvalue))
    if res["n_pairs"] < 6:
        res["note"] = (f"only {res['n_pairs']} paired runs: the smallest attainable one-sided p is "
                       f"{0.5 ** res['n_pairs']:.3g}; not suitable for inference")
    return res


def bh_fdr(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p values (q). NaNs are left as NaN."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def spearman(x, y) -> tuple[float, float]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    r = sps.spearmanr(x[ok], y[ok])
    return float(r.statistic), float(r.pvalue)
