"""Atlas-to-atlas mapping and signed low-rank spectral transformation.

Manuscript reference: Methods, "Connectome transformations"; Fig. 1a.

Mapping convention (format version 2)
-------------------------------------
``T`` has shape (n_target, n_source). Column ``p`` describes how source region
``p`` is distributed across target regions, so every column is non-negative
and sums to one ("source_columns_sum_to_one").

Fit
---
1. For each reference participant, Pearson-correlate every target-region time
   series with every source-region time series, min-max normalize the
   (n_target x n_source) correlation matrix across all region pairs, and set
   cost = 1 - normalized similarity.
2. Average the participant costs into one group cost matrix ``M``.
3. For every participant and time point, min-max normalize the target and
   source activity vectors, add a small positive constant, normalize each to
   sum to one, and solve the entropic OT problem with the *same* group cost
   ``M`` (Sinkhorn, regularization lambda = 0.01). The plan ``P`` has
   marginals ``P 1 = v_target`` and ``P^T 1 = v_source``.
4. Column-normalize each plan, average plans within temporal windows of each
   participant, then average across windows and participants.

Apply
-----
For each source connectome ``C_s``: eigendecompose, keep the ``k``
components with the largest |eigenvalue|, split by sign, form
``U+ = Q+ (L+)^1/2`` and ``U- = Q- |L-|^1/2``, project ``U_t = T U_s`` and
reconstruct ``C_t = U_t+ U_t+^T - U_t- U_t-^T``.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass

import numpy as np

from . import __version__
from .data import (ConnectomeSet, DataContractError, TimeSeriesSet, _ensure_parent,
                   _read_npz, _scalar_str, align_to, as_id_array, check_atlas)

MAPPING_FORMAT = "atlas_interop.mapping"
MAPPING_FORMAT_VERSION = 2
ORIENTATION = "target_by_source"
NORMALIZATION = "source_columns_sum_to_one"


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
def _pearson_cross(target_ts: np.ndarray, source_ts: np.ndarray) -> np.ndarray:
    """Pearson correlation between target rows and source rows, (n_t, n_s)."""
    a = target_ts - target_ts.mean(axis=1, keepdims=True)
    b = source_ts - source_ts.mean(axis=1, keepdims=True)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    if (na == 0).any() or (nb == 0).any():
        raise DataContractError("A regional time series is constant; Pearson correlation is undefined.")
    return (a / na[:, None]) @ (b / nb[:, None]).T


def participant_cost(target_ts: np.ndarray, source_ts: np.ndarray) -> np.ndarray:
    """One participant's cost: 1 - min-max normalized cross-atlas correlation."""
    r = _pearson_cross(target_ts, source_ts)
    lo, hi = r.min(), r.max()
    if hi == lo:
        raise DataContractError("Cross-atlas correlations are constant; cost normalization is undefined.")
    return 1.0 - (r - lo) / (hi - lo)


def group_cost(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Average participant costs. Inputs are (n_subj, n_regions, n_time)."""
    return np.mean([participant_cost(target[i], source[i]) for i in range(target.shape[0])], axis=0)


def activity_distribution(v: np.ndarray, eps: float) -> np.ndarray:
    """Min-max normalize, shift by ``eps``, normalize to sum to one.

    A constant vector becomes the uniform distribution.
    """
    lo, hi = v.min(), v.max()
    x = np.zeros_like(v, dtype=float) if hi == lo else (v - lo) / (hi - lo)
    x = x + eps
    return x / x.sum()


@dataclass
class FitSettings:
    reg: float = 0.01                 # Methods: lambda = 0.01
    window_size: int = 1              # time points per temporal window
    shift: float = 1e-5               # positive constant added after min-max (legacy value)
    sinkhorn_max_iter: int = 1_000_000  # legacy value
    sinkhorn_stop_thr: float = 1e-9   # POT default

    def as_dict(self) -> dict:
        return dict(reg=self.reg, window_size=self.window_size, shift=self.shift,
                    sinkhorn_max_iter=self.sinkhorn_max_iter,
                    sinkhorn_stop_thr=self.sinkhorn_stop_thr, solver="POT ot.sinkhorn (method='sinkhorn')")


def sinkhorn_plan(a: np.ndarray, b: np.ndarray, M: np.ndarray, s: FitSettings) -> tuple[np.ndarray, int]:
    """Entropic OT plan with row marginal ``a`` and column marginal ``b``.

    Returns the plan and the number of solver warnings raised.
    """
    import ot  # POT

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        P = ot.sinkhorn(a, b, M, s.reg, method="sinkhorn",
                        numItermax=s.sinkhorn_max_iter, stopThr=s.sinkhorn_stop_thr)
    if not np.isfinite(P).all():
        raise FloatingPointError("Sinkhorn returned non-finite values; try a larger --reg.")
    return P, len(caught)


def fit_mapping_arrays(target: np.ndarray, source: np.ndarray, settings: FitSettings | None = None,
                       progress: bool = False) -> tuple[np.ndarray, dict]:
    """Fit T (n_target, n_source) from paired arrays (n_subj, n_regions, n_time)."""
    s = settings or FitSettings()
    if target.shape[0] != source.shape[0]:
        raise DataContractError("Paired time series must have the same number of participants.")
    if target.shape[2] != source.shape[2]:
        raise DataContractError(
            f"Paired time series must have the same number of time points "
            f"(target {target.shape[2]}, source {source.shape[2]})."
        )
    if s.window_size < 1:
        raise ValueError("window_size must be >= 1")
    n_subj, n_t, n_time = target.shape
    n_s = source.shape[1]
    M = group_cost(target, source)
    subject_maps = np.zeros((n_subj, n_t, n_s))
    n_warn = 0
    iterator = range(n_subj)
    if progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="fit mapping", unit="participant")
    for i in iterator:
        window_means = []
        for start in range(0, n_time, s.window_size):
            acc = np.zeros((n_t, n_s))
            stop = min(start + s.window_size, n_time)
            for t in range(start, stop):
                a = activity_distribution(target[i, :, t], s.shift)
                b = activity_distribution(source[i, :, t], s.shift)
                P, w = sinkhorn_plan(a, b, M, s)
                n_warn += w
                acc += P / P.sum(axis=0, keepdims=True)
            window_means.append(acc / (stop - start))
        subject_maps[i] = np.mean(window_means, axis=0)
    T = subject_maps.mean(axis=0)
    info = dict(n_reference_participants=int(n_subj), n_timepoints=int(n_time),
                n_windows_per_participant=int(np.ceil(n_time / s.window_size)),
                sinkhorn_warnings=int(n_warn),
                column_sum_max_abs_error=float(np.abs(T.sum(axis=0) - 1).max()))
    return T, info


# ---------------------------------------------------------------------------
# Mapping artifact
# ---------------------------------------------------------------------------
@dataclass
class AtlasMapping:
    T: np.ndarray                     # (n_target, n_source)
    source_atlas: str
    target_atlas: str
    source_region_ids: np.ndarray | None = None
    target_region_ids: np.ndarray | None = None
    fit_info: dict | None = None
    path: str | None = None

    @property
    def n_source(self) -> int:
        return int(self.T.shape[1])

    @property
    def n_target(self) -> int:
        return int(self.T.shape[0])

    def validate(self, atol: float = 1e-6) -> None:
        if self.T.ndim != 2:
            raise DataContractError("mapping matrix must be 2-D")
        if not np.isfinite(self.T).all():
            raise DataContractError("mapping matrix has non-finite values")
        if (self.T < -atol).any():
            raise DataContractError("mapping matrix has negative weights")
        err = np.abs(self.T.sum(axis=0) - 1).max()
        if err > atol:
            raise DataContractError(
                f"mapping columns must each sum to one (source_columns_sum_to_one); max error {err:.2e}"
            )


def fit_mapping(source: TimeSeriesSet, target: TimeSeriesSet, *, source_atlas: str, target_atlas: str,
                settings: FitSettings | None = None, progress: bool = False) -> tuple[AtlasMapping, np.ndarray]:
    """Fit from paired reference time series aligned by sample key.

    Both files must contain exactly the same participants/sessions. Returns the
    mapping and the aligned reference subject IDs (participant-level; keep local).
    """
    if source.atlas is not None and source.atlas != source_atlas:
        raise DataContractError(f"{source.label()}: atlas '{source.atlas}' != --source-atlas '{source_atlas}'")
    if target.atlas is not None and target.atlas != target_atlas:
        raise DataContractError(f"{target.label()}: atlas '{target.atlas}' != --target-atlas '{target_atlas}'")
    if (source.session_ids is None) != (target.session_ids is None):
        raise DataContractError("Both time-series files must either have session_ids or not.")
    idx = align_to(source.keys, target.keys, ref_name=source.label(), other_name=target.label(),
                   allow_extra=False)
    tgt = target.take(idx)
    T, info = fit_mapping_arrays(tgt.mats, source.mats, settings, progress=progress)
    s = settings or FitSettings()
    info.update(settings=s.as_dict(), n_reference_unique_subjects=int(len(set(source.subject_ids))))
    mapping = AtlasMapping(T, source_atlas, target_atlas, source.region_ids, target.region_ids, info)
    mapping.validate()
    return mapping, np.asarray(source.subject_ids)


def save_mapping(path: str, m: AtlasMapping) -> str:
    """Save a version-2 mapping. Contains no participant-level information."""
    m.validate()
    _ensure_parent(path)
    payload = dict(
        format=np.array(MAPPING_FORMAT), format_version=np.array(MAPPING_FORMAT_VERSION),
        mapping=m.T.astype(np.float64), orientation=np.array(ORIENTATION),
        normalization=np.array(NORMALIZATION), source_atlas=np.array(m.source_atlas),
        target_atlas=np.array(m.target_atlas), n_source=np.array(m.n_source),
        n_target=np.array(m.n_target), toolkit_version=np.array(__version__),
        fit_info_json=np.array(json.dumps(m.fit_info or {}, sort_keys=True)),
    )
    if m.source_region_ids is not None:
        payload["source_region_ids"] = np.asarray(m.source_region_ids, dtype=str)
    if m.target_region_ids is not None:
        payload["target_region_ids"] = np.asarray(m.target_region_ids, dtype=str)
    np.savez_compressed(path, **payload)
    return path


def load_mapping(path: str) -> AtlasMapping:
    d = _read_npz(path)
    if "format" not in d:
        if "mapping_matrix" in d:
            raise DataContractError(f"{path}: {LEGACY_EXPLANATION}")
        raise DataContractError(f"{path}: not an atlas_interop mapping file (no 'format' key).")
    if _scalar_str(d["format"]) != MAPPING_FORMAT:
        raise DataContractError(f"{path}: unexpected format '{_scalar_str(d['format'])}'.")
    version = int(d["format_version"])
    if version != MAPPING_FORMAT_VERSION:
        raise DataContractError(f"{path}: unsupported mapping format_version {version}.")
    if _scalar_str(d["orientation"]) != ORIENTATION or _scalar_str(d["normalization"]) != NORMALIZATION:
        raise DataContractError(
            f"{path}: orientation/normalization must be {ORIENTATION}/{NORMALIZATION}; got "
            f"{_scalar_str(d['orientation'])}/{_scalar_str(d['normalization'])}."
        )
    T = np.asarray(d["mapping"], dtype=np.float64)
    if T.shape != (int(d["n_target"]), int(d["n_source"])):
        raise DataContractError(f"{path}: mapping shape {T.shape} disagrees with n_target/n_source.")
    m = AtlasMapping(
        T=T, source_atlas=_scalar_str(d["source_atlas"]), target_atlas=_scalar_str(d["target_atlas"]),
        source_region_ids=as_id_array(d["source_region_ids"], "source_region_ids", path) if "source_region_ids" in d else None,
        target_region_ids=as_id_array(d["target_region_ids"], "target_region_ids", path) if "target_region_ids" in d else None,
        fit_info=json.loads(_scalar_str(d["fit_info_json"])), path=path,
    )
    m.validate()
    return m


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------
def signed_embedding(C: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (U+, U-) for the top-k components ranked by |eigenvalue|."""
    n = C.shape[0]
    if not 1 <= k <= n:
        raise ValueError(f"rank k={k} must be between 1 and the number of source regions ({n}).")
    w, Q = np.linalg.eigh((C + C.T) / 2.0)
    order = np.argsort(-np.abs(w), kind="stable")[:k]
    wk, Qk = w[order], Q[:, order]
    pos, neg = wk > 0, wk < 0
    U_pos = Qk[:, pos] * np.sqrt(wk[pos])
    U_neg = Qk[:, neg] * np.sqrt(np.abs(wk[neg]))
    return U_pos, U_neg


def transform_connectome(C: np.ndarray, T: np.ndarray, k: int) -> tuple[np.ndarray, int, int]:
    U_pos, U_neg = signed_embedding(C, k)
    Ut_pos, Ut_neg = T @ U_pos, T @ U_neg
    C_t = Ut_pos @ Ut_pos.T - Ut_neg @ Ut_neg.T
    return (C_t + C_t.T) / 2.0, U_pos.shape[1], U_neg.shape[1]


def apply_mapping(conn: ConnectomeSet, mapping: AtlasMapping, k: int, modality: str | None = None,
                  check_source_atlas: bool = True) -> tuple[ConnectomeSet, dict]:
    """Transform source-atlas connectomes into the mapping's target atlas.

    Requires only the mapping and the source connectomes. Output sample order
    and IDs equal the input order and IDs.
    """
    note = check_atlas(conn, expected_atlas=mapping.source_atlas if check_source_atlas else None,
                       n_regions=mapping.n_source, region_ids=mapping.source_region_ids,
                       context="apply mapping (source)")
    if k > mapping.n_source:
        raise ValueError(f"rank k={k} exceeds the number of source regions ({mapping.n_source}).")
    n = conn.n_samples
    out = np.zeros((mapping.n_target, mapping.n_target, n))
    n_pos = np.zeros(n, dtype=int)
    n_neg = np.zeros(n, dtype=int)
    for i in range(n):
        out[:, :, i], n_pos[i], n_neg[i] = transform_connectome(conn.mats[:, :, i], mapping.T, k)
    meta = dict(conn.metadata)
    meta.update(transformed_from=mapping.source_atlas, mapping_file=os.path.basename(mapping.path or ""),
                spectral_rank=int(k), toolkit_version=__version__)
    if modality:
        meta["modality"] = modality
    mapped = ConnectomeSet(out, conn.subject_ids.copy(),
                           None if conn.session_ids is None else conn.session_ids.copy(),
                           mapping.target_atlas, mapping.target_region_ids, meta)
    report = dict(n_samples=int(n), spectral_rank=int(k), source_region_check=note,
                  positive_components_min=int(n_pos.min()), positive_components_max=int(n_pos.max()),
                  negative_components_min=int(n_neg.min()), negative_components_max=int(n_neg.max()),
                  samples_with_zero_eigenvalues_in_top_k=int(((n_pos + n_neg) < k).sum()))
    return mapped, report


def reconstruction_similarity(mapped: ConnectomeSet, native: ConnectomeSet) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample Pearson r between vectorized mapped and native target connectomes.

    Samples are matched by key; every mapped sample must be present in ``native``.
    Returns (keys, r) in the order of ``mapped``.
    """
    if mapped.n_regions != native.n_regions:
        raise DataContractError(
            f"mapped connectomes have {mapped.n_regions} regions but native target has {native.n_regions}."
        )
    if mapped.atlas and native.atlas and mapped.atlas != native.atlas:
        raise DataContractError(f"atlas mismatch: mapped '{mapped.atlas}' vs native '{native.atlas}'.")
    if (mapped.session_ids is None) != (native.session_ids is None):
        raise DataContractError("mapped and native files must both have or both lack session_ids.")
    idx = align_to(mapped.keys, native.keys, ref_name=mapped.label(), other_name=native.label())
    a = mapped.edges()
    b = native.take(idx).edges()
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    r = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return mapped.keys, r
