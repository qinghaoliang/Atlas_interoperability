"""Public data contract: file formats, ID handling, alignment and cohort checks.

File formats
------------
Connectomes (``.npz``)
    ``mats``          float array, shape (n_regions, n_regions, n_samples)
    ``subject_ids``   string array, shape (n_samples,)
    ``session_ids``   optional string array, shape (n_samples,)   (repeat visits)
    ``atlas``         optional string scalar
    ``region_ids``    optional string array, shape (n_regions,)   (region order)
    ``metadata_json`` optional JSON string (e.g. modality, provenance)

Regional time series (``.npz``)
    ``mats``          float array, shape (n_samples, n_regions, n_timepoints)
    ``subject_ids``, ``session_ids``, ``atlas``, ``region_ids`` as above.

Phenotype tables (``.csv`` or ``.tsv``)
    A ``subject_id`` column, an optional ``session_id`` column, and one named
    numeric column per phenotype. Empty cells and NA/NaN mean "missing".

ID lists (``.txt``)
    One ID per line; blank lines and lines starting with ``#`` are ignored.

Rules
-----
* IDs are opaque strings. They are never parsed and never converted to
  numbers (leading zeros are preserved). Numeric ID arrays are rejected.
* A sample key is the subject ID, or (subject ID, session ID) when sessions
  are present. Joins use sample keys; cohort separation uses subject IDs.
* Inputs are aligned by key, never by position. Duplicate keys are rejected.
  Missing keys are rejected wherever exact pairing is required.
* ``.npz`` files are loaded with ``allow_pickle=False``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

KEY_SEP = "\x1f"  # unit separator: cannot appear in normal ID strings
MISSING_TOKENS = {"", "na", "nan", "n/a", "null", "none"}
VECTORIZATION = "upper_triangle_k1_row_major"


class DataContractError(ValueError):
    """Raised when inputs violate the public data contract."""


# ---------------------------------------------------------------------------
# IDs and keys
# ---------------------------------------------------------------------------
def _preview(items: Sequence, n: int = 5) -> str:
    items = list(items)
    head = ", ".join(repr(str(x)) for x in items[:n])
    return head + (f", ... ({len(items)} total)" if len(items) > n else "")


def as_id_array(values, name: str, source: str = "input") -> np.ndarray:
    """Validate and return a 1-D array of non-empty string IDs."""
    arr = np.asarray(values)
    if arr.dtype.kind == "S":
        arr = np.char.decode(arr, "utf-8")
    if arr.dtype.kind == "O":
        if not all(isinstance(x, str) for x in arr.ravel()):
            raise DataContractError(f"{source}: '{name}' must contain only strings.")
        arr = arr.astype(str)
    if arr.dtype.kind != "U":
        raise DataContractError(
            f"{source}: '{name}' must be stored as strings (got dtype {arr.dtype}). "
            "Numeric IDs can lose leading zeros; save IDs with np.asarray(ids, dtype=str)."
        )
    if arr.ndim != 1:
        raise DataContractError(f"{source}: '{name}' must be 1-D (got shape {arr.shape}).")
    stripped = np.char.strip(arr)
    if (stripped == "").any():
        raise DataContractError(f"{source}: '{name}' contains empty IDs.")
    if (stripped != arr).any():
        raise DataContractError(f"{source}: '{name}' contains IDs with leading/trailing whitespace.")
    return arr.astype(str)


def make_keys(subject_ids: np.ndarray, session_ids: np.ndarray | None) -> np.ndarray:
    if session_ids is None:
        return np.asarray(subject_ids, dtype=str)
    return np.array([f"{s}{KEY_SEP}{v}" for s, v in zip(subject_ids, session_ids)], dtype=str)


def describe_key(key: str) -> str:
    return key.replace(KEY_SEP, " / session ")


def find_duplicates(keys: Iterable[str]) -> list[str]:
    seen, dup = set(), []
    for k in keys:
        if k in seen and k not in dup:
            dup.append(k)
        seen.add(k)
    return dup


def require_unique(keys: Sequence[str], what: str) -> None:
    dup = find_duplicates(keys)
    if dup:
        raise DataContractError(
            f"{what}: duplicate sample keys: {_preview([describe_key(d) for d in dup])}. "
            "If these are repeat visits, supply session IDs."
        )


def align_to(ref_keys: Sequence[str], other_keys: Sequence[str], *, ref_name: str,
             other_name: str, allow_extra: bool = True) -> np.ndarray:
    """Return ``idx`` such that ``other_keys[idx]`` equals ``ref_keys`` element-wise.

    Raises if either side has duplicates, if any reference key is missing from
    ``other``, or (``allow_extra=False``) if ``other`` has keys not in ``ref``.
    """
    ref_keys = list(ref_keys)
    other_keys = list(other_keys)
    require_unique(ref_keys, ref_name)
    require_unique(other_keys, other_name)
    pos = {k: i for i, k in enumerate(other_keys)}
    missing = [k for k in ref_keys if k not in pos]
    if missing:
        raise DataContractError(
            f"{len(missing)} sample(s) in {ref_name} are missing from {other_name}: "
            f"{_preview([describe_key(m) for m in missing])}."
        )
    if not allow_extra and len(other_keys) != len(ref_keys):
        ref_set = set(ref_keys)
        extra = [k for k in other_keys if k not in ref_set]
        raise DataContractError(
            f"{len(extra)} sample(s) in {other_name} are not in {ref_name}: "
            f"{_preview([describe_key(e) for e in extra])}. Exact pairing is required."
        )
    idx = np.array([pos[k] for k in ref_keys], dtype=np.int64)
    if not np.array_equal(np.asarray(other_keys, dtype=str)[idx], np.asarray(ref_keys, dtype=str)):
        raise AssertionError("internal alignment check failed")
    return idx


def assert_disjoint_subjects(groups: Mapping[str, Iterable[str]]) -> None:
    """Raise if any subject ID appears in more than one named group."""
    names = list(groups)
    sets = {n: set(map(str, groups[n])) for n in names}
    problems = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = sorted(sets[a] & sets[b])
            if shared:
                problems.append(f"{a} and {b} share {len(shared)} subject(s): {_preview(shared)}")
    if problems:
        raise DataContractError("Prohibited participant overlap. " + "; ".join(problems))


def load_id_list(path: str) -> np.ndarray:
    with open(path, encoding="utf-8") as fh:
        ids = [line.strip() for line in fh]
    ids = [i for i in ids if i and not i.startswith("#")]
    arr = as_id_array(np.array(ids, dtype=str), "ids", path) if ids else np.array([], dtype=str)
    require_unique(arr, path)
    return arr


def fingerprint(ids: Iterable[str], n_hex: int = 16) -> str:
    """Order-insensitive fingerprint of a set of IDs (SHA-256 of the sorted IDs).

    Two splits have the same fingerprint exactly when they contain the same
    IDs, so fingerprints let outputs prove that methods saw the same
    participants without writing the IDs themselves.
    """
    import hashlib
    payload = "\n".join(sorted(str(i) for i in ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:n_hex]


def save_id_list(path: str, ids: Iterable[str], header: str | None = None) -> str:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as fh:
        if header:
            for line in header.splitlines():
                fh.write(f"# {line}\n")
        for i in ids:
            fh.write(f"{i}\n")
    return path


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------
@dataclass
class SampleSet:
    """Common fields for connectome and time-series containers."""

    mats: np.ndarray
    subject_ids: np.ndarray
    session_ids: np.ndarray | None = None
    atlas: str | None = None
    region_ids: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)
    path: str | None = None

    @property
    def keys(self) -> np.ndarray:
        return make_keys(self.subject_ids, self.session_ids)

    @property
    def n_samples(self) -> int:
        return len(self.subject_ids)

    def label(self) -> str:
        return self.path or f"<{type(self).__name__} atlas={self.atlas}>"


@dataclass
class ConnectomeSet(SampleSet):
    """Connectomes with shape (n_regions, n_regions, n_samples)."""

    @property
    def n_regions(self) -> int:
        return int(self.mats.shape[0])

    def take(self, idx: np.ndarray) -> "ConnectomeSet":
        idx = np.asarray(idx, dtype=np.int64)
        return ConnectomeSet(
            mats=self.mats[:, :, idx], subject_ids=self.subject_ids[idx],
            session_ids=None if self.session_ids is None else self.session_ids[idx],
            atlas=self.atlas, region_ids=self.region_ids, metadata=dict(self.metadata),
            path=self.path,
        )

    def edges(self) -> np.ndarray:
        """Vectorized upper triangle (k=1, row-major), shape (n_samples, n_edges)."""
        return vectorize(self.mats)


@dataclass
class TimeSeriesSet(SampleSet):
    """Regional time series with shape (n_samples, n_regions, n_timepoints)."""

    @property
    def n_regions(self) -> int:
        return int(self.mats.shape[1])

    @property
    def n_timepoints(self) -> int:
        return int(self.mats.shape[2])

    def take(self, idx: np.ndarray) -> "TimeSeriesSet":
        idx = np.asarray(idx, dtype=np.int64)
        return TimeSeriesSet(
            mats=self.mats[idx], subject_ids=self.subject_ids[idx],
            session_ids=None if self.session_ids is None else self.session_ids[idx],
            atlas=self.atlas, region_ids=self.region_ids, metadata=dict(self.metadata),
            path=self.path,
        )


def vectorize(mats: np.ndarray) -> np.ndarray:
    n = mats.shape[0]
    iu = np.triu_indices(n, 1)
    return np.ascontiguousarray(mats[iu[0], iu[1], :].T)


def n_edges(n_regions: int) -> int:
    return n_regions * (n_regions - 1) // 2


# ---------------------------------------------------------------------------
# NPZ I/O
# ---------------------------------------------------------------------------
def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


def _read_npz(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _scalar_str(arr) -> str | None:
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.ndim != 0:
        raise DataContractError(f"expected a string scalar, got shape {a.shape}")
    return str(a)


def _load_samples(path: str, kind: str):
    d = _read_npz(path)
    if "mats" not in d:
        raise DataContractError(f"{path}: missing required key 'mats'.")
    if "subject_ids" not in d:
        raise DataContractError(
            f"{path}: missing required key 'subject_ids'. Positional (unlabelled) arrays "
            "are not accepted; add a string array of subject IDs."
        )
    mats = np.asarray(d["mats"], dtype=np.float64)
    subject_ids = as_id_array(d["subject_ids"], "subject_ids", path)
    session_ids = as_id_array(d["session_ids"], "session_ids", path) if "session_ids" in d else None
    if session_ids is not None and len(session_ids) != len(subject_ids):
        raise DataContractError(f"{path}: session_ids and subject_ids differ in length.")
    region_ids = as_id_array(d["region_ids"], "region_ids", path) if "region_ids" in d else None
    meta = json.loads(_scalar_str(d["metadata_json"])) if "metadata_json" in d else {}
    atlas = _scalar_str(d.get("atlas"))
    return mats, subject_ids, session_ids, atlas, region_ids, meta


def _check_finite(mats: np.ndarray, sample_axis: int, keys: np.ndarray, path: str) -> None:
    other = tuple(a for a in range(mats.ndim) if a != sample_axis)
    bad = ~np.isfinite(mats).all(axis=other)
    if bad.any():
        raise DataContractError(
            f"{path}: non-finite values for {int(bad.sum())} sample(s): "
            f"{_preview([describe_key(k) for k in keys[bad]])}. Remove or repair them explicitly."
        )


def load_connectomes(path: str, *, require_symmetric: bool = True, atol: float = 1e-6) -> ConnectomeSet:
    mats, sid, ses, atlas, rid, meta = _load_samples(path, "connectomes")
    if mats.ndim != 3 or mats.shape[0] != mats.shape[1]:
        raise DataContractError(
            f"{path}: connectome 'mats' must have shape (n_regions, n_regions, n_samples); "
            f"got {mats.shape}. (Time series use (n_samples, n_regions, n_timepoints).)"
        )
    if mats.shape[2] != len(sid):
        raise DataContractError(
            f"{path}: mats has {mats.shape[2]} samples on axis 2 but {len(sid)} subject_ids."
        )
    if rid is not None and len(rid) != mats.shape[0]:
        raise DataContractError(f"{path}: region_ids length {len(rid)} != n_regions {mats.shape[0]}.")
    if rid is not None:
        require_unique(rid, f"{path} region_ids")
    keys = make_keys(sid, ses)
    require_unique(keys, path)
    _check_finite(mats, 2, keys, path)
    if require_symmetric:
        asym = np.abs(mats - mats.transpose(1, 0, 2)).max(axis=(0, 1))
        bad = asym > atol * np.maximum(1.0, np.abs(mats).max(axis=(0, 1)))
        if bad.any():
            raise DataContractError(
                f"{path}: {int(bad.sum())} connectome(s) are not symmetric (e.g. "
                f"{_preview([describe_key(k) for k in keys[bad]])})."
            )
    return ConnectomeSet(mats, sid, ses, atlas, rid, meta, path)


def load_timeseries(path: str) -> TimeSeriesSet:
    mats, sid, ses, atlas, rid, meta = _load_samples(path, "time series")
    if mats.ndim != 3:
        raise DataContractError(
            f"{path}: time-series 'mats' must have shape (n_samples, n_regions, n_timepoints); "
            f"got {mats.shape}."
        )
    if mats.shape[0] != len(sid):
        raise DataContractError(
            f"{path}: mats has {mats.shape[0]} samples on axis 0 but {len(sid)} subject_ids. "
            "Time series use (n_samples, n_regions, n_timepoints)."
        )
    if rid is not None and len(rid) != mats.shape[1]:
        raise DataContractError(f"{path}: region_ids length {len(rid)} != n_regions {mats.shape[1]}.")
    keys = make_keys(sid, ses)
    require_unique(keys, path)
    _check_finite(mats, 0, keys, path)
    return TimeSeriesSet(mats, sid, ses, atlas, rid, meta, path)


def _save_samples(path: str, s: SampleSet) -> str:
    _ensure_parent(path)
    payload = {"mats": s.mats, "subject_ids": np.asarray(s.subject_ids, dtype=str)}
    if s.session_ids is not None:
        payload["session_ids"] = np.asarray(s.session_ids, dtype=str)
    if s.atlas is not None:
        payload["atlas"] = np.array(s.atlas)
    if s.region_ids is not None:
        payload["region_ids"] = np.asarray(s.region_ids, dtype=str)
    if s.metadata:
        payload["metadata_json"] = np.array(json.dumps(s.metadata, sort_keys=True))
    np.savez_compressed(path, **payload)
    return path


def save_connectomes(path: str, cs: ConnectomeSet) -> str:
    return _save_samples(path, cs)


def save_timeseries(path: str, ts: TimeSeriesSet) -> str:
    return _save_samples(path, ts)


# ---------------------------------------------------------------------------
# Phenotype tables
# ---------------------------------------------------------------------------
@dataclass
class PhenotypeTable:
    subject_ids: np.ndarray
    session_ids: np.ndarray | None
    values: pd.DataFrame  # float columns, index = row order
    path: str | None = None

    @property
    def keys(self) -> np.ndarray:
        return make_keys(self.subject_ids, self.session_ids)


def load_phenotypes(path: str, columns: Sequence[str] | None = None) -> PhenotypeTable:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Phenotype table not found: {path}")
    sep = "\t" if path.lower().endswith((".tsv", ".tab")) else ","
    df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    if "subject_id" not in df.columns:
        raise DataContractError(f"{path}: phenotype table needs a 'subject_id' column.")
    sid = as_id_array(df["subject_id"].to_numpy(dtype=str), "subject_id", path)
    ses = as_id_array(df["session_id"].to_numpy(dtype=str), "session_id", path) if "session_id" in df.columns else None
    keys = make_keys(sid, ses)
    require_unique(keys, path)
    available = [c for c in df.columns if c not in ("subject_id", "session_id")]
    if columns is None:
        columns = available
    missing_cols = [c for c in columns if c not in df.columns]
    if missing_cols:
        raise DataContractError(f"{path}: phenotype column(s) not found: {missing_cols}. Available: {available}")
    out = {}
    for c in columns:
        raw = df[c].astype(str).str.strip()
        is_missing = raw.str.lower().isin(MISSING_TOKENS)
        num = pd.to_numeric(raw.where(~is_missing), errors="coerce")
        bad = num.isna() & ~is_missing
        if bad.any():
            raise DataContractError(
                f"{path}: column '{c}' has non-numeric values, e.g. {_preview(raw[bad].tolist(), 3)}."
            )
        out[c] = num.astype(float).to_numpy()
    return PhenotypeTable(sid, ses, pd.DataFrame(out), path)


def phenotypes_for(samples: SampleSet, table: PhenotypeTable, columns: Sequence[str]) -> np.ndarray:
    """Return a (n_samples, len(columns)) float array aligned to ``samples`` by key.

    Every sample must have a row in the table (missing rows raise); missing
    *values* are returned as NaN and handled by the caller.
    """
    if (samples.session_ids is None) != (table.session_ids is None):
        raise DataContractError(
            f"Key structure mismatch: {samples.label()} "
            f"{'has' if samples.session_ids is not None else 'has no'} session IDs but "
            f"{table.path} {'has' if table.session_ids is not None else 'has no'} 'session_id' column."
        )
    idx = align_to(samples.keys, table.keys, ref_name=samples.label(), other_name=str(table.path))
    missing_cols = [c for c in columns if c not in table.values.columns]
    if missing_cols:
        raise DataContractError(f"{table.path}: phenotype column(s) not loaded: {missing_cols}")
    return table.values[list(columns)].to_numpy(dtype=float)[idx]


# ---------------------------------------------------------------------------
# Atlas / dimension checks
# ---------------------------------------------------------------------------
def check_atlas(samples: SampleSet, *, expected_atlas: str | None, n_regions: int,
                region_ids: np.ndarray | None, context: str) -> str:
    """Validate atlas name, dimension and (if available) region order.

    Returns a short note on what could be verified.
    """
    if samples.atlas is not None and expected_atlas is not None and samples.atlas != expected_atlas:
        raise DataContractError(
            f"{context}: {samples.label()} is in atlas '{samples.atlas}' but '{expected_atlas}' is required."
        )
    if samples.n_regions != n_regions:
        raise DataContractError(
            f"{context}: {samples.label()} has {samples.n_regions} regions; expected {n_regions}."
        )
    if samples.region_ids is not None and region_ids is not None:
        if not np.array_equal(np.asarray(samples.region_ids, str), np.asarray(region_ids, str)):
            raise DataContractError(
                f"{context}: region order of {samples.label()} differs from the expected region_ids."
            )
        return "region order verified by region_ids"
    return "region order NOT verified (region_ids unavailable); only the region count was checked"


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
PARTICIPANT_DIR = "participant_level"
PARTICIPANT_NOTICE = (
    "This directory contains participant-level outputs (IDs, per-participant scores,\n"
    "predictions, split membership). Keep it local. Do not publish or share it.\n"
)


def participant_dir(out_dir: str) -> str:
    d = os.path.join(out_dir, PARTICIPANT_DIR)
    os.makedirs(d, exist_ok=True)
    notice = os.path.join(d, "NOTICE.txt")
    if not os.path.exists(notice):
        with open(notice, "w", encoding="utf-8") as fh:
            fh.write(PARTICIPANT_NOTICE)
    return d


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def _clean_nan(o):
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _clean_nan(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_nan(v) for v in o]
    return o


def write_json(path: str, obj) -> str:
    _ensure_parent(path)
    text = json.dumps(obj, default=_json_default, indent=2, sort_keys=False)
    obj2 = _clean_nan(json.loads(text))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj2, fh, indent=2)
        fh.write("\n")
    return path


def resolve_path(base_dir: str, p: str | None) -> str | None:
    if p is None:
        return None
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base_dir, p))
