"""
data_loading.py
================

I/O utilities for the connectome-mapping pipeline: loading raw connectomes /
timeseries, and saving / loading learned atlas-to-atlas mapping matrices and
the connectomes produced by applying them.

Repository layout (relevant to this file)
-------------------------------------------
    project_root/
    ├── data/
    │   ├── raw/
    │   │   ├── sc/           -> structural connectomes,   <atlas>.npz
    │   │   ├── fc/           -> functional connectomes,   <atlas>.npz
    │   │   └── timeseries/   -> node timeseries,           <atlas>.npz
    │   ├── mappings/         -> learned source->target mapping matrices,
    │   │                        <source_atlas>_to_<target_atlas>.npz
    │   └── mapped/
    │       ├── sc/           -> mapped SC,   <source_atlas>_to_<target_atlas>.npz
    │       ├── fc/           -> mapped FC,   <source_atlas>_to_<target_atlas>.npz
    │       └── timeseries/   -> mapped ts,   <source_atlas>_to_<target_atlas>.npz
    └── src/
        ├── FLECHA/
        ├── mapping/
        │   ├── map_connectome.py     -> applies a saved mapping to sc/fc
        │   ├── map_nodetime.py       -> fits a mapping via optimal transport
        │   └── spectral_embedding.py -> used to evaluate mapping quality
        ├── MECHA/
        └── utils/
            └── data_loading.py   <- this file

Design note
-----------
Each atlas has exactly one canonical SC and one canonical FC (no per-task or
per-reconstruction-method variants). Each atlas pair has exactly one learned
mapping matrix (fit via optimal transport on paired timeseries), which is
then applied to both SC and FC to produce mapped connectomes.

Note on data availability
--------------------------
This repository is prepared for peer review. Due to data-use agreements
governing HCP, HCP-Development (HCP-D), and UK Biobank (UKB), no raw
connectome / timeseries files are included in this submission. All `raw`
loaders below raise an informative FileNotFoundError if the expected file
is missing. `mappings` and `mapped` directories are populated at runtime
once a user supplies their own data and runs the fitting / mapping scripts
in `src/mapping/`. Example data will be released separately upon request.
"""

import os
import numpy as np


# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
# This file lives at <project_root>/src/utils/data_loading.py, so we walk two
# directories up to locate <project_root>/data, regardless of the current
# working directory the caller happens to run scripts from.
_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_UTILS_DIR)
PROJECT_ROOT = os.path.dirname(_SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

RAW_DIR = os.path.join(DATA_DIR, "raw")
MAPPINGS_DIR = os.path.join(DATA_DIR, "mappings")
MAPPED_DIR = os.path.join(DATA_DIR, "mapped")


def _require_path(path):
    """Fail early with a clear message if a required input file is missing
    (e.g. because HCP / HCP-D / UKB data cannot be redistributed here)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Could not find '{path}'.\n"
            f"This repository does not ship HCP / HCP-D / UKB data due to "
            f"data-use agreements. Please obtain access from the original "
            f"data providers and place the corresponding .npz file at the "
            f"path above."
        )
    return path


def _ensure_dir(path):
    """Create the parent directory of `path` if it doesn't exist yet, so
    save functions work on a fresh clone of the repo."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Raw data loaders
# ---------------------------------------------------------------------------
def load_raw_sc(atlas):
    """Load structural connectomes.

    File: data/raw/sc/<atlas>.npz  (key 'mats')
    Array shape: (node, node, nsub)
    """
    path = _require_path(os.path.join(RAW_DIR, "sc", f"{atlas}.npz"))
    x = np.load(path)["mats"]
    print(atlas, x.shape)
    return x


def load_raw_fc(atlas):
    """Load functional connectomes.

    File: data/raw/fc/<atlas>.npz  (key 'mats')
    Array shape: (node, node, nsub)
    """
    path = _require_path(os.path.join(RAW_DIR, "fc", f"{atlas}.npz"))
    x = np.load(path)["mats"]
    print(atlas, x.shape)
    return x


def load_raw_timeseries(atlas):
    """Load node timeseries.

    File: data/raw/timeseries/<atlas>.npz  (key 'mats')
    Array shape: (nsub, node, ntime)
    """
    path = _require_path(os.path.join(RAW_DIR, "timeseries", f"{atlas}.npz"))
    x = np.load(path)["mats"]
    print(atlas, x.shape)
    return x


# ---------------------------------------------------------------------------
# Mapping matrix save / load
# ---------------------------------------------------------------------------
def _mapping_filename(source_atlas, target_atlas):
    return f"{source_atlas}_to_{target_atlas}.npz"


def save_mapping(source_atlas, target_atlas, mapping_matrix, **extra_meta):
    """Save a learned source->target mapping matrix for later reuse.

    File: data/mappings/<source_atlas>_to_<target_atlas>.npz

    Parameters
    ----------
    mapping_matrix : np.ndarray
        The learned optimal-transport plan mapping `source_atlas` node
        space onto `target_atlas` node space, fit from paired timeseries.
        This single matrix is reused to map both SC and FC.
    **extra_meta
        Any additional metadata to store alongside the matrix (e.g. fit
        parameters, convergence stats, evaluation scores).
    """
    fname = _mapping_filename(source_atlas, target_atlas)
    path = _ensure_dir(os.path.join(MAPPINGS_DIR, fname))
    np.savez(
        path,
        mapping_matrix=mapping_matrix,
        source_atlas=source_atlas,
        target_atlas=target_atlas,
        **extra_meta,
    )
    print(f"Saved mapping {source_atlas} -> {target_atlas} to {path}")
    return path


def load_mapping(source_atlas, target_atlas):
    """Load a previously learned source->target mapping matrix.

    File: data/mappings/<source_atlas>_to_<target_atlas>.npz

    Returns
    -------
    dict with keys: 'mapping_matrix', 'source_atlas', 'target_atlas', plus
    any extra metadata that was stored at save time.
    """
    fname = _mapping_filename(source_atlas, target_atlas)
    path = _require_path(os.path.join(MAPPINGS_DIR, fname))
    data = np.load(path, allow_pickle=True)
    result = {k: data[k] for k in data.files}
    print(f"Loaded mapping {source_atlas} -> {target_atlas}, "
          f"matrix shape {result['mapping_matrix'].shape}")
    return result


# ---------------------------------------------------------------------------
# Mapped data save / load
# ---------------------------------------------------------------------------
def _mapped_path(modality, source_atlas, target_atlas):
    fname = f"{source_atlas}_to_{target_atlas}.npz"
    return os.path.join(MAPPED_DIR, modality, fname)


def save_mapped_data(modality, source_atlas, target_atlas, mats):
    """Save connectomes / timeseries produced by applying a mapping.

    Parameters
    ----------
    modality : str
        One of 'sc', 'fc', 'timeseries'.
    """
    path = _ensure_dir(_mapped_path(modality, source_atlas, target_atlas))
    np.savez(path, mats=mats)
    print(f"Saved mapped {modality} ({source_atlas} -> {target_atlas}) "
          f"to {path}, shape {mats.shape}")
    return path


def load_mapped_data(modality, source_atlas, target_atlas):
    """Load previously mapped connectomes / timeseries.

    Parameters
    ----------
    modality : str
        One of 'sc', 'fc', 'timeseries'.
    """
    path = _require_path(_mapped_path(modality, source_atlas, target_atlas))
    x = np.load(path)["mats"]
    print(f"Loaded mapped {modality} ({source_atlas} -> {target_atlas}), "
          f"shape {x.shape}")
    return x


# Thin, explicit wrappers (nice for readability / autocomplete in notebooks)
def save_mapped_sc(source_atlas, target_atlas, mats):
    return save_mapped_data("sc", source_atlas, target_atlas, mats)


def load_mapped_sc(source_atlas, target_atlas):
    return load_mapped_data("sc", source_atlas, target_atlas)


def save_mapped_fc(source_atlas, target_atlas, mats):
    return save_mapped_data("fc", source_atlas, target_atlas, mats)


def load_mapped_fc(source_atlas, target_atlas):
    return load_mapped_data("fc", source_atlas, target_atlas)


def save_mapped_timeseries(source_atlas, target_atlas, mats):
    return save_mapped_data("timeseries", source_atlas, target_atlas, mats)


def load_mapped_timeseries(source_atlas, target_atlas):
    return load_mapped_data("timeseries", source_atlas, target_atlas)


