"""Synthetic cohort for the canonical MECHA example.

Not a model of real brain data. It produces inputs with the structure MECHA
needs, at a meta-training scale where pretraining can be learned. By design,
the meta-training and held-out phenotypes depend on partly the same
connectome-predictable latent factors (shared predictive signal); this is a
property of the synthetic generator, not something the experiment tests.

Latent connectivity process
---------------------------
Every participant has 11 independent standard-normal phenotype-relevant
latent factors ``z`` and 30 nuisance factors ``n`` that never enter any
phenotype. Each factor perturbs a shared latent mixing matrix (48 latent
parcels x 8 sources) with its own random loading pattern, all drawn from the
same distribution, so every factor changes functional connectivity by a
comparable amount. Latent time series are
``(B + sum_f z_f L_f + sum_g n_g N_g) s + noise``. The nuisance factors make
connectivity vary in many directions that do not predict any label, so a
small support set cannot find the predictive directions as easily as a
model that has seen 10,000 labelled participants.
Two atlases with unequal region counts (``synS``: 36, ``synT``: 28) are fixed
soft projections of the latent parcels; regional time series are projections
plus noise, and FC is their Pearson correlation. Both atlas representations of
a participant therefore come from the same latent connectivity. Phenotype
labels are never written into the connectomes.

Factor groups
-------------
* ``A`` = factors 0-3: shared; drive both the meta-training and the held-out
  phenotypes.
* factors 4-6: affect connectivity but no phenotype in this experiment.
* ``C`` = factors 7-10: held-out-specific connectome signal, used by no
  meta-training phenotype.

Phenotypes (all labels complete)
--------------------------------
* 16 meta-training phenotypes: ``train_i = b_i . z[A] + sqrt(0.6) u_i``.
* 10 held-out phenotypes:
  ``heldout_j = sqrt(0.75) a_j . z[A] + sqrt(0.25) c_j . z[C] + sqrt(0.6) v_j``.

``b``, ``a``, ``c`` are unit-norm coefficient rows. ``u`` and ``v`` are
independent standard-normal phenotype-specific terms that do not appear in the
connectomes. The connectome-explained label variance is 1.0 of 1.6 for every
phenotype. The held-out phenotypes therefore share 75% of their connectome
signal variance with the factors that the meta-training phenotypes use, and
25% is specific to them.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import data as D

ATLASES = {"synS": 36, "synT": 28}          # source, target (unequal sizes)
SOURCE, TARGET = "synS", "synT"
COHORTS = {"reference": 20, "meta_training": 10_000, "validation": 1_000, "support": 1_000, "test": 2_000}
N_TRAIN_PHEN, N_HELDOUT_PHEN = 16, 10
FACTORS_A, FACTORS_C = (0, 1, 2, 3), (7, 8, 9, 10)
N_FACTORS = 11                              # factors 4-6 affect connectivity only
N_NUISANCE = 30
N_LATENT, N_SOURCES = 48, 8
T_REFERENCE, T_DOWNSTREAM = 200, 300
LOADING_SCALE = 0.25
SPECIFIC_VAR = 0.6

TRAIN_COLUMNS = [f"train_{i:02d}" for i in range(1, N_TRAIN_PHEN + 1)]
HELDOUT_COLUMNS = [f"heldout_{j:02d}" for j in range(1, N_HELDOUT_PHEN + 1)]


def _projection(rng, n_regions):
    centers = np.sort(rng.uniform(0, N_LATENT, n_regions))
    grid = np.arange(N_LATENT)
    W = np.exp(-0.5 * ((grid[None, :] - centers[:, None]) / (N_LATENT / n_regions)) ** 2)
    return W / W.sum(1, keepdims=True)


def _unit_rows(rng, n, d):
    M = rng.normal(size=(n, d))
    return M / np.linalg.norm(M, axis=1, keepdims=True)


class MechaWorld:
    def __init__(self, seed: int):
        rng = np.random.default_rng([seed, 0])
        self.proj = {a: _projection(rng, n) for a, n in ATLASES.items()}
        self.base = rng.normal(size=(N_LATENT, N_SOURCES)) / np.sqrt(N_SOURCES)
        self.loadings = LOADING_SCALE * rng.normal(size=(N_FACTORS + N_NUISANCE, N_LATENT, N_SOURCES)) / np.sqrt(N_SOURCES)
        coef = np.random.default_rng([seed, 1])
        self.b = _unit_rows(coef, N_TRAIN_PHEN, len(FACTORS_A))
        self.a = _unit_rows(coef, N_HELDOUT_PHEN, len(FACTORS_A))
        self.c = _unit_rows(coef, N_HELDOUT_PHEN, len(FACTORS_C))
        self.seed = seed

    def timeseries(self, z, n_time, rng):
        """Regional time series for each atlas, dict atlas -> (n, regions, time)."""
        n = len(z)
        zn = np.concatenate([z, rng.normal(size=(n, N_NUISANCE))], axis=1)   # nuisance: no phenotype
        A = self.base[None] + np.einsum("nf,fls->nls", zn, self.loadings)
        s = rng.normal(size=(n, N_SOURCES, n_time))
        lat = np.einsum("nls,nst->nlt", A, s) + 0.5 * rng.normal(size=(n, N_LATENT, n_time))
        out = {}
        for atlas, W in self.proj.items():
            ts = np.einsum("rl,nlt->nrt", W, lat)
            out[atlas] = ts + 0.3 * rng.normal(size=ts.shape)
        return out

    @staticmethod
    def fc(ts):
        x = ts - ts.mean(axis=2, keepdims=True)
        x /= np.linalg.norm(x, axis=2, keepdims=True)
        c = np.einsum("nrt,nqt->nrq", x, x)
        c = (c + c.transpose(0, 2, 1)) / 2
        idx = np.arange(c.shape[1])
        c[:, idx, idx] = 1.0
        return np.ascontiguousarray(c.transpose(1, 2, 0))       # (r, r, n)

    def phenotypes(self, z, rng):
        n = len(z)
        u = rng.normal(size=(n, N_TRAIN_PHEN))
        v = rng.normal(size=(n, N_HELDOUT_PHEN))
        tr = z[:, FACTORS_A] @ self.b.T + np.sqrt(SPECIFIC_VAR) * u
        ho = (np.sqrt(0.75) * z[:, FACTORS_A] @ self.a.T + np.sqrt(0.25) * z[:, FACTORS_C] @ self.c.T
              + np.sqrt(SPECIFIC_VAR) * v)
        return np.concatenate([tr, ho], axis=1)


def _region_ids(atlas):
    return np.array([f"{atlas}_r{j:03d}" for j in range(ATLASES[atlas])])


def estimate_resources(counts: dict | None = None) -> dict:
    """Approximate disk and peak-memory needs of the generated inputs (bytes)."""
    counts = counts or COHORTS
    nS, nT = ATLASES[SOURCE], ATLASES[TARGET]
    source_fc = (counts["meta_training"] + counts["validation"]) * nS * nS * 8
    target_fc = (counts["support"] + counts["test"]) * nT * nT * 8
    ts = counts["reference"] * (nS + nT) * T_REFERENCE * 8
    n_down = sum(v for k, v in counts.items() if k != "reference")
    table = n_down * (len(TRAIN_COLUMNS) + len(HELDOUT_COLUMNS) + 1) * 20
    disk = source_fc + target_fc + ts + table
    # peak during training: source FC + transformed FC (float64) + edge features (float32)
    nE = nT * (nT - 1) // 2
    train_peak = (counts["meta_training"] * (nS * nS + nT * nT) * 8
                  + (counts["meta_training"] + counts["validation"]) * nE * 4 * 3)
    return dict(disk_bytes=int(disk), peak_training_memory_bytes=int(train_peak),
                note="estimate from array sizes; excludes Python/PyTorch overhead (~0.5-1 GB)")


def generate(out_dir: str, seed: int = 20261120, counts: dict | None = None, chunk: int = 500) -> dict:
    """Write the synthetic MECHA cohort. Returns a manifest (no participant IDs)."""
    counts = dict(counts or COHORTS)
    w = MechaWorld(seed)
    rng = np.random.default_rng([seed, 2])
    total = sum(counts.values())
    pool = rng.permutation(total)                         # opaque IDs, roles not inferable from IDs
    ids_all = np.array([f"syn{p:06d}" for p in pool], dtype=str)
    roles, start = {}, 0
    for role, n in counts.items():
        roles[role] = ids_all[start:start + n]
        start += n
    D.assert_disjoint_subjects(roles)
    z_all = {role: rng.normal(size=(n, N_FACTORS)) for role, n in counts.items()}
    files = {}

    def path(*p):
        full = os.path.join(out_dir, *p)
        files["/".join(p)] = "/".join(p)
        return full

    # reference cohort: paired time series only
    ts = w.timeseries(z_all["reference"], T_REFERENCE, rng)
    for k, atlas in enumerate(ATLASES):
        order = np.random.default_rng([seed, 3, k]).permutation(counts["reference"])  # rows differ per file
        D.save_timeseries(path("reference", f"timeseries_{atlas}.npz"),
                          D.TimeSeriesSet(ts[atlas][order], roles["reference"][order], None, atlas,
                                          _region_ids(atlas), {"synthetic": True}))
    # downstream cohorts: FC in the atlas each role needs
    need = {"meta_training": SOURCE, "validation": SOURCE, "support": TARGET, "test": TARGET}
    phen_rows = []
    for role, atlas in need.items():
        z = z_all[role]
        mats = np.zeros((ATLASES[atlas], ATLASES[atlas], len(z)))
        for s in range(0, len(z), chunk):
            mats[:, :, s:s + chunk] = w.fc(w.timeseries(z[s:s + chunk], T_DOWNSTREAM, rng)[atlas])
        D.save_connectomes(path(f"{role}_fc_{atlas}.npz"),
                           D.ConnectomeSet(mats, roles[role], None, atlas, _region_ids(atlas),
                                           {"modality": "fc", "synthetic": True, "cohort": role}))
        phen_rows.append(pd.DataFrame(w.phenotypes(z, rng), columns=TRAIN_COLUMNS + HELDOUT_COLUMNS)
                         .assign(subject_id=roles[role]))
    table = pd.concat(phen_rows, ignore_index=True)
    table = table[["subject_id"] + TRAIN_COLUMNS + HELDOUT_COLUMNS]
    table = table.sample(frac=1.0, random_state=seed).reset_index(drop=True)   # never row-aligned
    D._ensure_parent(path("phenotypes.csv"))
    table.to_csv(path("phenotypes.csv"), index=False, float_format="%.6g")
    manifest = dict(
        seed=seed, counts=counts, atlases=dict(ATLASES), source_atlas=SOURCE, target_atlas=TARGET,
        cohort_atlas={"reference": "both (paired time series)", **need}, files=files,
        phenotypes=dict(meta_training=TRAIN_COLUMNS, heldout=HELDOUT_COLUMNS),
        design=dict(
            n_factors=N_FACTORS, n_nuisance_factors=N_NUISANCE, factors_shared_A=list(FACTORS_A),
            factors_heldout_specific_C=list(FACTORS_C), factors_connectivity_only=[4, 5, 6],
            connectome_explained_label_variance=1.0, phenotype_specific_variance=SPECIFIC_VAR,
            heldout_weights={"A_shared": 0.75, "C_specific": 0.25},
            loading_scale=LOADING_SCALE, n_latent=N_LATENT, n_sources=N_SOURCES,
            timepoints={"reference": T_REFERENCE, "downstream": T_DOWNSTREAM}),
        resources=estimate_resources(counts),
        cohort_fingerprints={r: D.fingerprint(i) for r, i in roles.items()},
    )
    D.write_json(os.path.join(out_dir, "generator_manifest.json"), manifest)
    return manifest
