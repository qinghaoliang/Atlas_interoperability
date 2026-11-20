"""Deterministic synthetic data for the demo suite and tests.

Not a model of real brain data. It only provides inputs with the right
structure: paired multi-atlas time series from shared latent signals,
FC/SC in several atlases with unequal region counts, and phenotypes carrying
subject-level signal.

Generative model
----------------
* ``n_latent`` latent "parcels". Each atlas is a fixed random soft assignment
  of latent parcels to its regions (unequal region counts), so atlases share
  structure without a one-to-one correspondence.
* Each participant has latent factor scores ``z`` (n_factors). Latent time
  series are drawn from a covariance that depends on ``z``; regional time
  series are atlas projections of the latent series plus noise. FC is the
  Pearson correlation of regional time series.
* SC is a symmetric, non-negative atlas projection of a latent structural
  matrix modulated by ``z``.
* Phenotypes are noisy linear functions of ``z``.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import data as D

DEMO_ATLASES = {"synA": 24, "synB": 18, "synC": 30}  # unequal sizes


def _atlas_projection(rng, n_latent, n_regions):
    """Soft, row-normalized assignment (n_regions, n_latent)."""
    centers = np.sort(rng.uniform(0, n_latent, n_regions))
    grid = np.arange(n_latent)
    W = np.exp(-0.5 * ((grid[None, :] - centers[:, None]) / (n_latent / n_regions)) ** 2)
    return W / W.sum(1, keepdims=True)


class SyntheticWorld:
    def __init__(self, seed=20261119, n_latent=48, n_factors=4, atlases=None):
        self.rng = np.random.default_rng(seed)
        self.n_latent, self.n_factors = n_latent, n_factors
        self.atlases = dict(atlases or DEMO_ATLASES)
        self.proj = {a: _atlas_projection(self.rng, n_latent, n) for a, n in self.atlases.items()}
        self.n_sources = 6
        self.base_mix = self.rng.normal(size=(n_latent, self.n_sources)) / np.sqrt(self.n_sources)
        # subject factors perturb the mixing matrix -> first-order changes in FC
        self.loadings = 0.15 * self.rng.normal(size=(n_factors, n_latent, self.n_sources)) / np.sqrt(self.n_sources)
        base_sc = np.abs(self.rng.normal(size=(n_latent, n_latent)))
        self.base_sc = (base_sc + base_sc.T) / 2
        self.sc_mod = self.rng.normal(size=(n_factors, n_latent))
        self._next_id = 0

    def subjects(self, n, prefix="sub-"):
        ids = np.array([f"{prefix}{self._next_id + i:05d}" for i in range(n)], dtype=str)  # leading zeros
        self._next_id += n
        z = self.rng.normal(size=(n, self.n_factors))
        return ids, z

    def latent_timeseries(self, z, n_time):
        out = np.zeros((len(z), self.n_latent, n_time))
        for i, zi in enumerate(z):
            A = self.base_mix + np.tensordot(zi, self.loadings, axes=1)   # (n_latent, n_sources)
            s = self.rng.normal(size=(self.n_sources, n_time))
            out[i] = A @ s + 0.5 * self.rng.normal(size=(self.n_latent, n_time))
        return out

    def regional_timeseries(self, latent, atlas, noise=0.3):
        W = self.proj[atlas]
        ts = np.einsum("rl,slt->srt", W, latent)
        return ts + noise * self.rng.normal(size=ts.shape)

    @staticmethod
    def fc_from_ts(ts):
        n_s, n_r, _ = ts.shape
        out = np.zeros((n_r, n_r, n_s))
        for i in range(n_s):
            c = np.corrcoef(ts[i])
            np.fill_diagonal(c, 1.0)
            out[:, :, i] = (c + c.T) / 2
        return out

    def sc(self, z, atlas):
        W = self.proj[atlas]
        n_r = W.shape[0]
        out = np.zeros((n_r, n_r, len(z)))
        for i, zi in enumerate(z):
            mod = np.exp(0.1 * (zi @ self.sc_mod))
            L = self.base_sc * np.outer(mod, mod)
            S = W @ L @ W.T
            noise = np.abs(self.rng.normal(scale=0.02, size=S.shape))
            S = S + (noise + noise.T) / 2
            np.fill_diagonal(S, 0.0)
            out[:, :, i] = S
        return out

    def phenotypes(self, z, names, noise=1.5, coef_seed=None):
        n_old = 10   # column count of the earlier shared table; keeps "age" values unchanged
        coef = np.random.default_rng(len(names) + 7 if coef_seed is None else coef_seed).normal(
            size=(self.n_factors, max(len(names), n_old if coef_seed is not None else len(names))))
        coef = coef[:, :len(names)]
        noise_draw = self.rng.normal(size=(len(z), max(len(names), n_old if coef_seed is not None else len(names))))
        Y = z @ coef + noise * noise_draw[:, :len(names)]
        return pd.DataFrame(Y, columns=list(names))


def _save_conn(path, mats, ids, atlas, modality):
    D.save_connectomes(path, D.ConnectomeSet(mats, ids, None, atlas,
                                             np.array([f"{atlas}_r{j:03d}" for j in range(mats.shape[0])]),
                                             {"modality": modality, "synthetic": True}))


def _save_ts(path, ts, ids, atlas):
    D.save_timeseries(path, D.TimeSeriesSet(ts, ids, None, atlas,
                                            np.array([f"{atlas}_r{j:03d}" for j in range(ts.shape[1])]),
                                            {"synthetic": True}))


def _save_table(path, ids, df, shuffle_seed):
    df = df.copy()
    df.insert(0, "subject_id", ids)
    df = df.sample(frac=1.0, random_state=shuffle_seed).reset_index(drop=True)  # never row-aligned
    D._ensure_parent(path)
    df.to_csv(path, index=False)


def generate(out_dir: str, seed: int = 20261119) -> dict:
    """Write the full synthetic dataset under ``out_dir``. Returns a manifest."""
    w = SyntheticWorld(seed)
    atlases = list(w.atlases)
    n_time = 120
    man = {"atlases": dict(w.atlases), "files": {}}

    def f(*parts):
        p = os.path.join(out_dir, *parts)
        man["files"]["/".join(parts)] = os.path.relpath(p, out_dir)
        return p

    # 1. Reference cohort: time series only (mapping fit)
    ref_ids, ref_z = w.subjects(12, prefix="ref-")
    lat = w.latent_timeseries(ref_z, n_time)
    for ai, a in enumerate(atlases):
        ts = w.regional_timeseries(lat, a)
        order = np.random.default_rng([seed, ai]).permutation(len(ref_ids))  # rows differ per file
        _save_ts(f("reference", f"timeseries_{a}.npz"), ts[order], ref_ids[order], a)

    # 2. Downstream cohort: FC/SC in every atlas + phenotypes (disjoint from reference)
    n_down = 360
    ids, z = w.subjects(n_down, prefix="sub-")
    lat = w.latent_timeseries(z, 400)  # longer scans -> less FC estimation noise
    for a in atlases:
        _save_conn(f("downstream", f"fc_{a}.npz"), w.fc_from_ts(w.regional_timeseries(lat, a)), ids, a, "fc")
        _save_conn(f("downstream", f"sc_{a}.npz"), w.sc(z, a), ids, a, "sc")
    # "age" is the only phenotype the transform/FLECHA/evaluation workflows use.
    # The coefficient seed depends on the column count; it is pinned to the value
    # used before the MECHA columns were removed so these workflows' data are unchanged.
    phen_names = ["age"]
    ph = w.phenotypes(z, phen_names, coef_seed=10 + 7)
    miss = np.random.default_rng(seed + 3).random((len(ph), 10))[:, :1] < 0.03   # same mask as before
    ph = ph.mask(miss)
    _save_table(f("downstream", "phenotypes.csv"), ids, ph, seed)
    D.save_id_list(f("downstream", "subject_ids.txt"), ids)
    man.update(reference_ids=ref_ids.tolist()[:3], n_reference=len(ref_ids), n_downstream=n_down,
               phenotypes=phen_names)
    D.write_json(os.path.join(out_dir, "manifest.json"), man)
    return man
