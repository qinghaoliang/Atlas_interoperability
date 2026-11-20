"""Synthetic five-site, five-atlas FLECHA experiment for reviewers.

Run with::

    python -m atlas_interop demo --out-dir demo_runs/flecha5 --workflows flecha-sim

This is a fixed synthetic design for reviewers only. It does not constrain the
public explicit-site interface (``atlas_interop flecha``), which accepts any
number of sites in any atlases, or the manuscript simulated-site command
(``atlas_interop flecha-simulate``) and its presets.

Design
------
* Five atlases ``synA`` 24, ``synB`` 18, ``synC`` 30, ``synD`` 21 and
  ``synE`` 27 regions, all soft projections of one shared latent
  connectivity process (``synthetic.SyntheticWorld``).
* Independent mapping-reference cohort (time series in all five atlases).
  The four directed mappings ``synA/B/C/D -> synE`` are fit from these
  paired time series only. No held-out participant is used to fit anything.
* Downstream cohort with FC natively generated in every atlas and one
  phenotype (``age``) with subject-level connectome signal.
* For each predetermined partition seed, the downstream participants are
  split into five disjoint, nonempty sites:
  training sites 1-4 have native atlases synA, synB, synC, synD; the held-out
  test site has native atlas synE, the common target atlas.

Methods (all scored on the same held-out synE participants)
------------------------------------------------------------
* FLECHA: each training site transforms its own native FC into synE;
  ``flecha.fedavg_train`` runs local updates and equal-weight parameter
  averaging across the four sites. Site data are never pooled.
* Uniform: the same four site memberships and the same ``fedavg_train``
  procedure, but on those participants' native synE FC (generated directly
  in synE, not mapped).
* Single-site: for each of the four training sites separately, transform its
  FC into synE and fit one standalone model with ``flecha.train_single_site``
  on that site's labelled participants only: no server, rounds,
  synchronization or averaging. The manuscript Methods say that, when several
  source-site values exist within a run, they were aggregated by averaging
  Fisher-z-transformed correlations and back-transforming. That rule is
  prespecified here: every site is fit and reported, and the per-partition
  Single-site summary is the Fisher-z mean of the four site scores. No site is
  ever selected by test score.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from . import __version__
from . import data as D
from . import flecha as F
from .models import get_device
from .presets import FLECHA_PRESETS
from .stats import fisher_mean_r, fisher_z, wilcoxon_greater, bh_fdr
from .synthetic import SyntheticWorld
from .transform import load_mapping

ATLASES = {"synA": 24, "synB": 18, "synC": 30, "synD": 21, "synE": 27}
TRAIN_ATLASES = ("synA", "synB", "synC", "synD")
TARGET = "synE"
COUNTS = {"reference": 12, "downstream": 600}
PARTITION_SEEDS = (101, 102, 103, 104, 105)
TRAIN_SEED = 0
RANK = 10
PRESET = "demo"
GENERATOR_SEED = 20261121
N_TIME = {"reference": 120, "downstream": 400}


class FlechaDesignError(D.DataContractError):
    pass


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def generate(out_dir: str, seed: int = GENERATOR_SEED, counts: dict | None = None) -> dict:
    counts = dict(counts or COUNTS)
    w = SyntheticWorld(seed=seed, atlases=ATLASES)
    files = {}

    def path(*p):
        full = os.path.join(out_dir, *p)
        files["/".join(p)] = "/".join(p)
        return full

    def rid(a):
        return np.array([f"{a}_r{j:03d}" for j in range(ATLASES[a])])

    ref_ids, ref_z = w.subjects(counts["reference"], prefix="ref-")
    lat = w.latent_timeseries(ref_z, N_TIME["reference"])
    for k, a in enumerate(ATLASES):
        order = np.random.default_rng([seed, k]).permutation(len(ref_ids))      # rows differ per file
        D.save_timeseries(path("reference", f"timeseries_{a}.npz"),
                          D.TimeSeriesSet(w.regional_timeseries(lat, a)[order], ref_ids[order], None, a, rid(a),
                                          {"synthetic": True}))
    ids, z = w.subjects(counts["downstream"], prefix="sub-")
    lat = w.latent_timeseries(z, N_TIME["downstream"])
    for a in ATLASES:
        D.save_connectomes(path("downstream", f"fc_{a}.npz"),
                           D.ConnectomeSet(w.fc_from_ts(w.regional_timeseries(lat, a)), ids, None, a, rid(a),
                                           {"modality": "fc", "synthetic": True, "native": True}))
    ph = w.phenotypes(z, ["age"])
    ph.insert(0, "subject_id", ids)
    ph.sample(frac=1.0, random_state=seed).to_csv(path("downstream", "phenotypes.csv"), index=False)
    D.assert_disjoint_subjects({"reference": ref_ids, "downstream": ids})
    man = dict(seed=seed, counts=counts, atlases=dict(ATLASES), files=files,
               cohort_fingerprints={"reference": D.fingerprint(ref_ids), "downstream": D.fingerprint(ids)})
    D.write_json(os.path.join(out_dir, "generator_manifest.json"), man)
    return man


# ---------------------------------------------------------------------------
# Site design
# ---------------------------------------------------------------------------
def site_design(train_atlases=TRAIN_ATLASES, target=TARGET) -> list[dict]:
    """Five roles: four training sites with distinct native atlases, then the test site."""
    roles = [dict(site=f"site{i}", role="train", native_atlas=a) for i, a in enumerate(train_atlases, start=1)]
    roles.append(dict(site="test", role="test", native_atlas=target))
    natives = [r["native_atlas"] for r in roles]
    if len(roles) != 5:
        raise FlechaDesignError(f"the reviewer simulation needs exactly 5 sites, got {len(roles)}")
    if len(set(natives)) != len(natives):
        dup = sorted({a for a in natives if natives.count(a) > 1})
        raise FlechaDesignError(f"each simulated site needs a distinct native atlas; duplicated: {dup}")
    return roles


def partition(subject_ids: np.ndarray, seed: int, n_sites: int = 5) -> list[np.ndarray]:
    """Disjoint, nonempty subject-level partition (index arrays), reusing flecha.partition_sites."""
    parts = F.partition_sites(subject_ids, subject_ids, n_sites, seed)
    if any(len(p) == 0 for p in parts):
        raise FlechaDesignError("a simulated site is empty")
    D.assert_disjoint_subjects({f"site{i}": subject_ids[p] for i, p in enumerate(parts)})
    return parts


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
def run(out: str, *, data_dir: str, mappings: dict[str, str], preset_name: str = PRESET,
        seeds=PARTITION_SEEDS, rank: int = RANK, device: str = "cpu",
        train_atlases=TRAIN_ATLASES, target=TARGET, reference_ids: str | None = None) -> dict:
    """Run FLECHA, Uniform and Single-site on every predetermined partition."""
    roles = site_design(train_atlases, target)
    preset = FLECHA_PRESETS[preset_name]
    dev = get_device(device)
    os.makedirs(out, exist_ok=True)
    pdir = D.participant_dir(out)
    config = dict(experiment="synthetic five-site FLECHA (reviewer example)", toolkit_version=__version__,
                  site_roles=roles, target_atlas=target, partition_seeds=list(seeds), train_seed=TRAIN_SEED,
                  spectral_rank=rank, preset=preset.as_dict(),
                  single_site_rule=("each training site fit separately with train_single_site; "
                                    "per-partition summary = Fisher-z mean of the four site r values "
                                    "(manuscript Methods aggregation rule); no selection by test score"),
                  single_site_budget=("epochs = ceil(rounds*local_steps*n_clients*batch_size / n_site), clipped "
                                      f"to [{preset.epochs_min}, {preset.epochs_max}] (flecha.matched_epochs)"),
                  planned_comparisons=["FLECHA - Single-site (paired over partitions, one-sided Wilcoxon)",
                                       "Uniform - FLECHA (descriptive gap)"])
    D.write_json(os.path.join(out, "experiment_config.json"), config)   # frozen before any score

    native = {a: D.load_connectomes(os.path.join(data_dir, "downstream", f"fc_{a}.npz")) for a in ATLASES}
    for a, c in native.items():
        if c.atlas != a or c.metadata.get("transformed_from"):
            raise FlechaDesignError(f"fc_{a}.npz must be native {a} connectomes")
    ref = native[target]
    aligned = {a: c.take(D.align_to(ref.keys, c.keys, ref_name=ref.label(), other_name=c.label(),
                                    allow_extra=False)) for a, c in native.items()}
    if reference_ids:
        D.assert_disjoint_subjects({"mapping reference": D.load_id_list(reference_ids),
                                    "downstream": ref.subject_ids})
    table = D.load_phenotypes(os.path.join(data_dir, "downstream", "phenotypes.csv"), ["age"])
    y = D.phenotypes_for(ref, table, ["age"])[:, 0]
    maps = {a: load_mapping(mappings[a]) for a in train_atlases}
    for a, m in maps.items():
        if (m.source_atlas, m.target_atlas) != (a, target):
            raise FlechaDesignError(f"mapping for {a} is {m.source_atlas}->{m.target_atlas}")

    rows, site_rows, membership = [], [], []
    for seed in seeds:
        parts = partition(ref.subject_ids, seed)
        sites = {}
        for r, idx in zip(roles, parts):
            sites[r["site"]] = idx
            membership += [dict(seed=seed, site=r["site"], role=r["role"], native_atlas=r["native_atlas"],
                                subject_id=s) for s in ref.subject_ids[idx]]
        test_idx = sites["test"]
        test = F.prepare_site("test", aligned[target].take(test_idx), y[test_idx], target_atlas=target,
                              n_target_regions=ATLASES[target], mapping=None, rank=None)
        hetero, uniform = [], []
        for r in roles[:-1]:
            idx = sites[r["site"]]
            hetero.append(F.prepare_site(r["site"], aligned[r["native_atlas"]].take(idx), y[idx], target_atlas=target,
                                         n_target_regions=ATLASES[target], mapping=maps[r["native_atlas"]], rank=rank))
            uniform.append(F.prepare_site(r["site"], aligned[target].take(idx), y[idx], target_atlas=target,
                                          n_target_regions=ATLASES[target], mapping=None, rank=None))
        D.assert_disjoint_subjects({s.name: s.subject_ids for s in hetero + [test]})
        for h, u in zip(hetero, uniform):
            if not np.array_equal(h.subject_ids, u.subject_ids):
                raise FlechaDesignError(f"seed {seed}: FLECHA and Uniform training IDs differ at {h.name}")
        test_fp = D.fingerprint(test.subject_ids)
        base = dict(seed=seed, n_test=int(len(test.y)), test_fingerprint=test_fp)
        for r, h in zip(roles[:-1], hetero):
            site_rows.append(dict(seed=seed, site=h.name, native_atlas=r["native_atlas"], n_labelled=int(len(h.y)),
                                  n_missing_label=h.report["n_missing_label"], fingerprint=D.fingerprint(h.subject_ids)))
        site_rows.append(dict(seed=seed, site="test", native_atlas=target, n_labelled=int(len(test.y)),
                              n_missing_label=test.report["n_missing_label"], fingerprint=test_fp))

        t0 = time.time()
        m, info = F.fedavg_train(hetero, preset, TRAIN_SEED, dev)
        met, _ = F.evaluate(m, test, dev)
        rows.append(dict(base, method="FLECHA", site="site1-4", **met, update_steps=sum(info["client_update_steps"]),
                         samples_processed=info["samples_processed"], seconds=round(time.time() - t0, 2),
                         train_fingerprint=D.fingerprint(np.concatenate([s.subject_ids for s in hetero]))))
        t0 = time.time()
        m, info = F.fedavg_train(uniform, preset, TRAIN_SEED, dev)
        met, _ = F.evaluate(m, test, dev)
        rows.append(dict(base, method="Uniform", site="site1-4", **met, update_steps=sum(info["client_update_steps"]),
                         samples_processed=info["samples_processed"], seconds=round(time.time() - t0, 2),
                         train_fingerprint=D.fingerprint(np.concatenate([s.subject_ids for s in uniform]))))
        for r, h in zip(roles[:-1], hetero):
            t0 = time.time()
            ep = F.matched_epochs(len(h.y), len(hetero), preset)
            m, info = F.train_single_site(h, preset, ep, TRAIN_SEED, dev)
            met, _ = F.evaluate(m, test, dev)
            rows.append(dict(base, method="Single-site", site=h.name, native_atlas=r["native_atlas"], **met,
                             epochs=ep, update_steps=info["update_steps"], samples_processed=info["samples_processed"],
                             seconds=round(time.time() - t0, 2), train_fingerprint=D.fingerprint(h.subject_ids)))

    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(out, "runs.csv"), index=False)
    pd.DataFrame(site_rows).to_csv(os.path.join(out, "sites.csv"), index=False)
    pd.DataFrame(membership).to_csv(os.path.join(pdir, "site_membership.csv"), index=False)
    summary = summarize(runs)
    summary.to_csv(os.path.join(out, "summary.csv"), index=False)
    return dict(runs=runs, summary=summary, config=config)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """Per-partition table: FLECHA, Uniform, Single-site (Fisher-z mean of 4 sites)."""
    per = []
    for seed, g in runs.groupby("seed", sort=True):
        ss = g[g.method == "Single-site"]
        per.append(dict(seed=int(seed), FLECHA=float(g[g.method == "FLECHA"].r.iloc[0]),
                        Uniform=float(g[g.method == "Uniform"].r.iloc[0]),
                        SingleSite_mean=fisher_mean_r(ss.r), SingleSite_min=float(ss.r.min()),
                        SingleSite_max=float(ss.r.max()), n_test=int(g.n_test.iloc[0])))
    per = pd.DataFrame(per)
    per["FLECHA_minus_SingleSite"] = per.FLECHA - per.SingleSite_mean
    per["Uniform_minus_FLECHA"] = per.Uniform - per.FLECHA
    return per


def overall(per: pd.DataFrame) -> dict:
    w = wilcoxon_greater(fisher_z(per.FLECHA), fisher_z(per.SingleSite_mean))
    return dict(n_partitions=int(len(per)),
                FLECHA_mean=float(per.FLECHA.mean()), Uniform_mean=float(per.Uniform.mean()),
                SingleSite_mean=float(per.SingleSite_mean.mean()),
                FLECHA_minus_SingleSite_mean=float(per.FLECHA_minus_SingleSite.mean()),
                partitions_FLECHA_better=int((per.FLECHA_minus_SingleSite > 0).sum()),
                Uniform_minus_FLECHA_mean=float(per.Uniform_minus_FLECHA.mean()),
                wilcoxon_p_FLECHA_gt_SingleSite=w["p"], wilcoxon_note=w["note"])
