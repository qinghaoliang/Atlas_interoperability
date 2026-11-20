"""Synthetic demo suite: runs the real CLI entry points and checks their outputs.

Usage::

    python -m atlas_interop demo --out-dir demo_runs/all [--workflows transform,flecha,...] [--force]

Everything is written under ``--out-dir``; nothing else is read or written.

* ``transform``, ``flecha`` and ``evaluate`` use one small shared synthetic
  cohort and the small ``demo`` presets.
* ``flecha-sim`` runs the synthetic five-site, five-atlas FLECHA experiment
  (:mod:`atlas_interop.flecha_experiment`) on its own cohort.
* ``mecha`` runs the canonical synthetic MECHA experiment
  (:mod:`atlas_interop.mecha_experiment`): 10,000 meta-training and 1,000
  validation participants, one meta-training DNN, K = 10/20/40/100 and
  10 support draws.

The numbers are synthetic and are not manuscript results.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from . import data as D
from .synthetic import generate

WORKFLOWS = ("transform", "flecha", "flecha-sim", "mecha", "evaluate")
DEMO_RANK = "10"
SENTINEL = ".atlas_interop_demo"


class DemoCheckError(AssertionError):
    pass


def check(cond, msg):
    if not cond:
        raise DemoCheckError(msg)


def _cli(args, log):
    cmd = [sys.executable, "-m", "atlas_interop", *args]
    env = dict(os.environ)
    src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    log.append(dict(command="python -m atlas_interop " + " ".join(args), returncode=res.returncode,
                    seconds=round(time.time() - t0, 2), stdout=res.stdout[-4000:], stderr=res.stderr[-4000:]))
    if res.returncode != 0:
        raise DemoCheckError(f"command failed ({res.returncode}): {' '.join(args)}\n{res.stderr[-2000:]}")
    return res


def _subset(conn_path, ids, out_path):
    c = D.load_connectomes(conn_path)
    idx = D.align_to(list(ids), c.keys, ref_name="requested ids", other_name=conn_path)
    D.save_connectomes(out_path, c.take(idx))
    return out_path


# ---------------------------------------------------------------------------
def wf_transform(d, log):
    out = os.path.join(d["root"], "transform")
    ref = os.path.join(d["data"], "reference")
    down = os.path.join(d["data"], "downstream")
    for src, tgt in (("synA", "synB"), ("synB", "synA")):
        mp = os.path.join(out, "mappings", f"{src}_to_{tgt}.npz")
        # Fit uses ONLY paired reference time series (no connectomes are passed).
        _cli(["fit-mapping", "--source-timeseries", f"{ref}/timeseries_{src}.npz",
              "--target-timeseries", f"{ref}/timeseries_{tgt}.npz", "--source-atlas", src,
              "--target-atlas", tgt, "--out", mp, "--quiet",
              "--reference-ids-out", os.path.join(out, "participant_level", "reference_ids.txt")], log)
        from .transform import load_mapping
        m = load_mapping(mp)                                   # reload
        n_s, n_t = d["atlases"][src], d["atlases"][tgt]
        check(m.T.shape == (n_t, n_s), f"mapping shape {m.T.shape} != (n_target, n_source) {(n_t, n_s)}")
        check((m.T >= 0).all(), "negative mapping weights")
        check(np.allclose(m.T.sum(0), 1, atol=1e-6), "mapping source columns do not sum to one")
        with np.load(mp, allow_pickle=False) as z:
            check("subject_ids" not in z.files and "eval_scores" not in z.files,
                  "mapping file contains participant-level data")
        for mod in ("fc", "sc"):
            src_file = f"{down}/{mod}_{src}.npz"
            mapped = os.path.join(out, "mapped", f"{mod}_{src}_to_{tgt}.npz")
            # Apply uses ONLY the mapping and source connectomes.
            _cli(["apply-mapping", "--mapping", mp, "--connectomes", src_file, "--out", mapped,
                  "--modality", mod, "--rank", DEMO_RANK], log)
            mc, sc_in = D.load_connectomes(mapped), D.load_connectomes(src_file)
            check(mc.mats.shape == (n_t, n_t, sc_in.n_samples), f"mapped shape {mc.mats.shape}")
            check(np.isfinite(mc.mats).all(), "non-finite mapped values")
            check(np.array_equal(mc.subject_ids, sc_in.subject_ids), "mapped ID order differs from input")
            check(mc.atlas == tgt, "mapped atlas label wrong")
            # Optional evaluation needs the native target connectomes.
            ev = os.path.join(out, "reconstruction", f"{mod}_{src}_to_{tgt}")
            _cli(["eval-reconstruction", "--mapped", mapped, "--native-target", f"{down}/{mod}_{tgt}.npz",
                  "--out-dir", ev], log)
            with open(os.path.join(ev, "reconstruction_summary.json")) as fh:
                s = json.load(fh)
            check(np.isfinite(s["r_mean"]), "reconstruction r not finite")
            d["results"][f"reconstruction_{mod}_{src}_to_{tgt}"] = round(s["r_mean"], 4)
    # Legacy artifact is rejected with an explanation.
    legacy = os.path.join(out, "legacy_mapping.npz")
    np.savez(legacy, mapping_matrix=np.ones((3, 2)) / 2)
    from .transform import load_mapping
    try:
        load_mapping(legacy)
        raise DemoCheckError("legacy mapping was accepted")
    except D.DataContractError as e:
        check("legacy" in str(e).lower(), "legacy rejection lacks explanation")


def _all_mappings(d, log, modality_atlases, out):
    ref = os.path.join(d["data"], "reference")
    paths = {}
    for src in modality_atlases:
        for tgt in modality_atlases:
            if src == tgt:
                continue
            mp = os.path.join(out, f"{src}_to_{tgt}.npz")
            if not os.path.exists(mp):
                _cli(["fit-mapping", "--source-timeseries", f"{ref}/timeseries_{src}.npz",
                      "--target-timeseries", f"{ref}/timeseries_{tgt}.npz", "--source-atlas", src,
                      "--target-atlas", tgt, "--out", mp, "--quiet"], log)
            paths[f"{src}->{tgt}"] = mp
    return paths


def wf_flecha(d, log):
    """Primary public interface: explicit sites in different native atlases."""
    out = os.path.join(d["root"], "flecha_sites")
    down = os.path.join(d["data"], "downstream")
    maps = _all_mappings(d, log, ["synA", "synB", "synC"], os.path.join(d["root"], "mappings_all"))
    ids = D.load_id_list(os.path.join(down, "subject_ids.txt"))
    rng = np.random.default_rng(1)
    perm = rng.permutation(ids)
    groups = {"siteA": perm[:100], "siteC": perm[100:200], "test": perm[200:280]}
    inp = os.path.join(out, "inputs")
    # each site supplies connectomes ONLY in its own native atlas
    _subset(f"{down}/fc_synA.npz", groups["siteA"], f"{inp}/siteA_fc_synA.npz")
    _subset(f"{down}/fc_synC.npz", groups["siteC"], f"{inp}/siteC_fc_synC.npz")
    _subset(f"{down}/fc_synB.npz", groups["test"], f"{inp}/test_fc_synB.npz")
    cfg = dict(target_atlas="synB", phenotype="age", rank=int(DEMO_RANK),
               train_sites=[dict(name="siteA", connectomes="siteA_fc_synA.npz", labels="../../data/downstream/phenotypes.csv",
                                 mapping=os.path.relpath(maps["synA->synB"], inp)),
                            dict(name="siteC", connectomes="siteC_fc_synC.npz", labels="../../data/downstream/phenotypes.csv",
                                 mapping=os.path.relpath(maps["synC->synB"], inp))],
               test_site=dict(name="test", connectomes="test_fc_synB.npz", labels="../../data/downstream/phenotypes.csv"))
    cfg_path = os.path.join(inp, "sites.json")
    D.write_json(cfg_path, cfg)
    _cli(["flecha", "--config", cfg_path, "--preset", "demo", "--single-site", "--out-dir", out], log)
    res = pd.read_csv(os.path.join(out, "results.csv"))
    check(set(res["method"]) == {"FLECHA", "Single-site"}, f"methods: {set(res['method'])}")
    check(np.isfinite(res["r"]).all(), "non-finite FLECHA metrics")
    with open(os.path.join(out, "run.json")) as fh:
        run = json.load(fh)
    check(run["training"]["n_clients"] == 2, "expected two FLECHA clients")
    check(min(run["training"]["client_update_steps"]) > 0, "a client took no update steps")
    check(all(s["transform"].startswith(("synA", "synC")) for s in run["sites"]), "sites were not transformed")
    pred = pd.read_csv(os.path.join(out, "participant_level", "test_predictions.csv"), dtype={"subject_id": str})
    test_in = list(D.load_connectomes(f"{inp}/test_fc_synB.npz").subject_ids)
    check(list(pred["subject_id"]) == test_in, "test prediction ID order differs from the test file")
    labelled = pred["observed"].notna()
    check(pred.loc[~labelled, "predicted_flecha"].isna().all(), "unlabelled test samples were scored")
    check(int(labelled.sum()) == run["test_site"]["n_used"], "realized test size mismatch")
    d["results"]["flecha_sites_r"] = res.set_index("method")["r"].round(4).to_dict() if res["method"].is_unique else \
        res.groupby("method")["r"].mean().round(4).to_dict()


def wf_flecha_sim(d, log):
    """Synthetic five-site, five-atlas FLECHA experiment (atlas_interop.flecha_experiment)."""
    from . import flecha_experiment as FX
    out = os.path.join(d["root"], "flecha_simulated")
    data = os.path.join(out, "data")
    FX.generate(data)
    ref = os.path.join(data, "reference")
    maps = {}
    for a in FX.TRAIN_ATLASES:
        mp = os.path.join(out, "mappings", f"{a}_to_{FX.TARGET}.npz")
        _cli(["fit-mapping", "--source-timeseries", f"{ref}/timeseries_{a}.npz",
              "--target-timeseries", f"{ref}/timeseries_{FX.TARGET}.npz", "--source-atlas", a,
              "--target-atlas", FX.TARGET, "--out", mp, "--quiet",
              "--reference-ids-out", os.path.join(out, "mappings", "participant_level", "reference_ids.txt")], log)
        maps[a] = mp
    _cli(["flecha-sim5", "--data-dir", data, "--out-dir", os.path.join(out, "results"),
          "--reference-ids", os.path.join(out, "mappings", "participant_level", "reference_ids.txt"),
          *sum((["--mapping", f"{a}={p}"] for a, p in maps.items()), [])], log)
    res = os.path.join(out, "results")
    runs = pd.read_csv(os.path.join(res, "runs.csv"))
    sites = pd.read_csv(os.path.join(res, "sites.csv"))
    mem = pd.read_csv(os.path.join(res, "participant_level", "site_membership.csv"), dtype={"subject_id": str})
    seeds = FX.PARTITION_SEEDS
    check(sorted(runs.seed.unique()) == list(seeds), "every predetermined partition seed ran")
    for seed in seeds:
        g = runs[runs.seed == seed]
        m = mem[mem.seed == seed]
        check(m.site.nunique() == 5 and m.native_atlas.nunique() == 5 and m.subject_id.is_unique,
              f"seed {seed}: five disjoint sites with five distinct native atlases")
        check(dict(m.groupby("site").native_atlas.first()) ==
              {"site1": "synA", "site2": "synB", "site3": "synC", "site4": "synD", "test": "synE"},
              f"seed {seed}: site roles synA-D train, synE test")
        check(sorted(g.method.value_counts().items()) == [("FLECHA", 1), ("Single-site", 4), ("Uniform", 1)],
              f"seed {seed}: FLECHA, Uniform and four Single-site fits")
        check(g.test_fingerprint.nunique() == 1 and g.n_test.nunique() == 1,
              f"seed {seed}: all methods scored on the same held-out synE participants")
        test_labelled = sites[(sites.seed == seed) & (sites.site == "test")].fingerprint.iloc[0]
        check(g.test_fingerprint.iloc[0] == test_labelled,
              f"seed {seed}: scored test IDs equal the labelled members of the synE test site")
        fl = g[g.method == "FLECHA"].train_fingerprint.iloc[0]
        check(fl == g[g.method == "Uniform"].train_fingerprint.iloc[0], f"seed {seed}: FLECHA and Uniform train IDs match")
        st = sites[(sites.seed == seed) & (sites.site != "test")].set_index("site").fingerprint
        for r in g[g.method == "Single-site"].itertuples():
            check(r.train_fingerprint == st[r.site], f"seed {seed}: Single-site {r.site} saw only its own site")
        check((g.update_steps > 0).all(), f"seed {seed}: every training branch took steps")
        check(np.isfinite(g.r).all(), f"seed {seed}: finite scores")
    summ = pd.read_csv(os.path.join(res, "summary.csv"))
    d["results"]["flecha_simulated_5site"] = summ.round(4).to_dict(orient="records")


def wf_mecha(d, log):
    """Canonical synthetic MECHA experiment (see atlas_interop.mecha_experiment)."""
    from .mecha_experiment import run_experiment
    res = run_experiment(os.path.join(d["root"], "mecha"), cli=lambda args: _cli(args, log), check=check)
    d["results"]["mecha"] = res


def wf_evaluate(d, log):
    out = os.path.join(d["root"], "evaluation")
    down = os.path.join(d["data"], "downstream")
    atl = ["synA", "synB"]
    maps = _all_mappings(d, log, atl, os.path.join(d["root"], "mappings_all"))
    mapped = {}
    for k, mp in maps.items():
        src, tgt = k.split("->")
        p = os.path.join(out, "mapped", f"fc_{src}_to_{tgt}.npz")
        _cli(["apply-mapping", "--mapping", mp, "--connectomes", f"{down}/fc_{src}.npz", "--out", p,
              "--modality", "fc", "--rank", DEMO_RANK], log)
        mapped[k] = os.path.relpath(p, out)
    cfg = dict(modality="fc", native={a: os.path.relpath(f"{down}/fc_{a}.npz", out) for a in atl}, mapped=mapped,
               phenotypes=os.path.relpath(f"{down}/phenotypes.csv", out), phenotype_columns=["age"])
    cfg_path = os.path.join(out, "evaluation.json")
    D.write_json(cfg_path, cfg)
    _cli(["evaluate", "--config", cfg_path, "--preset", "demo", "--out-dir", out], log)
    pred = pd.read_csv(os.path.join(out, "prediction_runs.csv"))
    check(len(pred) == 4 * 3, f"expected 4 pairs x 3 repeats, got {len(pred)} rows")
    check(np.isfinite(pred["r"]).all(), "non-finite prediction r")
    with open(os.path.join(out, "analysis_summary.json")) as fh:
        a = json.load(fh)
    check("resolution_vs_reconstruction" in a and "reconstruction_vs_prediction" in a, "analysis summaries missing")
    d["results"]["prediction_mean_r"] = pred.groupby(["source_atlas", "target_atlas"])["r"].mean().round(4).to_dict()
    d["results"]["prediction_mean_r"] = {f"{s}->{t}": v for (s, t), v in d["results"]["prediction_mean_r"].items()}



RUNNERS = {"transform": wf_transform, "flecha": wf_flecha, "flecha-sim": wf_flecha_sim,
           "mecha": wf_mecha, "evaluate": wf_evaluate}
SHARED_COHORT_WORKFLOWS = {"transform", "flecha", "flecha-sim", "evaluate"}


def run_demo(out_dir: str, workflows=None, force: bool = False) -> bool:
    workflows = list(workflows or WORKFLOWS)
    unknown = [w for w in workflows if w not in RUNNERS]
    if unknown:
        raise SystemExit(f"unknown workflow(s): {unknown}; choose from {list(WORKFLOWS)}")
    root = os.path.abspath(out_dir)
    if os.path.exists(root) and os.listdir(root):
        if not os.path.exists(os.path.join(root, SENTINEL)):
            raise SystemExit(f"{root} exists and was not created by the demo; refusing to write into it.")
        if not force:
            raise SystemExit(f"{root} already holds demo output; pass --force to replace it.")
        shutil.rmtree(root)
    os.makedirs(root)
    open(os.path.join(root, SENTINEL), "w").close()
    d = dict(root=root, data=os.path.join(root, "data"), atlases={}, results={})
    if SHARED_COHORT_WORKFLOWS & set(workflows):
        d["atlases"] = generate(d["data"])["atlases"]
    log, status = [], {}
    for w in workflows:
        t0 = time.time()
        try:
            RUNNERS[w](d, log)
            status[w] = dict(status="passed", seconds=round(time.time() - t0, 1))
        except Exception as e:  # report and continue with the other workflows
            status[w] = dict(status="FAILED", error=f"{type(e).__name__}: {e}", seconds=round(time.time() - t0, 1))
        print(f"[demo] {w:<11} {status[w]['status']}  ({status[w]['seconds']} s)"
              + (f"\n        {status[w]['error']}" if status[w]["status"] != "passed" else ""), flush=True)
    D.write_json(os.path.join(root, "demo_report.json"), dict(
        workflows=status, results=d["results"], commands=[{k: v for k, v in c.items() if k != "stdout"} for c in log],
        note="Synthetic data; these numbers are not manuscript results."))
    ok = all(s["status"] == "passed" for s in status.values())
    print(f"[demo] {'ALL PASSED' if ok else 'SOME WORKFLOWS FAILED'}; report: {os.path.join(root, 'demo_report.json')}")
    return ok
