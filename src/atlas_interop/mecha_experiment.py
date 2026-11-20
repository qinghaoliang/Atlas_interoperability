"""Canonical synthetic MECHA experiment.

Run with::

    python -m atlas_interop demo --out-dir demo_runs/mecha --workflows mecha

Design (fixed; see synthetic_mecha.py for the generative model)
---------------------------------------------------------------
* Cohorts (subject-disjoint): mapping reference 20, meta-training 10,000,
  validation 1,000, support pool 1,000, test 2,000.
* Atlases: source ``synS`` (36 regions), target ``synT`` (28 regions).
  The mapping is fit from paired reference time series only.
* Meta-training and validation connectomes are source-atlas FC transformed
  into the target atlas. Support and test connectomes are native target FC.
* One meta-training DNN checkpoint, trained on 16 meta-training phenotypes
  and reused for every K and support draw. By design of the generator, these
  phenotypes share connectome-predictable latent factors with the 10 held-out
  phenotypes.
* 10 held-out phenotypes, K = 10, 20, 40, 100, 10 predetermined draw seeds;
  each seed is one permutation of the support pool and K takes its first K IDs.
* Support-only Ridge and DNN are fit on the identical draws and test cohort.

Every step calls a public CLI command. The configuration and the planned
comparisons are written to ``experiment_config.json`` before any held-out
score exists.
"""

from __future__ import annotations

import json
import os
import platform
import time

import numpy as np
import pandas as pd

from . import __version__
from . import data as D
from . import synthetic_mecha as SM
from .presets import MECHA_PRESETS, SYNTHETIC_MECHA_DRAW_SEEDS, SYNTHETIC_MECHA_TRAIN_SEED

PRESET = "synthetic"
RANK = 20
GENERATOR_SEED = 20261120


def _config(out: str) -> dict:
    p = MECHA_PRESETS[PRESET]
    return dict(
        experiment="canonical synthetic MECHA example", toolkit_version=__version__,
        generator_seed=GENERATOR_SEED, cohorts=dict(SM.COHORTS),
        atlases=dict(SM.ATLASES), source_atlas=SM.SOURCE, target_atlas=SM.TARGET, spectral_rank=RANK,
        meta_training_phenotypes=SM.TRAIN_COLUMNS, heldout_phenotypes=SM.HELDOUT_COLUMNS,
        k_list=list(p.k_list), draw_seeds=list(SYNTHETIC_MECHA_DRAW_SEEDS), dnn_train_seed=SYNTHETIC_MECHA_TRAIN_SEED,
        adaptation_seed=0, preset=p.as_dict(),
        planned_comparisons=[
            "per K: MECHA vs support-only Ridge and vs support-only DNN, paired over the 10 draws "
            "(mean over the 10 held-out phenotypes per draw), one-sided Wilcoxon on Fisher-z, BH over K "
            "within each baseline",
        ],
        caveats=["The 10 support draws share one test cohort; they are not independent cohort replications.",
                 "Synthetic data and synthetic-specific settings; not manuscript results."],
        resources_estimate=SM.estimate_resources(),
        environment=dict(python=platform.python_version(), platform=platform.platform()),
        output_dir=out,
    )


def _timed(cli, args, times, label):
    t0 = time.time()
    cli(args)
    times[label] = round(time.time() - t0, 1)


def run_experiment(out: str, cli, check, counts: dict | None = None, preset: str | None = None,
                   draw_seeds=None, rank: int | None = None) -> dict:
    """Run the experiment under ``out`` using ``cli(args)`` for every step.

    ``counts``/``preset``/``draw_seeds``/``rank`` override the canonical design
    for the developer preflight only; the reviewer command uses the defaults.
    """
    os.makedirs(out, exist_ok=True)
    cfg = _config(out)
    canonical = counts is None and preset is None and draw_seeds is None and rank is None
    if not canonical:
        cfg.update(experiment="developer preflight (NOT the canonical experiment)")
    counts = counts or dict(SM.COHORTS)
    preset_name = preset or PRESET
    draw_seeds = list(draw_seeds or SYNTHETIC_MECHA_DRAW_SEEDS)
    rank = rank or RANK
    p = MECHA_PRESETS[preset_name]
    cfg.update(cohorts=counts, preset=p.as_dict(), draw_seeds=draw_seeds, spectral_rank=rank, k_list=list(p.k_list))
    D.write_json(os.path.join(out, "experiment_config.json"), cfg)       # before any held-out score
    times = {}

    # 1. data
    t0 = time.time()
    data = os.path.join(out, "data")
    man = SM.generate(data, seed=GENERATOR_SEED, counts=counts)
    times["generate_data"] = round(time.time() - t0, 1)
    f = {k: os.path.join(data, k) for k in man["files"]}
    phen = f["phenotypes.csv"]
    src_fc = {r: f[f"{r}_fc_{SM.SOURCE}.npz"] for r in ("meta_training", "validation")}
    tgt_fc = {r: f[f"{r}_fc_{SM.TARGET}.npz"] for r in ("support", "test")}

    # 2. mapping from paired reference time series only
    mapping = os.path.join(out, "mapping", f"{SM.SOURCE}_to_{SM.TARGET}.npz")
    _timed(cli, ["fit-mapping", "--source-timeseries", f[f"reference/timeseries_{SM.SOURCE}.npz"],
                 "--target-timeseries", f[f"reference/timeseries_{SM.TARGET}.npz"],
                 "--source-atlas", SM.SOURCE, "--target-atlas", SM.TARGET, "--out", mapping, "--quiet",
                 "--reference-ids-out", os.path.join(out, "mapping", "participant_level", "reference_ids.txt")],
           times, "fit_mapping")

    # 3. nested support draws
    splits = os.path.join(out, "splits", "participant_level")
    _timed(cli, ["mecha-make-splits", "--support-connectomes", tgt_fc["support"],
                 "--k-list", ",".join(map(str, p.k_list)), "--draw-seeds", ",".join(map(str, draw_seeds)),
                 "--out-dir", splits], times, "make_splits")

    # 4. one meta-training DNN
    tr = os.path.join(out, "train")
    _timed(cli, ["mecha-train", "--train-connectomes", src_fc["meta_training"],
                 "--val-connectomes", src_fc["validation"], "--mapping", mapping, "--rank", str(rank),
                 "--phenotypes", phen, "--train-phenotypes", ",".join(SM.TRAIN_COLUMNS), "--preset", preset_name,
                 "--seed", str(SYNTHETIC_MECHA_TRAIN_SEED), "--out-dir", tr], times, "train")

    # 5. MECHA adaptation and support-only baselines on the same draws
    common = ["--support-connectomes", tgt_fc["support"], "--test-connectomes", tgt_fc["test"],
              "--phenotypes", phen, "--test-phenotypes", ",".join(SM.HELDOUT_COLUMNS), "--preset", preset_name,
              "--splits-dir", splits, "--k-list", ",".join(map(str, p.k_list)),
              "--n-repeats", str(len(draw_seeds)), "--seed", "0", "--modality", "fc",
              "--source-label", SM.SOURCE, "--target-label", SM.TARGET]
    _timed(cli, ["mecha-adapt", "--checkpoint", os.path.join(tr, "mecha_dnn.pt"), *common, "--baselines", "ridge,dnn",
                 "--meta-training-ids", os.path.join(tr, "participant_level", "meta_training_ids.txt"),
                 "--out-dir", os.path.join(out, "adapt")], times, "adapt_and_baselines")

    # 6. matched comparison
    _timed(cli, ["mecha-compare", "--per-phenotype", os.path.join(out, "adapt", "per_phenotype.csv"),
                 "--out-dir", os.path.join(out, "results")], times, "compare")

    res = verify_and_summarize(out, cfg, man, check)
    res["runtimes_seconds"] = times
    res["runtime_total_seconds"] = round(sum(times.values()), 1)
    D.write_json(os.path.join(out, "results", "experiment_report.json"), res)
    _write_markdown(out, res)
    return dict(summary_table=os.path.join(out, "results", "summary.md"), technically_valid=res["technically_valid"],
                runtime_total_seconds=res["runtime_total_seconds"])


def verify_and_summarize(out: str, cfg: dict, man: dict, check) -> dict:
    import torch
    counts = cfg["cohorts"]
    report = dict(checks=[])

    def ok(cond, msg):
        check(cond, msg)
        report["checks"].append(msg)

    # realized cohorts after loading
    data = os.path.join(out, "data")
    realized = {}
    for role in ("meta_training", "validation"):
        realized[role] = D.load_connectomes(os.path.join(data, f"{role}_fc_{SM.SOURCE}.npz")).n_samples
    for role in ("support", "test"):
        realized[role] = D.load_connectomes(os.path.join(data, f"{role}_fc_{SM.TARGET}.npz")).n_samples
    realized["reference"] = D.load_timeseries(os.path.join(data, f"reference/timeseries_{SM.SOURCE}.npz")).n_samples
    ok(realized == {k: counts[k] for k in realized}, f"realized cohort sizes equal planned sizes {counts}")

    tr = os.path.join(out, "train")
    ad = os.path.join(out, "adapt")
    five = dict(
        reference=D.load_id_list(os.path.join(out, "mapping", "participant_level", "reference_ids.txt")),
        meta_training=D.load_id_list(os.path.join(tr, "participant_level", "meta_training_ids.txt")),
        validation=D.load_id_list(os.path.join(tr, "participant_level", "validation_ids.txt")),
        support=D.load_connectomes(os.path.join(data, f"support_fc_{SM.TARGET}.npz")).subject_ids,
        test=D.load_id_list(os.path.join(ad, "participant_level", "test_ids.txt")))
    D.assert_disjoint_subjects(five)        # raises on any overlap
    ok(all(len(v) == counts[k] for k, v in five.items()),
       "reference, meta-training, validation, support pool and test cohorts are subject-disjoint with planned sizes")

    # mapping
    from .transform import load_mapping
    m = load_mapping(os.path.join(out, "mapping", f"{SM.SOURCE}_to_{SM.TARGET}.npz"))
    ok(m.T.shape == (SM.ATLASES[SM.TARGET], SM.ATLASES[SM.SOURCE]) and (m.T >= 0).all()
       and np.allclose(m.T.sum(0), 1), "mapping is target-by-source, non-negative, source columns sum to one")

    # checkpoint: reload through the public loader and check its contents
    from .mecha import load_checkpoint
    model, ck, _ = load_checkpoint(os.path.join(tr, "mecha_dnn.pt"), device="cpu")
    ok(ck["train_phenotypes"] == SM.TRAIN_COLUMNS and ck["spec"]["output_size"] == len(SM.TRAIN_COLUMNS),
       f"checkpoint reloads; output order equals the {len(SM.TRAIN_COLUMNS)} meta-training phenotypes")
    ok(ck["source_atlas"] == SM.SOURCE and ck["target_atlas"] == SM.TARGET, "checkpoint atlases")
    ok(not set(ck["train_phenotypes"]) & set(SM.HELDOUT_COLUMNS), "meta-training and held-out phenotypes are distinct")
    hist = pd.read_csv(os.path.join(tr, "training_history.csv"))
    ok(hist["train_loss"].iloc[-1] < hist["train_loss"].iloc[0], "training loss decreased")
    vm = pd.read_csv(os.path.join(tr, "validation_metrics.csv"))
    with open(os.path.join(tr, "train_run.json")) as fh:
        run = json.load(fh)
    ok(int(hist["val_loss"].idxmin()) == run["best_epoch"], "saved checkpoint is the lowest-validation-loss epoch")
    val = dict(validation_r_mean=float(vm["validation_r"].mean()), validation_r_min=float(vm["validation_r"].min()),
               validation_r_max=float(vm["validation_r"].max()), best_epoch=run["best_epoch"],
               epochs_run=run["epochs_run"], best_val_loss=run["best_val_loss"],
               n_meta_training=run["cohorts"][0]["n_used"], n_validation=run["cohorts"][1]["n_used"])
    report["validation"] = val
    with open(os.path.join(ad, "adapt_run.json")) as fh:
        adr = json.load(fh)
    ok(adr["include_mecha"] and adr["baselines"] == ["ridge", "dnn"], "adaptation ran MECHA, Ridge and DNN")

    # split coverage and matching
    k_list, draws = cfg["k_list"], len(cfg["draw_seeds"])
    sm = pd.read_csv(os.path.join(out, "splits", "participant_level", "splits_manifest.csv"))
    draws_csv = pd.read_csv(os.path.join(ad, "participant_level", "support_draws.csv"), dtype={"subject_id": str})
    sizes = draws_csv.groupby(["k", "repeat"]).size()
    ok(all(sizes.loc[(k, r)] == k for k in k_list for r in range(draws)), "every support draw has exactly K IDs")
    nested = all(set(draws_csv[(draws_csv.k == a) & (draws_csv.repeat == r)].subject_id)
                 <= set(draws_csv[(draws_csv.k == b) & (draws_csv.repeat == r)].subject_id)
                 for r in range(draws) for a, b in zip(k_list, k_list[1:]))
    ok(nested, "support draws are nested prefixes within each draw seed")
    ok(set(draws_csv.subject_id) <= set(five["support"]) and not (set(draws_csv.subject_id) & set(five["test"])),
       "support draws come from the support pool only; none is in the test cohort")
    per = pd.read_csv(os.path.join(ad, "per_phenotype.csv"))
    ok(set(per.method) == {"MECHA", "Ridge", "DNN"}, "per-phenotype file has MECHA, Ridge and DNN rows")
    ok((per.loc[per.method == "DNN", "update_steps"] > 0).all(), "support-only DNN took optimizer steps")
    expected_fp = {(int(r.k), int(r.repeat)): r.support_fingerprint for r in sm.itertuples()}
    ok(all(expected_fp[(int(r.k), int(r.repeat))] == r.support_fingerprint for r in per.itertuples()),
       "support fingerprints equal the pre-written split files")
    ok(per.test_fingerprint.nunique() == 1 and per.test_fingerprint.iloc[0] == D.fingerprint(five["test"]),
       "one test cohort for every method, K, draw and phenotype")
    same = per.groupby(["k", "repeat", "phenotype"])[["support_fingerprint", "test_fingerprint"]].nunique()
    ok(bool((same == 1).all().all()), "MECHA, Ridge and DNN share support and test IDs for each K, draw and phenotype")
    ok((per.n_support_used == per.n_support_requested).all() and (per.n_support_requested == per.k).all()
       and (per.n_test_used == counts["test"]).all(), "realized support size equals K and test size equals plan")
    grid = per.groupby(["method", "k", "repeat"]).phenotype.nunique()
    ok(len(grid) == 3 * len(k_list) * draws and (grid == len(SM.HELDOUT_COLUMNS)).all(),
       "every method x K x draw x held-out phenotype is present")
    ok(np.isfinite(per.r).all(), "all test correlations are finite")

    # comparison outputs
    res_dir = os.path.join(out, "results")
    summ = pd.read_csv(os.path.join(res_dir, "summary.csv"))
    ok(len(summ) == len(k_list) and (summ.n_draws == draws).all()
       and (summ.n_phenotypes == len(SM.HELDOUT_COLUMNS)).all(), "summary covers every K with all draws")
    report["technically_valid"] = True
    report["realized_counts"] = realized
    report["cohort_fingerprints"] = man["cohort_fingerprints"]
    report["test_fingerprint"] = per.test_fingerprint.iloc[0]
    report["support_fingerprints"] = sm.to_dict(orient="records")
    report["summary"] = summ.round(5).to_dict(orient="records")
    report["config"] = cfg
    return report


def _fmt(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def _write_markdown(out: str, res: dict) -> None:
    v = res["validation"]
    lines = ["# Synthetic MECHA experiment: results", "",
             "Synthetic data; not manuscript results. The meta-training and held-out phenotypes share "
             "connectome-predictable latent factors by design of the generator. The 10 support draws share "
             "one test cohort and are not independent cohort replications. Values are means over draws of the "
             "per-draw mean Pearson r across the 10 held-out phenotypes; ± is the SD across draws. `wins` counts "
             "draws (of 10) where MECHA's draw mean exceeds the baseline's. q: one-sided Wilcoxon on Fisher-z, "
             "BH across K within each baseline.", "",
             "## Meta-training DNN (validation cohort)", "",
             "| n meta-training | n validation | best epoch / run | validation r (mean, min–max over 16 phenotypes) |",
             "|---|---|---|---|",
             f"| {v['n_meta_training']} | {v['n_validation']} | {v['best_epoch']} / {v['epochs_run']} | "
             f"{_fmt(v['validation_r_mean'])} ({_fmt(v['validation_r_min'])}–{_fmt(v['validation_r_max'])}) |",
             "", "## Held-out test performance", "",
             "| K | MECHA | Ridge | DNN | MECHA − Ridge | wins | q | MECHA − DNN | wins | q |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["summary"]:
        lines.append(
            f"| {r['k']} | {_fmt(r['r_MECHA_mean'])} ± {_fmt(r['r_MECHA_sd'])} | "
            f"{_fmt(r['r_Ridge_mean'])} ± {_fmt(r['r_Ridge_sd'])} | {_fmt(r['r_DNN_mean'])} ± {_fmt(r['r_DNN_sd'])} | "
            f"{_fmt(r['diff_vs_Ridge_mean'])} ± {_fmt(r['diff_vs_Ridge_sd'])} | {r['draws_MECHA_better_than_Ridge']} | "
            f"{_fmt(r['q_bh_vs_Ridge'], 4)} | {_fmt(r['diff_vs_DNN_mean'])} ± {_fmt(r['diff_vs_DNN_sd'])} | "
            f"{r['draws_MECHA_better_than_DNN']} | {_fmt(r['q_bh_vs_DNN'], 4)} |")
    lines += ["", f"Total runtime: {res['runtime_total_seconds']} s. Step runtimes (s): "
              + ", ".join(f"{k} {v}" for k, v in res["runtimes_seconds"].items()), ""]
    with open(os.path.join(out, "results", "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
