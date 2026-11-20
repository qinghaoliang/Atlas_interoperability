"""Command-line interface: ``python -m atlas_interop <command> --help``."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from . import __version__
from . import data as D
from . import presets as P


def _rank(args, default_preset="manuscript"):
    return args.rank if args.rank is not None else P.TRANSFORM_RANK[default_preset]


def _set_threads(a):
    n = getattr(a, "threads", None)
    if n:
        import torch
        torch.set_num_threads(int(n))


def _split(s):
    return [x.strip() for x in s.split(",") if x.strip()] if s else []


# ---------------------------------------------------------------------------
def cmd_fit_mapping(a):
    from .transform import FitSettings, fit_mapping, save_mapping
    src = D.load_timeseries(a.source_timeseries)
    tgt = D.load_timeseries(a.target_timeseries)
    s = FitSettings(reg=a.reg, window_size=a.window_size)
    m, ids = fit_mapping(src, tgt, source_atlas=a.source_atlas, target_atlas=a.target_atlas, settings=s,
                         progress=not a.quiet)
    save_mapping(a.out, m)
    if a.reference_ids_out:
        D.save_id_list(a.reference_ids_out, sorted(set(ids.tolist())),
                       header="Mapping reference participants (participant-level; keep local).")
    print(f"saved mapping {a.source_atlas} -> {a.target_atlas}: shape {m.T.shape} "
          f"(target x source) to {a.out}")
    if m.fit_info["sinkhorn_warnings"]:
        print(f"WARNING: Sinkhorn emitted {m.fit_info['sinkhorn_warnings']} convergence warning(s).")


def cmd_apply_mapping(a):
    from .transform import apply_mapping, load_mapping
    m = load_mapping(a.mapping)
    c = D.load_connectomes(a.connectomes, require_symmetric=True)
    out, rep = apply_mapping(c, m, _rank(a), modality=a.modality)
    D.save_connectomes(a.out, out)
    D.write_json(os.path.splitext(a.out)[0] + "_report.json", rep)
    print(f"transformed {rep['n_samples']} connectomes {m.source_atlas} -> {m.target_atlas} (rank {rep['spectral_rank']}) to {a.out}")


def cmd_eval_reconstruction(a):
    from .transform import reconstruction_similarity
    mapped = D.load_connectomes(a.mapped)
    native = D.load_connectomes(a.native_target)
    keys, r = reconstruction_similarity(mapped, native)
    os.makedirs(a.out_dir, exist_ok=True)
    pd.DataFrame({"sample": [D.describe_key(k) for k in keys], "r": r}).to_csv(
        os.path.join(D.participant_dir(a.out_dir), "reconstruction_per_participant.csv"), index=False)
    summ = dict(n=int(len(r)), r_mean=float(np.mean(r)), r_sd=float(np.std(r, ddof=1)) if len(r) > 1 else None,
                r_median=float(np.median(r)), mapped=os.path.basename(a.mapped), native=os.path.basename(a.native_target))
    D.write_json(os.path.join(a.out_dir, "reconstruction_summary.json"), summ)
    print(f"reconstruction similarity: mean r = {summ['r_mean']:.4f} over {summ['n']} participants")


def cmd_flecha_sites(a):
    from . import flecha
    preset = P.override(P.get_preset(P.FLECHA_PRESETS, a.preset), rounds=a.rounds, batch_size=a.batch_size,
                        local_steps=a.local_steps)
    cfg = flecha.load_site_config(a.config)
    res = flecha.run_sites(cfg, preset, rank=_rank(a), seed=a.seed, out_dir=a.out_dir, device=a.device,
                           single_site=a.single_site)
    for r in res["rows"]:
        print(f"{r['method']:<12} r = {r['r']:.4f}  mse = {r['mse']:.4f}  (n_test = {r['n_test']})")


def cmd_flecha_simulate(a):
    from . import flecha
    preset = P.get_preset(P.FLECHA_PRESETS, a.preset)
    cfg = flecha.load_simulation_config(a.config)
    seeds = list(range(a.n_seeds if a.n_seeds is not None else preset.n_seeds))
    runs = flecha.run_simulated(cfg, preset, rank=_rank(a), seeds=seeds, out_dir=a.out_dir, device=a.device,
                                methods=tuple(_split(a.methods)))
    summ = flecha.summarize(runs, family_label=a.family or "")
    summ.to_csv(os.path.join(a.out_dir, "summary.csv"), index=False)
    print(summ.to_string(index=False))


def cmd_flecha_sim5(a):
    from . import flecha_experiment as FX
    maps = dict(item.split("=", 1) for item in a.mapping)
    res = FX.run(a.out_dir, data_dir=a.data_dir, mappings=maps, device=a.device, reference_ids=a.reference_ids)
    ov = FX.overall(res["summary"])
    D.write_json(os.path.join(a.out_dir, "overall.json"), ov)
    print(res["summary"].round(4).to_string(index=False))


def cmd_flecha_summarize(a):
    from . import flecha
    runs = pd.concat([pd.read_csv(p) for p in a.runs], ignore_index=True)
    summ = flecha.summarize(runs, family_label=a.family or "")
    summ.to_csv(a.out, index=False)
    print(summ.to_string(index=False))


def cmd_mecha_train(a):
    from . import mecha
    preset = P.override(P.get_preset(P.MECHA_PRESETS, a.preset), max_epochs=a.max_epochs)
    _set_threads(a)
    run = mecha.train_dnn(train_conn=D.load_connectomes(a.train_connectomes),
                          val_conn=D.load_connectomes(a.val_connectomes),
                          table=D.load_phenotypes(a.phenotypes), train_phenotypes=_split(a.train_phenotypes),
                          mapping_path=a.mapping, rank=_rank(a), preset=preset, seed=a.seed, out_dir=a.out_dir,
                          device=a.device)
    print(f"trained MECHA DNN: best epoch {run['best_epoch']} of {run['epochs_run']}, "
          f"validation loss {run['best_val_loss']:.4f}")


def cmd_mecha_adapt(a):
    from . import mecha
    preset = P.override(P.get_preset(P.MECHA_PRESETS, a.preset),
                        k_list=tuple(int(k) for k in _split(a.k_list)) if a.k_list else None,
                        n_repeats=a.n_repeats)
    _set_threads(a)
    runs = mecha.adapt_and_evaluate(
        checkpoint=a.checkpoint, support_conn=D.load_connectomes(a.support_connectomes),
        test_conn=D.load_connectomes(a.test_connectomes), table=D.load_phenotypes(a.phenotypes),
        test_phenotypes=_split(a.test_phenotypes), preset=preset, seed=a.seed, out_dir=a.out_dir,
        splits_dir=a.splits_dir, baselines=tuple(_split(a.baselines)), device=a.device,
        meta_training_ids=a.meta_training_ids, include_mecha=not a.baselines_only)
    for c, v in (("modality", a.modality), ("source_atlas", a.source_label), ("target_atlas", a.target_label)):
        if v:
            runs[c] = v
    runs.to_csv(os.path.join(a.out_dir, "runs.csv"), index=False)
    summ = mecha.summarize(runs)
    summ.to_csv(os.path.join(a.out_dir, "summary.csv"), index=False)
    print(summ.to_string(index=False))


def cmd_mecha_make_splits(a):
    from . import mecha
    conn = D.load_connectomes(a.support_connectomes)
    seeds = [int(s) for s in _split(a.draw_seeds)]
    tab = mecha.make_nested_supports(conn.subject_ids, [int(k) for k in _split(a.k_list)], seeds, a.out_dir)
    print(f"wrote {len(tab)} nested support splits ({len(seeds)} draws x {tab['k'].nunique()} K) to {a.out_dir}")


def cmd_mecha_compare(a):
    from . import mecha
    res = mecha.compare_matched(pd.read_csv(a.per_phenotype),
                                pd.read_csv(a.baselines) if a.baselines else None)
    os.makedirs(a.out_dir, exist_ok=True)
    res["per_draw"].to_csv(os.path.join(a.out_dir, "per_draw_per_phenotype.csv"), index=False)
    res["draw_means"].to_csv(os.path.join(a.out_dir, "per_draw.csv"), index=False)
    res["summary"].to_csv(os.path.join(a.out_dir, "summary.csv"), index=False)
    cols = ["k", "r_MECHA_mean", "r_Ridge_mean", "r_DNN_mean", "diff_vs_Ridge_mean", "diff_vs_DNN_mean"]
    print(res["summary"][cols].round(4).to_string(index=False))


def cmd_mecha_summarize(a):
    from . import mecha
    runs = pd.concat([pd.read_csv(p) for p in a.runs], ignore_index=True)
    summ = mecha.summarize(runs)
    summ.to_csv(a.out, index=False)
    print(summ.to_string(index=False))


def cmd_evaluate(a):
    from . import evaluation
    with open(a.config, encoding="utf-8") as fh:
        cfg = json.load(fh)
    base = os.path.dirname(os.path.abspath(a.config))
    native = {k: D.load_connectomes(D.resolve_path(base, p)) for k, p in cfg["native"].items()}
    mapped = {}
    for key, p in cfg.get("mapped", {}).items():
        src, tgt = key.split("->")
        mapped[(src.strip(), tgt.strip())] = D.load_connectomes(D.resolve_path(base, p))
    table = D.load_phenotypes(D.resolve_path(base, cfg["phenotypes"])) if cfg.get("phenotypes") else None
    preset = P.get_preset(P.PREDICTION_PRESETS, a.preset)
    res = evaluation.run_evaluation(native=native, mapped=mapped, table=table,
                                    phenotypes=cfg.get("phenotype_columns", []), preset=preset, seed=a.seed,
                                    out_dir=a.out_dir, score=a.score, modality=cfg.get("modality", ""))
    print(res["reconstruction"].to_string(index=False))
    if len(res["prediction"]):
        print(res["prediction"].groupby(["phenotype", "source_atlas", "target_atlas"])["r"].mean().to_string())


def cmd_demo(a):
    from .demo import run_demo
    ok = run_demo(a.out_dir, workflows=_split(a.workflows) or None, force=a.force)
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="atlas_interop", description=__doc__)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    def rank_arg(sp):
        sp.add_argument("--rank", type=int, default=None,
                        help="spectral rank k (default: manuscript value 50)")

    sp = sub.add_parser("fit-mapping", help="fit a directed atlas mapping from paired reference time series")
    sp.add_argument("--source-timeseries", required=True)
    sp.add_argument("--target-timeseries", required=True)
    sp.add_argument("--source-atlas", required=True)
    sp.add_argument("--target-atlas", required=True)
    sp.add_argument("--out", required=True, help="output mapping .npz")
    sp.add_argument("--reg", type=float, default=0.01, help="entropic regularization (manuscript: 0.01)")
    sp.add_argument("--window-size", type=int, default=1,
                    help="time points per temporal window (ASSUMPTION: 1; see implementation notes)")
    sp.add_argument("--reference-ids-out", default=None,
                    help="optional local file listing reference participant IDs (for exclusion checks)")
    sp.add_argument("--quiet", action="store_true")
    sp.set_defaults(func=cmd_fit_mapping)

    sp = sub.add_parser("apply-mapping", help="transform source-atlas FC/SC with a saved mapping")
    sp.add_argument("--mapping", required=True)
    sp.add_argument("--connectomes", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--modality", choices=["fc", "sc"], default=None)
    rank_arg(sp)
    sp.set_defaults(func=cmd_apply_mapping)

    sp = sub.add_parser("eval-reconstruction", help="participant-matched reconstruction similarity")
    sp.add_argument("--mapped", required=True)
    sp.add_argument("--native-target", required=True)
    sp.add_argument("--out-dir", required=True)
    sp.set_defaults(func=cmd_eval_reconstruction)

    sp = sub.add_parser("flecha", help="FLECHA on explicitly supplied sites (public workflow)")
    sp.add_argument("--config", required=True, help="site configuration JSON")
    sp.add_argument("--preset", default="ukb", choices=sorted(P.FLECHA_PRESETS))
    sp.add_argument("--rounds", type=int, default=None)
    sp.add_argument("--local-steps", type=int, default=None)
    sp.add_argument("--batch-size", type=int, default=None)
    sp.add_argument("--single-site", action="store_true", help="also train Single-site baselines")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--device", default="auto")
    sp.add_argument("--out-dir", required=True)
    rank_arg(sp)
    sp.set_defaults(func=cmd_flecha_sites)

    sp = sub.add_parser("flecha-simulate", help="manuscript simulated-site experiment")
    sp.add_argument("--config", required=True)
    sp.add_argument("--preset", default="ukb", choices=sorted(P.FLECHA_PRESETS))
    sp.add_argument("--n-seeds", type=int, default=None, help="default: preset value (manuscript 10)")
    sp.add_argument("--methods", default="FLECHA,Single-site,Uniform")
    sp.add_argument("--family", default=None, help="label for the BH comparison family")
    sp.add_argument("--device", default="auto")
    sp.add_argument("--out-dir", required=True)
    rank_arg(sp)
    sp.set_defaults(func=cmd_flecha_simulate)

    sp = sub.add_parser("flecha-sim5", help="synthetic five-site reviewer experiment (used by the demo)")
    sp.add_argument("--data-dir", required=True, help="output of flecha_experiment.generate")
    sp.add_argument("--mapping", action="append", required=True, help="ATLAS=path to ATLAS->synE mapping")
    sp.add_argument("--reference-ids", default=None, help="mapping-reference ID list; checked disjoint")
    sp.add_argument("--device", default="cpu")
    sp.add_argument("--out-dir", required=True)
    sp.set_defaults(func=cmd_flecha_sim5)

    sp = sub.add_parser("flecha-summarize", help="combine FLECHA runs.csv files into one comparison family")
    sp.add_argument("--runs", nargs="+", required=True)
    sp.add_argument("--family", default=None)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_flecha_summarize)

    sp = sub.add_parser("mecha-train", help="train the MECHA meta-training DNN")
    sp.add_argument("--train-connectomes", required=True)
    sp.add_argument("--val-connectomes", required=True)
    sp.add_argument("--phenotypes", required=True)
    sp.add_argument("--train-phenotypes", required=True, help="comma-separated column names")
    sp.add_argument("--mapping", default=None, help="source->target mapping (omit if already in target atlas)")
    sp.add_argument("--preset", default="ukb", choices=sorted(P.MECHA_PRESETS))
    sp.add_argument("--max-epochs", type=int, default=None)
    sp.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: torch default)")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--device", default="auto")
    sp.add_argument("--out-dir", required=True)
    rank_arg(sp)
    sp.set_defaults(func=cmd_mecha_train)

    sp = sub.add_parser("mecha-adapt", help="MECHA adaptation, baselines and evaluation")
    sp.add_argument("--checkpoint", required=True)
    sp.add_argument("--support-connectomes", required=True)
    sp.add_argument("--test-connectomes", required=True)
    sp.add_argument("--phenotypes", required=True)
    sp.add_argument("--test-phenotypes", required=True, help="comma-separated held-out column names")
    sp.add_argument("--splits-dir", default=None, help="explicit support_k{K}_rep{R}.txt files")
    sp.add_argument("--meta-training-ids", default=None, help="optional ID list to check cohort separation")
    sp.add_argument("--k-list", default=None)
    sp.add_argument("--n-repeats", type=int, default=None)
    sp.add_argument("--baselines", default="ridge,dnn", help="support-only baselines to fit ('' for none)")
    sp.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: torch default)")
    sp.add_argument("--baselines-only", action="store_true",
                    help="fit only the support-only baselines (to share them across MECHA checkpoints)")
    sp.add_argument("--preset", default="ukb", choices=sorted(P.MECHA_PRESETS))
    sp.add_argument("--modality", default=None)
    sp.add_argument("--source-label", default=None)
    sp.add_argument("--target-label", default=None)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--device", default="auto")
    sp.add_argument("--out-dir", required=True)
    sp.set_defaults(func=cmd_mecha_adapt)

    sp = sub.add_parser("mecha-make-splits", help="write nested support draws (one permutation per seed)")
    sp.add_argument("--support-connectomes", required=True)
    sp.add_argument("--k-list", required=True, help="comma-separated support sizes")
    sp.add_argument("--draw-seeds", required=True, help="comma-separated draw seeds, one per repeat")
    sp.add_argument("--out-dir", required=True)
    sp.set_defaults(func=cmd_mecha_make_splits)

    sp = sub.add_parser("mecha-compare", help="matched MECHA vs support-only Ridge/DNN comparison")
    sp.add_argument("--per-phenotype", required=True, help="per_phenotype.csv from mecha-adapt")
    sp.add_argument("--baselines", default=None,
                    help="optional per_phenotype.csv of a separate --baselines-only run")
    sp.add_argument("--out-dir", required=True)
    sp.set_defaults(func=cmd_mecha_compare)

    sp = sub.add_parser("mecha-summarize", help="combine MECHA runs.csv files into one comparison family")
    sp.add_argument("--runs", nargs="+", required=True)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_mecha_summarize)

    sp = sub.add_parser("evaluate", help="reconstruction similarity and native vs cross-atlas ridge prediction")
    sp.add_argument("--config", required=True)
    sp.add_argument("--preset", default="manuscript", choices=sorted(P.PREDICTION_PRESETS))
    sp.add_argument("--score", default="pooled", choices=["pooled", "fold_mean"])
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--out-dir", required=True)
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("demo", help="generate synthetic data and run the synthetic suite",
                        description="Workflows: transform, flecha, evaluate (small shared cohort), flecha-sim "
                                    "(synthetic five-site, five-atlas FLECHA experiment) and mecha (synthetic "
                                    "MECHA experiment, 10,000 meta-training participants). Default: all five.")
    sp.add_argument("--out-dir", required=True, help="new output directory (refuses non-demo directories)")
    sp.add_argument("--workflows", default=None, help="comma-separated subset: transform,flecha,flecha-sim,mecha,evaluate")
    sp.add_argument("--force", action="store_true", help="overwrite an existing demo output directory")
    sp.set_defaults(func=cmd_demo)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except D.DataContractError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
