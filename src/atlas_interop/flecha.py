"""FLECHA: federated learning across sites in heterogeneous atlas spaces.

Manuscript reference: Methods, "Federated learning with heterogeneous atlas
spaces"; Fig. 3; Supplementary Fig. 1; Supplementary Table 1.

One training backend (:func:`fedavg_train`, :func:`train_single_site`) serves
two entry paths:

* :func:`run_sites` - the public workflow. The caller supplies already
  partitioned training sites (each in its own native atlas, with a mapping to
  the common target atlas when needed) and a separate held-out test site in
  the native target atlas.
* :func:`run_simulated` - the manuscript experiment. It builds seeded
  simulated-site partitions of one multi-atlas cohort and passes the same
  site objects to the same backend for FLECHA, Single-site and Uniform.

The backend runs every client in one process. It needs read access to every
site's data and is a simulation of federated optimization, not a
cross-institution deployment, secure-aggregation system, or privacy guarantee.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import data as D
from .models import MLPRegressor, MLPSpec, get_device, predict, seed_everything
from .presets import FlechaPreset
from .stats import bh_fdr, fisher_mean_r, pearson, wilcoxon_greater
from .transform import AtlasMapping, apply_mapping, load_mapping


class FlechaInputError(D.DataContractError):
    pass


# ---------------------------------------------------------------------------
# Site preparation
# ---------------------------------------------------------------------------
@dataclass
class Site:
    name: str
    X: np.ndarray                 # (n, n_edges) target-atlas edge features
    y: np.ndarray                 # (n,)
    subject_ids: np.ndarray
    keys: np.ndarray
    native_atlas: str
    transformed: bool
    report: dict = field(default_factory=dict)


def prepare_site(name: str, conn: D.ConnectomeSet, labels: np.ndarray, *, target_atlas: str,
                 n_target_regions: int, mapping: AtlasMapping | None, rank: int | None,
                 target_region_ids=None, n_requested: int | None = None) -> Site:
    """Drop missing labels, transform into the target atlas if needed, vectorize."""
    labels = np.asarray(labels, dtype=float).ravel()
    if len(labels) != conn.n_samples:
        raise FlechaInputError(f"site '{name}': {len(labels)} labels for {conn.n_samples} connectomes")
    keep = np.isfinite(labels)
    report = dict(site=name, native_atlas=conn.atlas, n_requested=int(n_requested or conn.n_samples),
                  n_loaded=int(conn.n_samples), n_missing_label=int((~keep).sum()))
    conn = conn.take(np.flatnonzero(keep))
    labels = labels[keep]
    native = conn.atlas
    if native == target_atlas and mapping is None:
        D.check_atlas(conn, expected_atlas=target_atlas, n_regions=n_target_regions,
                      region_ids=target_region_ids, context=f"site '{name}'")
        transformed, mapped = False, conn
        report["transform"] = "none (native target atlas)"
    else:
        if mapping is None:
            raise FlechaInputError(
                f"site '{name}' is in atlas '{native}' but the target atlas is '{target_atlas}'; "
                "supply a mapping for this site."
            )
        if mapping.target_atlas != target_atlas:
            raise FlechaInputError(
                f"site '{name}': mapping targets '{mapping.target_atlas}', not '{target_atlas}'."
            )
        mapped, tr = apply_mapping(conn, mapping, rank)
        transformed = True
        report["transform"] = f"{mapping.source_atlas} -> {mapping.target_atlas} (rank {rank})"
        report["transform_report"] = tr
    if mapped.n_regions != n_target_regions:
        raise FlechaInputError(f"site '{name}': {mapped.n_regions} regions after transform; "
                               f"expected {n_target_regions}.")
    report["n_used"] = int(mapped.n_samples)
    return Site(name, mapped.edges().astype(np.float32), labels.astype(np.float32),
                np.asarray(conn.subject_ids), np.asarray(conn.keys), native or "", transformed, report)


def check_trainable(site: Site, batch_size: int) -> None:
    if len(site.y) == 0:
        raise FlechaInputError(f"site '{site.name}' has no samples with labels.")
    if len(site.y) < max(2, batch_size):
        raise FlechaInputError(
            f"site '{site.name}' has {len(site.y)} labelled samples, fewer than the batch size "
            f"({batch_size}). Minibatches are drawn with drop_last=True, so this site would never "
            "produce an update. Use a smaller --batch-size or merge sites."
        )


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------
class _Stream:
    """Endless shuffled minibatch stream (drop_last semantics)."""

    def __init__(self, X, y, batch_size, rng):
        self.X, self.y, self.bs, self.rng = X, y, batch_size, rng
        self._new_epoch()

    def _new_epoch(self):
        self.order = self.rng.permutation(len(self.y))
        self.pos = 0

    def next(self):
        if self.pos + self.bs > len(self.order):
            self._new_epoch()
        idx = self.order[self.pos:self.pos + self.bs]
        self.pos += self.bs
        return self.X[idx], self.y[idx]


def _spec(preset: FlechaPreset, input_size: int) -> MLPSpec:
    return MLPSpec(input_size, list(preset.hidden), 1, preset.dropout, preset.norm)


def _make_opt(model, preset):
    return torch.optim.Adam(model.parameters(), lr=preset.learning_rate, weight_decay=preset.weight_decay)


def _scale(preset, y_train_all):
    if preset.target_scaling == "zscore":
        mu, sd = float(np.mean(y_train_all)), float(np.std(y_train_all))
        return mu, (sd if sd > 0 else 1.0)
    return 0.0, 1.0


def average_state_dicts(states: list[dict]) -> dict:
    """Equal-weight average of floating-point tensors; integer buffers from client 0."""
    out = {}
    for key in states[0]:
        t0 = states[0][key]
        if t0.dtype.is_floating_point:
            out[key] = sum(s[key] for s in states) / len(states)
        else:
            out[key] = t0.clone()
    return out


def fedavg_train(sites: list[Site], preset: FlechaPreset, seed: int, device) -> tuple[nn.Module, dict]:
    """FedAvg with equal client weights.

    Each round: every client takes ``local_steps`` minibatch Adam updates from
    the current global weights; the server averages client parameters with
    equal weights and redistributes them.
    """
    for s in sites:
        check_trainable(s, preset.batch_size)
    seed_everything(seed)
    input_size = sites[0].X.shape[1]
    for s in sites:
        if s.X.shape[1] != input_size:
            raise FlechaInputError("all sites must have the same target-atlas feature dimension")
    mu, sd = _scale(preset, np.concatenate([s.y for s in sites]))
    spec = _spec(preset, input_size)
    global_model = MLPRegressor(spec).to(device)
    if preset.client_init == "shared":
        clients = [copy.deepcopy(global_model) for _ in sites]
    elif preset.client_init == "independent":
        clients = [MLPRegressor(spec).to(device) for _ in sites]
    else:
        raise ValueError("client_init must be 'shared' or 'independent'")
    opts = [_make_opt(m, preset) for m in clients]
    rng = np.random.default_rng(seed)
    streams = [_Stream(s.X, (s.y - mu) / sd, preset.batch_size, rng) for s in sites]
    loss_fn = nn.MSELoss()
    steps = np.zeros(len(sites), dtype=int)
    for _ in range(preset.rounds):
        for c, (model, stream) in enumerate(zip(clients, streams)):
            if preset.client_optimizer == "reset":
                opts[c] = _make_opt(model, preset)
            model.train()
            for _ in range(preset.local_steps):
                xb, yb = stream.next()
                xb = torch.as_tensor(xb, device=device)
                yb = torch.as_tensor(yb, device=device).view(-1, 1)
                opts[c].zero_grad()
                loss_fn(model(xb), yb).backward()
                opts[c].step()
                steps[c] += 1
        avg = average_state_dicts([m.state_dict() for m in clients])
        global_model.load_state_dict(avg)
        for m in clients:
            m.load_state_dict(avg)
    info = dict(n_clients=len(sites), rounds=preset.rounds, local_steps=preset.local_steps,
                client_update_steps=steps.tolist(),
                samples_processed=int(steps.sum() * preset.batch_size), target_mean=mu, target_sd=sd)
    return _Scaled(global_model, mu, sd), info


class _Scaled(nn.Module):
    def __init__(self, model, mu, sd):
        super().__init__()
        self.model, self.mu, self.sd = model, mu, sd
        self.spec = model.spec

    def forward(self, x):
        return self.model(x) * self.sd + self.mu


def matched_epochs(n_site: int, n_clients: int, preset: FlechaPreset) -> int:
    """Single-site epochs approximating FLECHA's total sample budget, clipped to bounds.

    FLECHA processes about rounds * local_steps * n_clients * batch_size samples.
    """
    budget = preset.rounds * preset.local_steps * n_clients * preset.batch_size
    return int(np.clip(int(np.ceil(budget / max(1, n_site))), preset.epochs_min, preset.epochs_max))


def train_single_site(site: Site, preset: FlechaPreset, epochs: int, seed: int, device) -> tuple[nn.Module, dict]:
    check_trainable(site, preset.batch_size)
    seed_everything(seed)
    mu, sd = _scale(preset, site.y)
    model = MLPRegressor(_spec(preset, site.X.shape[1])).to(device)
    opt = _make_opt(model, preset)
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(seed)
    y = (site.y - mu) / sd
    n_steps = 0
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(order) - preset.batch_size + 1, preset.batch_size):
            idx = order[start:start + preset.batch_size]
            xb = torch.as_tensor(site.X[idx], device=device)
            yb = torch.as_tensor(y[idx], device=device).view(-1, 1)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
            n_steps += 1
    if n_steps == 0:
        raise FlechaInputError(f"single-site '{site.name}': no optimizer steps were taken")
    return _Scaled(model, mu, sd), dict(epochs=int(epochs), update_steps=int(n_steps),
                                        samples_processed=int(n_steps * preset.batch_size))


def evaluate(model, test: Site, device) -> tuple[dict, np.ndarray]:
    pred = predict(model, test.X, device).ravel()
    r, p = pearson(test.y, pred)
    mse = float(np.mean((test.y - pred) ** 2))
    return dict(r=r, p=p, mse=mse, n_test=int(len(test.y))), pred


# ---------------------------------------------------------------------------
# Public workflow: explicit sites
# ---------------------------------------------------------------------------
def load_site_config(path: str) -> dict:
    """Read and validate a FLECHA site configuration (JSON).

    Schema::

        {
          "target_atlas": "atlasT",
          "phenotype": "age",
          "rank": 50,                          # optional; default from --rank
          "train_sites": [
            {"name": "siteA", "connectomes": "a.npz", "labels": "a.csv",
             "mapping": "atlasA_to_atlasT.npz"},
            {"name": "siteB", "connectomes": "b.npz", "labels": "b.csv"}  # native target atlas
          ],
          "test_site": {"name": "test", "connectomes": "t.npz", "labels": "t.csv"}
        }

    Relative paths are resolved against the configuration file's directory.
    """
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    base = os.path.dirname(os.path.abspath(path))
    for key in ("target_atlas", "phenotype", "train_sites", "test_site"):
        if key not in cfg:
            raise FlechaInputError(f"{path}: missing '{key}'")
    if not isinstance(cfg["train_sites"], list) or len(cfg["train_sites"]) < 1:
        raise FlechaInputError(f"{path}: 'train_sites' must be a non-empty list")
    names = [s.get("name") for s in cfg["train_sites"]] + [cfg["test_site"].get("name")]
    if any(not n for n in names) or len(set(names)) != len(names):
        raise FlechaInputError(f"{path}: every site needs a unique 'name'")
    for s in cfg["train_sites"] + [cfg["test_site"]]:
        for k in ("connectomes", "labels"):
            if not s.get(k):
                raise FlechaInputError(f"{path}: site '{s['name']}' needs '{k}'")
            s[k] = D.resolve_path(base, s[k])
        if s.get("mapping"):
            s["mapping"] = D.resolve_path(base, s["mapping"])
    if cfg["test_site"].get("mapping"):
        raise FlechaInputError(f"{path}: the test site must be native in the target atlas (no mapping)")
    return cfg


def _load_site(spec: dict, phenotype: str, target_atlas: str, n_target: int | None,
               rank: int, target_region_ids) -> tuple[Site, int]:
    conn = D.load_connectomes(spec["connectomes"])
    table = D.load_phenotypes(spec["labels"], [phenotype])
    y = D.phenotypes_for(conn, table, [phenotype])[:, 0]
    mapping = load_mapping(spec["mapping"]) if spec.get("mapping") else None
    if mapping is not None and conn.atlas is not None and conn.atlas != mapping.source_atlas:
        raise FlechaInputError(f"site '{spec['name']}': connectomes are '{conn.atlas}', mapping source "
                               f"is '{mapping.source_atlas}'")
    if n_target is None:
        n_target = mapping.n_target if mapping is not None else conn.n_regions
    site = prepare_site(spec["name"], conn, y, target_atlas=target_atlas, n_target_regions=n_target,
                        mapping=mapping, rank=rank, target_region_ids=target_region_ids)
    return site, n_target


def run_sites(cfg: dict, preset: FlechaPreset, *, rank: int, seed: int, out_dir: str,
              device: str = "auto", single_site: bool = False) -> dict:
    """Train FLECHA on explicitly supplied sites and evaluate on the test site."""
    dev = get_device(device)
    target_atlas, phen = cfg["target_atlas"], cfg["phenotype"]
    rank = int(cfg.get("rank", rank))
    test_conn = D.load_connectomes(cfg["test_site"]["connectomes"])
    if test_conn.atlas is not None and test_conn.atlas != target_atlas:
        raise FlechaInputError(f"test site is in '{test_conn.atlas}'; it must be native '{target_atlas}'")
    t_table = D.load_phenotypes(cfg["test_site"]["labels"], [phen])
    test = prepare_site(cfg["test_site"]["name"], test_conn, D.phenotypes_for(test_conn, t_table, [phen])[:, 0],
                        target_atlas=target_atlas, n_target_regions=test_conn.n_regions, mapping=None,
                        rank=None)
    sites = []
    for spec in cfg["train_sites"]:
        s, _ = _load_site(spec, phen, target_atlas, test_conn.n_regions, rank, test_conn.region_ids)
        sites.append(s)
    D.assert_disjoint_subjects({s.name: s.subject_ids for s in sites + [test]})
    model, info = fedavg_train(sites, preset, seed, dev)
    metrics, pred = evaluate(model, test, dev)
    rows = [dict(method="FLECHA", seed=seed, **metrics, n_clients=len(sites),
                 train_sites=";".join(s.name for s in sites))]
    if single_site:
        for s in sites:
            ep = matched_epochs(len(s.y), len(sites), preset)
            m, si = train_single_site(s, preset, ep, seed, dev)
            met, _ = evaluate(m, test, dev)
            rows.append(dict(method="Single-site", seed=seed, **met, train_sites=s.name,
                             matched_epochs=ep, update_steps=si["update_steps"]))
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "results.csv"), index=False)
    pdir = D.participant_dir(out_dir)
    # one row per input test sample, in input order; excluded samples have no prediction
    t_all = pd.DataFrame({"subject_id": test_conn.subject_ids})
    scored = pd.DataFrame({"subject_id": test.subject_ids, "observed": test.y, "predicted_flecha": pred})
    t_all.merge(scored, on="subject_id", how="left", validate="one_to_one").to_csv(
        os.path.join(pdir, "test_predictions.csv"), index=False)
    run = dict(workflow="flecha-sites", preset=preset.as_dict(), rank=rank, seed=seed,
               target_atlas=target_atlas, phenotype=phen, device=str(dev), training=info,
               sites=[s.report for s in sites], test_site=test.report,
               notice=("Single-process FedAvg simulation over the supplied site files; "
                       "not a cross-institution deployment or a privacy guarantee."))
    D.write_json(os.path.join(out_dir, "run.json"), run)
    return dict(rows=rows, run=run)


# ---------------------------------------------------------------------------
# Manuscript experiment: simulated sites
# ---------------------------------------------------------------------------
def load_simulation_config(path: str) -> dict:
    """Read a simulated-site configuration (JSON).

    Schema::

        {
          "phenotype": "age",
          "labels": "phenotypes.csv",
          "atlases": {"atlasA": "fc_atlasA.npz", "atlasB": "fc_atlasB.npz", ...},
          "mappings": {"atlasA->atlasB": "atlasA_to_atlasB.npz", ...},
          "target_atlases": ["atlasA", "atlasB"],      # optional; default: all atlases
          "exclude_subjects": "reference_ids.txt"       # optional
        }

    Every atlas file must contain the same participants (matched by ID). The
    number of simulated sites equals the number of atlases (Methods).
    """
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    base = os.path.dirname(os.path.abspath(path))
    for key in ("phenotype", "labels", "atlases", "mappings"):
        if key not in cfg:
            raise FlechaInputError(f"{path}: missing '{key}'")
    if len(cfg["atlases"]) < 2:
        raise FlechaInputError(f"{path}: at least two atlases are required")
    cfg["labels"] = D.resolve_path(base, cfg["labels"])
    cfg["atlases"] = {a: D.resolve_path(base, p) for a, p in cfg["atlases"].items()}
    cfg["mappings"] = {k: D.resolve_path(base, p) for k, p in cfg["mappings"].items()}
    if cfg.get("exclude_subjects"):
        cfg["exclude_subjects"] = D.resolve_path(base, cfg["exclude_subjects"])
    cfg.setdefault("target_atlases", list(cfg["atlases"]))
    unknown = [t for t in cfg["target_atlases"] if t not in cfg["atlases"]]
    if unknown:
        raise FlechaInputError(f"{path}: target atlases not in 'atlases': {unknown}")
    return cfg


def partition_sites(keys: np.ndarray, subject_ids: np.ndarray, n_sites: int, seed: int) -> list[np.ndarray]:
    """Random subject-level partition into ``n_sites`` groups of sample indices.

    All sessions of one participant go to the same site.
    """
    subjects = np.array(sorted(set(subject_ids.tolist())), dtype=str)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(subjects))
    groups = np.array_split(subjects[perm], n_sites)
    lookup = {}
    for g, members in enumerate(groups):
        for s in members:
            lookup[s] = g
    site_of = np.array([lookup[s] for s in subject_ids])
    return [np.flatnonzero(site_of == g) for g in range(n_sites)]


def _site_assignment(atlases: list[str], target: str) -> tuple[int, list[tuple[int, str]]]:
    """Site 0 is the held-out test site; training sites 1..N-1 use the other atlases."""
    others = [a for a in atlases if a != target]
    return 0, list(enumerate(others, start=1))


def run_simulated(cfg: dict, preset: FlechaPreset, *, rank: int, seeds: list[int], out_dir: str,
                  methods=("FLECHA", "Single-site", "Uniform"), device: str = "auto") -> pd.DataFrame:
    """Manuscript simulated-site experiment (FLECHA vs Single-site vs Uniform).

    For each seed and target atlas, participants are randomly partitioned into
    N = number of atlases sites. Site 0 (native target atlas) is the held-out
    test site. Each remaining site uses one non-target atlas as its native
    atlas. FLECHA and Single-site use those sites' connectomes transformed into
    the target atlas; Uniform uses the *same participants'* native target-atlas
    connectomes. All methods share partitions and test participants.
    """
    dev = get_device(device)
    phen = cfg["phenotype"]
    atlases = list(cfg["atlases"])
    conns = {a: D.load_connectomes(p) for a, p in cfg["atlases"].items()}
    for a, c in conns.items():
        if c.atlas is not None and c.atlas != a:
            raise FlechaInputError(f"{cfg['atlases'][a]}: atlas '{c.atlas}' listed as '{a}'")
    ref = conns[atlases[0]]
    aligned = {atlases[0]: ref}
    for a in atlases[1:]:
        idx = D.align_to(ref.keys, conns[a].keys, ref_name=ref.label(), other_name=conns[a].label(),
                         allow_extra=False)
        aligned[a] = conns[a].take(idx)
    table = D.load_phenotypes(cfg["labels"], [phen])
    y_all = D.phenotypes_for(ref, table, [phen])[:, 0]
    keep = np.isfinite(y_all)
    excluded = []
    if cfg.get("exclude_subjects"):
        excl = set(D.load_id_list(cfg["exclude_subjects"]).tolist())
        overlap = np.array([s in excl for s in ref.subject_ids])
        excluded = sorted(set(ref.subject_ids[overlap].tolist()))
        keep &= ~overlap
    cohort = dict(n_loaded=int(ref.n_samples), n_missing_label=int((~np.isfinite(y_all)).sum()),
                  n_excluded_reference=len(excluded), n_used=int(keep.sum()))
    idx_keep = np.flatnonzero(keep)
    aligned = {a: c.take(idx_keep) for a, c in aligned.items()}
    y_all = y_all[idx_keep]
    ref = aligned[atlases[0]]
    mappings = {}
    rows, membership = [], []
    pdir = D.participant_dir(out_dir)
    for target in cfg["target_atlases"]:
        for src in atlases:
            if src == target:
                continue
            key = f"{src}->{target}"
            if key not in cfg["mappings"]:
                raise FlechaInputError(f"missing mapping '{key}' for target atlas '{target}'")
            if key not in mappings:
                mappings[key] = load_mapping(cfg["mappings"][key])
        test_pos, train_assign = _site_assignment(atlases, target)
        for seed in seeds:
            parts = partition_sites(ref.keys, ref.subject_ids, len(atlases), seed)
            for g, idx in enumerate(parts):
                role = "test" if g == test_pos else "train"
                native = target if g == test_pos else dict(train_assign)[g]
                membership += [dict(target_atlas=target, seed=seed, site=g, role=role, native_atlas=native,
                                    subject_id=s) for s in ref.subject_ids[idx]]
            tconn = aligned[target]
            test = prepare_site("site0", tconn.take(parts[test_pos]), y_all[parts[test_pos]],
                                target_atlas=target, n_target_regions=tconn.n_regions, mapping=None, rank=None)
            hetero, uniform = [], []
            for g, native in train_assign:
                idx = parts[g]
                hetero.append(prepare_site(f"site{g}", aligned[native].take(idx), y_all[idx], target_atlas=target,
                                           n_target_regions=tconn.n_regions, mapping=mappings[f"{native}->{target}"],
                                           rank=rank))
                uniform.append(prepare_site(f"site{g}", tconn.take(idx), y_all[idx], target_atlas=target,
                                            n_target_regions=tconn.n_regions, mapping=None, rank=None))
            D.assert_disjoint_subjects({s.name: s.subject_ids for s in hetero + [test]})
            base = dict(target_atlas=target, seed=seed, n_sites=len(atlases), n_test=int(len(test.y)))
            if "FLECHA" in methods:
                m, info = fedavg_train(hetero, preset, seed, dev)
                met, _ = evaluate(m, test, dev)
                rows.append(dict(base, method="FLECHA", source_atlas=";".join(a for _, a in train_assign),
                                 **met, update_steps=sum(info["client_update_steps"])))
            if "Uniform" in methods:
                m, info = fedavg_train(uniform, preset, seed, dev)
                met, _ = evaluate(m, test, dev)
                rows.append(dict(base, method="Uniform", source_atlas=target, **met,
                                 update_steps=sum(info["client_update_steps"])))
            if "Single-site" in methods:
                for s, (g, native) in zip(hetero, train_assign):
                    ep = matched_epochs(len(s.y), len(hetero), preset)
                    m, info = train_single_site(s, preset, ep, seed, dev)
                    met, _ = evaluate(m, test, dev)
                    rows.append(dict(base, method="Single-site", source_atlas=native, site=s.name, **met,
                                     matched_epochs=ep, update_steps=info["update_steps"]))
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "runs.csv"), index=False)
    pd.DataFrame(membership).to_csv(os.path.join(pdir, "site_membership.csv"), index=False)
    D.write_json(os.path.join(out_dir, "run.json"), dict(
        workflow="flecha-simulate", preset=preset.as_dict(), rank=rank, seeds=list(seeds),
        atlases=atlases, target_atlases=cfg["target_atlases"], phenotype=phen, cohort=cohort,
        methods=list(methods), device=str(dev)))
    return df


def summarize(runs: pd.DataFrame, family_label: str = "") -> pd.DataFrame:
    """Methods comparison summary for each target atlas.

    * Single-site runs with several source sites are combined per seed by
      averaging Fisher-z transformed r and back-transforming.
    * FLECHA - Single-site: mean raw-r difference over matched seeds, with a
      one-sided paired Wilcoxon test (FLECHA > Single-site) and BH correction
      across all target-atlas configurations in ``runs`` (one family).
    * Uniform - FLECHA: descriptive mean raw-r gap.
    """
    out = []
    for target, g in runs.groupby("target_atlas", sort=False):
        per = {}
        for method, gm in g.groupby("method"):
            if method == "Single-site":
                per[method] = gm.groupby("seed")["r"].apply(fisher_mean_r)
            else:
                per[method] = gm.set_index("seed")["r"]
        row = dict(family=family_label, target_atlas=target)
        for m in ("Uniform", "FLECHA", "Single-site"):
            if m in per:
                row[f"r_{m}_mean"] = float(per[m].mean())
                row[f"r_{m}_sd"] = float(per[m].std(ddof=1)) if len(per[m]) > 1 else float("nan")
        if "FLECHA" in per and "Single-site" in per:
            j = pd.concat([per["FLECHA"], per["Single-site"]], axis=1, join="inner").dropna()
            row["delta_FLECHA_minus_SingleSite"] = float((j.iloc[:, 0] - j.iloc[:, 1]).mean())
            w = wilcoxon_greater(j.iloc[:, 0], j.iloc[:, 1])
            row.update(wilcoxon_p=w["p"], wilcoxon_statistic=w["statistic"], n_matched_seeds=w["n_pairs"],
                       test_note=w["note"])
        if "FLECHA" in per and "Uniform" in per:
            j = pd.concat([per["Uniform"], per["FLECHA"]], axis=1, join="inner").dropna()
            row["gap_Uniform_minus_FLECHA"] = float((j.iloc[:, 0] - j.iloc[:, 1]).mean())
        out.append(row)
    df = pd.DataFrame(out)
    if "wilcoxon_p" in df:
        df["q_bh"] = bh_fdr(df["wilcoxon_p"].to_numpy())
    return df
