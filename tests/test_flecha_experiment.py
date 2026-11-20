"""Regression tests for the synthetic five-site FLECHA reviewer experiment."""

import numpy as np
import pytest

from atlas_interop import data as D
from atlas_interop import flecha as F
from atlas_interop import flecha_experiment as FX
from atlas_interop.presets import FLECHA_PRESETS


def test_site_design_roles():
    roles = FX.site_design()
    assert [r["native_atlas"] for r in roles] == ["synA", "synB", "synC", "synD", "synE"]
    assert [r["role"] for r in roles] == ["train"] * 4 + ["test"]


def test_duplicate_native_atlas_rejected():
    with pytest.raises(D.DataContractError, match="distinct native atlas"):
        FX.site_design(train_atlases=("synA", "synB", "synB", "synD"))
    with pytest.raises(D.DataContractError, match="distinct native atlas"):
        FX.site_design(train_atlases=("synA", "synB", "synC", "synE"))      # target reused for training
    with pytest.raises(D.DataContractError, match="exactly 5"):
        FX.site_design(train_atlases=("synA", "synB", "synC"))


def test_partition_disjoint_nonempty_deterministic():
    ids = np.array([f"s{i:03d}" for i in range(53)])
    p1, p2 = FX.partition(ids, 101), FX.partition(ids, 101)
    assert len(p1) == 5 and all(len(p) > 0 for p in p1)
    assert all(np.array_equal(a, b) for a, b in zip(p1, p2))
    assert sorted(np.concatenate(p1)) == list(range(53))
    assert any(not np.array_equal(a, b) for a, b in zip(p1, FX.partition(ids, 102)))


def test_reference_overlap_rejected(tmp):
    FX.generate(f"{tmp}/data", counts={"reference": 3, "downstream": 30})
    ds = D.load_connectomes(f"{tmp}/data/downstream/fc_synE.npz").subject_ids
    D.save_id_list(f"{tmp}/ref.txt", list(ds[:2]))
    with pytest.raises(D.DataContractError, match="overlap"):
        FX.run(f"{tmp}/out", data_dir=f"{tmp}/data", mappings={}, reference_ids=f"{tmp}/ref.txt")


def _site(name, n, seed):
    rng = np.random.default_rng(seed)
    ids = np.array([f"{name}_{i}" for i in range(n)])
    return F.Site(name, rng.normal(size=(n, 6)).astype(np.float32), rng.normal(size=n).astype(np.float32),
                  ids, ids, "A", True)


def test_single_site_path_never_calls_federated_code(monkeypatch):
    """The actual Single-site execution path must not use FedAvg or aggregation."""
    def boom(*a, **k):
        raise AssertionError("Single-site called federated code")

    monkeypatch.setattr(F, "fedavg_train", boom)
    monkeypatch.setattr(F, "average_state_dicts", boom)
    preset = FLECHA_PRESETS["demo"]
    site = _site("s1", 40, 0)
    model, info = F.train_single_site(site, preset, epochs=3, seed=0, device="cpu")
    assert info["update_steps"] == 3 * (40 // preset.batch_size)


def test_experiment_single_site_branch_is_standalone(tmp, monkeypatch):
    """Run the reviewer experiment with spies: fedavg_train is called exactly twice
    per partition (FLECHA and Uniform), never with fewer than four clients, and
    every Single-site fit receives exactly one site."""
    FX.generate(f"{tmp}/data", counts={"reference": 4, "downstream": 200})
    from atlas_interop.transform import fit_mapping, save_mapping
    maps, ref_t = {}, D.load_timeseries(f"{tmp}/data/reference/timeseries_synE.npz")
    for a in FX.TRAIN_ATLASES:
        m, _ = fit_mapping(D.load_timeseries(f"{tmp}/data/reference/timeseries_{a}.npz"), ref_t,
                           source_atlas=a, target_atlas="synE")
        maps[a] = save_mapping(f"{tmp}/maps/{a}.npz", m)
    calls = {"fedavg": [], "single": [], "avg_in_single": 0}
    real_fed, real_single, real_avg = F.fedavg_train, F.train_single_site, F.average_state_dicts
    in_single = {"on": False}

    def fed_spy(sites, *a, **k):
        calls["fedavg"].append([s.name for s in sites])
        return real_fed(sites, *a, **k)

    def single_spy(site, *a, **k):
        assert isinstance(site, F.Site)
        calls["single"].append(site.name)
        in_single["on"] = True
        try:
            return real_single(site, *a, **k)
        finally:
            in_single["on"] = False

    def avg_spy(states):
        if in_single["on"]:
            calls["avg_in_single"] += 1
        return real_avg(states)

    monkeypatch.setattr(F, "fedavg_train", fed_spy)
    monkeypatch.setattr(F, "train_single_site", single_spy)
    monkeypatch.setattr(F, "average_state_dicts", avg_spy)
    res = FX.run(f"{tmp}/out", data_dir=f"{tmp}/data", mappings=maps, seeds=(1, 2))
    assert calls["fedavg"] == [["site1", "site2", "site3", "site4"]] * 4          # FLECHA + Uniform, 2 seeds
    assert calls["single"] == ["site1", "site2", "site3", "site4"] * 2
    assert calls["avg_in_single"] == 0
    runs = res["runs"]
    for seed, g in runs.groupby("seed"):
        assert g.test_fingerprint.nunique() == 1
        assert g[g.method == "FLECHA"].train_fingerprint.item() == g[g.method == "Uniform"].train_fingerprint.item()
        assert g[g.method == "Single-site"].train_fingerprint.nunique() == 4
    s = res["summary"]
    for seed, g in runs.groupby("seed"):
        ss = g[g.method == "Single-site"].r
        assert s[s.seed == seed].SingleSite_mean.item() == pytest.approx(np.tanh(np.arctanh(ss).mean()))
