"""Aggregation, matched epochs, statistics and model presets."""

import numpy as np
import pandas as pd
import pytest
import torch

from atlas_interop import flecha as F
from atlas_interop import mecha as ME
from atlas_interop import stats as S
from atlas_interop.models import MLPRegressor, MLPSpec, nan_mse
from atlas_interop.presets import FLECHA_PRESETS, MECHA_PRESETS, PREDICTION_PRESETS, TRANSFORM_RANK


def test_bh_matches_hand_computation():
    p = np.array([0.01, 0.04, 0.03, 0.2])
    # sorted 0.01,0.03,0.04,0.2 -> 0.04,0.06,0.0533,0.2 -> monotone 0.04,0.0533,0.0533,0.2
    np.testing.assert_allclose(S.bh_fdr(p), [0.04, 0.0533333, 0.0533333, 0.2], atol=1e-6)
    q = S.bh_fdr(np.array([0.01, np.nan]))
    assert q[0] == 0.01 and np.isnan(q[1])


def test_wilcoxon_one_sided_direction_and_small_n_note():
    a = np.arange(10) + 1.0
    b = np.zeros(10)
    r = S.wilcoxon_greater(a, b)
    assert r["p"] == pytest.approx(0.5 ** 10)
    assert S.wilcoxon_greater(b, a)["p"] > 0.99
    assert "not suitable" in S.wilcoxon_greater(a[:3], b[:3])["note"]
    assert np.isnan(S.wilcoxon_greater([1, 1], [1, 1])["p"])


def test_fisher_mean():
    assert S.fisher_mean_r([0.5, 0.5]) == pytest.approx(0.5)
    assert S.fisher_mean_r([0.1, 0.9]) == pytest.approx(np.tanh((np.arctanh(0.1) + np.arctanh(0.9)) / 2))


def test_equal_weight_averaging():
    a = {"w": torch.tensor([1.0, 3.0]), "n": torch.tensor(5)}
    b = {"w": torch.tensor([3.0, 5.0]), "n": torch.tensor(9)}
    out = F.average_state_dicts([a, b])
    assert torch.equal(out["w"], torch.tensor([2.0, 4.0]))
    assert int(out["n"]) == 5


def test_matched_epochs_rule_and_bounds():
    p = FLECHA_PRESETS["ukb"]
    # 100 rounds * 10 steps * 5 clients * 64 = 320000 samples
    assert F.matched_epochs(8000, 5, p) == 40
    assert F.matched_epochs(100, 5, p) == 80      # upper bound
    assert F.matched_epochs(10**7, 5, p) == 5     # lower bound


def test_fedavg_one_round_equals_manual_average():
    rng = np.random.default_rng(0)
    sites = [F.Site(f"s{i}", rng.normal(size=(8, 5)).astype(np.float32), rng.normal(size=8).astype(np.float32),
                    np.array([f"{i}_{j}" for j in range(8)]), np.array([f"{i}_{j}" for j in range(8)]), "A", False)
             for i in range(2)]
    preset = FLECHA_PRESETS["demo"].__class__(**{**FLECHA_PRESETS["demo"].as_dict(), "hidden": (4,), "rounds": 1,
                                                 "local_steps": 1, "batch_size": 4, "dropout": 0.0, "norm": "none"})
    model, info = F.fedavg_train(sites, preset, seed=0, device=torch.device("cpu"))
    assert info["client_update_steps"] == [1, 1]
    # replicate manually
    torch.manual_seed(0)
    np.random.seed(0)
    import random; random.seed(0)
    g = MLPRegressor(MLPSpec(5, [4], 1, 0.0, "none"))
    clients = [__import__("copy").deepcopy(g) for _ in sites]
    r = np.random.default_rng(0)
    streams = [F._Stream(s.X, s.y, 4, r) for s in sites]
    for m, st in zip(clients, streams):
        opt = torch.optim.Adam(m.parameters(), lr=preset.learning_rate, weight_decay=preset.weight_decay)
        xb, yb = st.next()
        opt.zero_grad()
        torch.nn.functional.mse_loss(m(torch.as_tensor(xb)), torch.as_tensor(yb).view(-1, 1)).backward()
        opt.step()
    avg = F.average_state_dicts([m.state_dict() for m in clients])
    for k, v in model.model.state_dict().items():
        torch.testing.assert_close(v, avg[k])


def test_empty_client_rejected():
    site = F.Site("tiny", np.zeros((3, 4), np.float32), np.zeros(3, np.float32), np.array(["a", "b", "c"]),
                  np.array(["a", "b", "c"]), "A", False)
    with pytest.raises(F.FlechaInputError, match="batch size"):
        F.check_trainable(site, 16)


def test_partition_is_subject_level_and_deterministic():
    sub = np.array(["a", "a", "b", "c", "d", "e"])
    keys = np.array(["a1", "a2", "b", "c", "d", "e"])
    p1 = F.partition_sites(keys, sub, 3, seed=1)
    p2 = F.partition_sites(keys, sub, 3, seed=1)
    assert all(np.array_equal(x, y) for x, y in zip(p1, p2))
    site_of = {i: g for g, idx in enumerate(p1) for i in idx}
    assert site_of[0] == site_of[1]        # both visits of 'a' together
    assert sorted(np.concatenate(p1)) == list(range(6))


def test_flecha_summary_fisher_and_bh():
    rows = []
    for seed in range(6):
        rows += [dict(target_atlas="T", seed=seed, method="FLECHA", r=0.5 + 0.01 * seed),
                 dict(target_atlas="T", seed=seed, method="Uniform", r=0.6),
                 dict(target_atlas="T", seed=seed, method="Single-site", r=0.1),
                 dict(target_atlas="T", seed=seed, method="Single-site", r=0.3)]
    s = F.summarize(pd.DataFrame(rows))
    ss = S.fisher_mean_r([0.1, 0.3])
    assert s["r_Single-site_mean"].iloc[0] == pytest.approx(ss)
    assert s["delta_FLECHA_minus_SingleSite"].iloc[0] == pytest.approx(np.mean(0.5 + 0.01 * np.arange(6)) - ss)
    assert s["gap_Uniform_minus_FLECHA"].iloc[0] == pytest.approx(0.6 - np.mean(0.5 + 0.01 * np.arange(6)))
    assert s["wilcoxon_p"].iloc[0] == pytest.approx(0.5 ** 6)


def test_mecha_summary_benchmark_selection():
    rows = []
    for rep in range(6):
        rows += [dict(k=10, repeat=rep, method="MECHA", mean_r=0.4),
                 dict(k=10, repeat=rep, method="Ridge", mean_r=0.2 + 0.01 * rep),
                 dict(k=10, repeat=rep, method="DNN", mean_r=0.1)]
    s = ME.summarize(pd.DataFrame(rows))
    assert s["benchmark"].iloc[0] == "Ridge"
    assert s["gain"].iloc[0] == pytest.approx(0.4 - np.mean(0.2 + 0.01 * np.arange(6)))


def test_nan_mse_masks_missing():
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    tgt = torch.tensor([[1.0, float("nan")], [5.0, 4.0]])
    assert float(nan_mse(pred, tgt)) == pytest.approx(4.0 / 3)


def test_manuscript_presets_match_methods():
    u, h, m = FLECHA_PRESETS["ukb"], FLECHA_PRESETS["hcp"], MECHA_PRESETS["ukb"]
    assert (u.hidden, u.norm, u.dropout, u.learning_rate, u.weight_decay, u.batch_size, u.rounds, u.local_steps) == \
        ((256, 128, 64), "batch", 0.2, 1e-3, 1e-4, 64, 100, 10)
    assert (h.hidden, h.norm, h.dropout, h.batch_size, h.rounds, h.local_steps) == ((128, 64), "layer", 0.2, 32, 200, 5)
    assert (u.epochs_min, u.epochs_max, u.n_seeds) == (5, 80, 10)
    assert (m.hidden, m.norm, m.dropout, m.learning_rate, m.weight_decay, m.batch_size) == \
        ((128, 64), "batch", 0.5, 1e-4, 1e-4, 256)
    assert m.k_list == (10, 20, 50, 100, 200) and m.n_repeats == 10
    p = PREDICTION_PRESETS["manuscript"]
    assert (p.tuning_fraction, p.p_threshold, p.inner_folds, p.outer_folds, p.n_repeats) == (0.2, 0.1, 3, 10, 10)
    assert min(p.alphas) == pytest.approx(0.1) and max(p.alphas) == pytest.approx(1000)
    assert TRANSFORM_RANK["manuscript"] == 50


def test_model_norm_variants():
    for norm, cls in (("batch", torch.nn.BatchNorm1d), ("layer", torch.nn.LayerNorm)):
        m = MLPRegressor(MLPSpec(10, [8, 4], 3, 0.2, norm))
        assert sum(isinstance(x, cls) for x in m.modules()) == 2
        assert m(torch.zeros(5, 10)).shape == (5, 3)
