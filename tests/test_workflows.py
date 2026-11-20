"""Small end-to-end tests of FLECHA, MECHA and evaluation on in-memory data."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from atlas_interop import data as D
from atlas_interop import evaluation as E
from atlas_interop import flecha as F
from atlas_interop import mecha as ME
from atlas_interop import presets as P
from atlas_interop import transform as T
from atlas_interop.synthetic import SyntheticWorld


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("w"))
    w = SyntheticWorld(seed=1, n_latent=20, atlases={"A": 10, "B": 8})
    ids, z = w.subjects(160)
    lat = w.latent_timeseries(z, 150)
    out = {}
    for a in ("A", "B"):
        fc = w.fc_from_ts(w.regional_timeseries(lat, a))
        D.save_connectomes(f"{d}/fc_{a}.npz", D.ConnectomeSet(fc, ids, None, a))
        out[a] = f"{d}/fc_{a}.npz"
    ph = w.phenotypes(z, ["age", "t0", "t1", "t2", "h0", "h1"])
    ph.insert(0, "subject_id", ids)
    ph.sample(frac=1, random_state=0).to_csv(f"{d}/ph.csv", index=False)
    rid, rz = w.subjects(4, prefix="ref-")
    rl = w.latent_timeseries(rz, 30)
    for a in ("A", "B"):
        D.save_timeseries(f"{d}/ts_{a}.npz", D.TimeSeriesSet(w.regional_timeseries(rl, a), rid, None, a))
    m, _ = T.fit_mapping(D.load_timeseries(f"{d}/ts_A.npz"), D.load_timeseries(f"{d}/ts_B.npz"),
                         source_atlas="A", target_atlas="B")
    T.save_mapping(f"{d}/A_to_B.npz", m)
    m, _ = T.fit_mapping(D.load_timeseries(f"{d}/ts_B.npz"), D.load_timeseries(f"{d}/ts_A.npz"),
                         source_atlas="B", target_atlas="A")
    T.save_mapping(f"{d}/B_to_A.npz", m)
    return dict(dir=d, ids=ids, fc=out, ph=f"{d}/ph.csv")


def _sub(src, ids, dst):
    c = D.load_connectomes(src)
    D.save_connectomes(dst, c.take(D.align_to(list(ids), c.keys, ref_name="r", other_name="o")))
    return dst


TINY = P.FlechaPreset("t", (8,), "batch", 0.1, 1e-3, 1e-4, 8, 3, 2, n_seeds=2)


def test_flecha_sites_and_overlap_rejection(world, tmp):
    ids = world["ids"]
    _sub(world["fc"]["A"], ids[:40], f"{tmp}/a.npz")
    _sub(world["fc"]["B"], ids[40:80], f"{tmp}/b.npz")
    _sub(world["fc"]["B"], ids[80:120], f"{tmp}/t.npz")
    cfg = dict(target_atlas="B", phenotype="age",
               train_sites=[dict(name="a", connectomes=f"{tmp}/a.npz", labels=world["ph"], mapping=f"{world['dir']}/A_to_B.npz"),
                            dict(name="b", connectomes=f"{tmp}/b.npz", labels=world["ph"])],
               test_site=dict(name="t", connectomes=f"{tmp}/t.npz", labels=world["ph"]))
    json.dump(cfg, open(f"{tmp}/c.json", "w"))
    res = F.run_sites(F.load_site_config(f"{tmp}/c.json"), TINY, rank=5, seed=0, out_dir=f"{tmp}/o", single_site=True)
    assert {r["method"] for r in res["rows"]} == {"FLECHA", "Single-site"}
    assert res["run"]["sites"][0]["transform"].startswith("A -> B")
    assert res["run"]["sites"][1]["transform"].startswith("none")
    # overlapping train/test participants
    _sub(world["fc"]["B"], ids[70:110], f"{tmp}/t2.npz")
    cfg["test_site"]["connectomes"] = f"{tmp}/t2.npz"
    json.dump(cfg, open(f"{tmp}/c2.json", "w"))
    with pytest.raises(D.DataContractError, match="overlap"):
        F.run_sites(F.load_site_config(f"{tmp}/c2.json"), TINY, rank=5, seed=0, out_dir=f"{tmp}/o2")


def test_flecha_site_without_mapping_in_other_atlas_rejected(world, tmp):
    ids = world["ids"]
    _sub(world["fc"]["A"], ids[:40], f"{tmp}/a.npz")
    _sub(world["fc"]["B"], ids[80:120], f"{tmp}/t.npz")
    cfg = dict(target_atlas="B", phenotype="age", train_sites=[dict(name="a", connectomes=f"{tmp}/a.npz", labels=world["ph"])],
               test_site=dict(name="t", connectomes=f"{tmp}/t.npz", labels=world["ph"]))
    json.dump(cfg, open(f"{tmp}/c.json", "w"))
    with pytest.raises(D.DataContractError):
        F.run_sites(F.load_site_config(f"{tmp}/c.json"), TINY, rank=5, seed=0, out_dir=f"{tmp}/o")


def test_flecha_missing_labels_not_replaced(world, tmp):
    ids = world["ids"]
    _sub(world["fc"]["B"], ids[:40], f"{tmp}/b.npz")
    _sub(world["fc"]["B"], ids[80:120], f"{tmp}/t.npz")
    cfg = dict(target_atlas="B", phenotype="nonexistent", train_sites=[dict(name="b", connectomes=f"{tmp}/b.npz", labels=world["ph"])],
               test_site=dict(name="t", connectomes=f"{tmp}/t.npz", labels=world["ph"]))
    json.dump(cfg, open(f"{tmp}/c.json", "w"))
    with pytest.raises(D.DataContractError, match="not found"):
        F.run_sites(F.load_site_config(f"{tmp}/c.json"), TINY, rank=5, seed=0, out_dir=f"{tmp}/o")


def test_flecha_simulated_matched_and_uniform_native(world, tmp):
    cfg = dict(phenotype="age", labels=world["ph"], atlases={"A": world["fc"]["A"], "B": world["fc"]["B"]},
               mappings={"A->B": f"{world['dir']}/A_to_B.npz", "B->A": f"{world['dir']}/B_to_A.npz"},
               target_atlases=["B"])
    json.dump(cfg, open(f"{tmp}/s.json", "w"))
    runs = F.run_simulated(F.load_simulation_config(f"{tmp}/s.json"), TINY, rank=5, seeds=[0, 1], out_dir=f"{tmp}/o")
    assert set(runs["method"]) == {"FLECHA", "Single-site", "Uniform"}
    for _, g in runs.groupby("seed"):
        assert g["n_test"].nunique() == 1
    assert (runs.loc[runs.method == "Uniform", "source_atlas"] == "B").all()
    s = F.summarize(runs)
    assert "q_bh" in s


def test_mecha_end_to_end_and_guards(world, tmp):
    ids = world["ids"]
    _sub(world["fc"]["A"], ids[:80], f"{tmp}/mt.npz")
    _sub(world["fc"]["A"], ids[80:100], f"{tmp}/va.npz")
    _sub(world["fc"]["B"], ids[100:130], f"{tmp}/su.npz")
    _sub(world["fc"]["B"], ids[130:160], f"{tmp}/te.npz")
    preset = P.override(P.MECHA_PRESETS["preflight"], max_epochs=5, batch_size=16, k_list=(10,), n_repeats=2,
                        baseline_dnn_epochs=3, baseline_dnn_hidden=(8,))
    table = D.load_phenotypes(world["ph"])
    run = ME.train_dnn(train_conn=D.load_connectomes(f"{tmp}/mt.npz"), val_conn=D.load_connectomes(f"{tmp}/va.npz"),
                       table=table, train_phenotypes=["t0", "t1", "t2"], mapping_path=f"{world['dir']}/A_to_B.npz",
                       rank=5, preset=preset, seed=0, out_dir=f"{tmp}/tr")
    model, ck, _ = ME.load_checkpoint(f"{tmp}/tr/mecha_dnn.pt")
    assert ck["spec"]["output_size"] == 3 and ck["train_phenotypes"] == ["t0", "t1", "t2"]
    assert ck["target_atlas"] == "B" and ck["vectorization"] == D.VECTORIZATION
    runs = ME.adapt_and_evaluate(checkpoint=f"{tmp}/tr/mecha_dnn.pt", support_conn=D.load_connectomes(f"{tmp}/su.npz"),
                                 test_conn=D.load_connectomes(f"{tmp}/te.npz"), table=table, test_phenotypes=["h0", "h1"],
                                 preset=preset, seed=0, out_dir=f"{tmp}/ad")
    assert set(runs["method"]) == {"MECHA", "Ridge", "DNN"}
    # held-out phenotypes may not overlap the training phenotypes
    with pytest.raises(D.DataContractError, match="overlap"):
        ME.adapt_and_evaluate(checkpoint=f"{tmp}/tr/mecha_dnn.pt", support_conn=D.load_connectomes(f"{tmp}/su.npz"),
                              test_conn=D.load_connectomes(f"{tmp}/te.npz"), table=table, test_phenotypes=["t0"],
                              preset=preset, seed=0, out_dir=f"{tmp}/x")
    # support/test participant overlap
    with pytest.raises(D.DataContractError, match="overlap"):
        ME.adapt_and_evaluate(checkpoint=f"{tmp}/tr/mecha_dnn.pt", support_conn=D.load_connectomes(f"{tmp}/su.npz"),
                              test_conn=D.load_connectomes(f"{tmp}/su.npz"), table=table, test_phenotypes=["h0"],
                              preset=preset, seed=0, out_dir=f"{tmp}/x")
    # source-atlas (non-native) support rejected
    with pytest.raises(D.DataContractError):
        ME.adapt_and_evaluate(checkpoint=f"{tmp}/tr/mecha_dnn.pt", support_conn=D.load_connectomes(f"{tmp}/va.npz"),
                              test_conn=D.load_connectomes(f"{tmp}/te.npz"), table=table, test_phenotypes=["h0"],
                              preset=preset, seed=0, out_dir=f"{tmp}/x")
    # train/validation overlap
    with pytest.raises(D.DataContractError, match="overlap"):
        ME.train_dnn(train_conn=D.load_connectomes(f"{tmp}/mt.npz"), val_conn=D.load_connectomes(f"{tmp}/mt.npz"),
                     table=table, train_phenotypes=["t0"], mapping_path=f"{world['dir']}/A_to_B.npz", rank=5,
                     preset=preset, seed=0, out_dir=f"{tmp}/x")


def test_mecha_explicit_split_size_checked(world, tmp):
    ids = world["ids"]
    os.makedirs(f"{tmp}/splits")
    D.save_id_list(f"{tmp}/splits/support_k10_rep0.txt", ids[100:109])
    with pytest.raises(D.DataContractError, match="k=10"):
        ME.load_or_draw_supports(ids[100:130], (10,), 1, 0, f"{tmp}/splits", ids[100:130])


def test_mecha_ignores_test_labels_in_fitting(world, tmp):
    """Changing test labels must not change support-fitted predictions."""
    ids = world["ids"]
    _sub(world["fc"]["B"], ids[:60], f"{tmp}/mt.npz")
    _sub(world["fc"]["B"], ids[60:80], f"{tmp}/va.npz")
    _sub(world["fc"]["B"], ids[80:110], f"{tmp}/su.npz")
    _sub(world["fc"]["B"], ids[110:140], f"{tmp}/te.npz")
    preset = P.override(P.MECHA_PRESETS["preflight"], max_epochs=3, batch_size=16, k_list=(10,), n_repeats=1,
                        baseline_dnn_epochs=2, baseline_dnn_hidden=(8,))
    table = D.load_phenotypes(world["ph"])
    ME.train_dnn(train_conn=D.load_connectomes(f"{tmp}/mt.npz"), val_conn=D.load_connectomes(f"{tmp}/va.npz"),
                 table=table, train_phenotypes=["t0"], mapping_path=None, rank=5, preset=preset, seed=0, out_dir=f"{tmp}/tr")
    df = pd.read_csv(world["ph"], dtype={"subject_id": str})
    test_ids = set(ids[110:140])
    df2 = df.copy()
    df2.loc[df2.subject_id.isin(test_ids), "h0"] = df2.loc[df2.subject_id.isin(test_ids), "h0"] * -1
    df2.to_csv(f"{tmp}/ph2.csv", index=False)
    kw = dict(checkpoint=f"{tmp}/tr/mecha_dnn.pt", support_conn=D.load_connectomes(f"{tmp}/su.npz"),
              test_conn=D.load_connectomes(f"{tmp}/te.npz"), test_phenotypes=["h0"], preset=preset, seed=0)
    ME.adapt_and_evaluate(table=table, out_dir=f"{tmp}/a1", **kw)
    r1 = pd.read_csv(f"{tmp}/a1/per_phenotype.csv")
    ME.adapt_and_evaluate(table=D.load_phenotypes(f"{tmp}/ph2.csv"), out_dir=f"{tmp}/a2", **kw)
    r2 = pd.read_csv(f"{tmp}/a2/per_phenotype.csv")
    np.testing.assert_allclose(r1["r"], -r2["r"], atol=1e-10)   # only scoring sign flips
    np.testing.assert_allclose(r1["alpha"].fillna(0), r2["alpha"].fillna(0))


def test_univariate_selection_and_fit_on_train_only():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 20))
    y = X[:, 0] * 2 + rng.normal(size=200) * 0.5
    mask = E.univariate_select(X, y, 0.1)
    assert mask[0]
    from scipy.stats import pearsonr
    pvals = np.array([pearsonr(X[:, j], y)[1] for j in range(20)])
    np.testing.assert_array_equal(mask, pvals < 0.1)


def test_prediction_tuning_participants_excluded():
    rng = np.random.default_rng(0)
    n = 100
    X = rng.normal(size=(n, 15))
    y = X[:, 0] + rng.normal(size=n) * 0.3
    ids = np.array([f"s{i}" for i in range(n)])
    rows, info = E.predict_pair(X, X, y, ids, P.PREDICTION_PRESETS["demo"], seed=0)
    assert info["n_tune"] == 20 and info["n_eval"] == 80
    assert all(r["n_eval"] == 80 for r in rows)
    assert np.mean([r["r"] for r in rows]) > 0.5


def test_cross_atlas_prediction_uses_native_test_features():
    """Train-space and native features differ; held-out predictions must use native."""
    rng = np.random.default_rng(1)
    n = 120
    Xn = rng.normal(size=(n, 10))
    y = Xn[:, 0] + 0.2 * rng.normal(size=n)
    X_train = Xn.copy()
    ids = np.array([f"s{i}" for i in range(n)])
    good, _ = E.predict_pair(X_train, Xn, y, ids, P.PREDICTION_PRESETS["demo"], seed=0)
    bad, _ = E.predict_pair(X_train, rng.normal(size=(n, 10)), y, ids, P.PREDICTION_PRESETS["demo"], seed=0)
    assert np.mean([r["r"] for r in good]) > 0.8 > np.mean([abs(r["r"]) for r in bad]) + 0.5


# --- regression checks for the canonical synthetic MECHA experiment -------------
def test_nested_supports_prefixes_and_fingerprints(tmp):
    ids = np.array([f"s{i:03d}" for i in range(60)])
    tab = ME.make_nested_supports(ids, [10, 20, 40], [5, 6], f"{tmp}/sp")
    for rep in range(2):
        sets = {k: set(D.load_id_list(f"{tmp}/sp/support_k{k}_rep{rep}.txt")) for k in (10, 20, 40)}
        assert sets[10] < sets[20] < sets[40]
        assert all(len(v) == k for k, v in sets.items())
        for k, v in sets.items():
            assert tab.query("k == @k and repeat == @rep").support_fingerprint.item() == D.fingerprint(v)
    assert D.fingerprint(["b", "a"]) == D.fingerprint(["a", "b"]) != D.fingerprint(["a", "c"])
    # deterministic in the seed, different across seeds
    tab2 = ME.make_nested_supports(ids, [10, 20, 40], [5, 6], f"{tmp}/sp2")
    assert tab.support_fingerprint.tolist() == tab2.support_fingerprint.tolist()
    assert tab.query("repeat == 0").support_fingerprint.tolist() != tab.query("repeat == 1").support_fingerprint.tolist()


def _small_mecha(world, tmp, train_cols=("t0", "t1", "t2")):
    ids = world["ids"]
    _sub(world["fc"]["A"], ids[:80], f"{tmp}/mt.npz")
    _sub(world["fc"]["A"], ids[80:100], f"{tmp}/va.npz")
    _sub(world["fc"]["B"], ids[100:130], f"{tmp}/su.npz")
    _sub(world["fc"]["B"], ids[130:160], f"{tmp}/te.npz")
    preset = P.override(P.MECHA_PRESETS["preflight"], k_list=(10, 20), n_repeats=2)
    table = D.load_phenotypes(world["ph"])
    ME.train_dnn(train_conn=D.load_connectomes(f"{tmp}/mt.npz"), val_conn=D.load_connectomes(f"{tmp}/va.npz"),
                 table=table, train_phenotypes=list(train_cols), mapping_path=f"{world['dir']}/A_to_B.npz",
                 rank=5, preset=preset, seed=0, out_dir=f"{tmp}/tr")
    ME.make_nested_supports(D.load_connectomes(f"{tmp}/su.npz").subject_ids, [10, 20], [1, 2], f"{tmp}/sp")
    return preset, table


def test_mecha_features_are_dnn_outputs_in_checkpoint_order(world, tmp, monkeypatch):
    preset, table = _small_mecha(world, tmp)
    model, ck, dev = ME.load_checkpoint(f"{tmp}/tr/mecha_dnn.pt")
    su = D.load_connectomes(f"{tmp}/su.npz")
    expected = ME.predict(model, su.edges().astype(np.float32), dev)
    assert expected.shape[1] == len(ck["train_phenotypes"]) == 3
    seen = []
    real = ME.fit_ridge_cv

    def spy(X, y, alphas, seed, scale):
        seen.append((X.copy(), scale))
        return real(X, y, alphas, seed, scale)

    monkeypatch.setattr(ME, "fit_ridge_cv", spy)
    ME.adapt_and_evaluate(checkpoint=f"{tmp}/tr/mecha_dnn.pt", support_conn=su, test_conn=D.load_connectomes(f"{tmp}/te.npz"),
                          table=table, test_phenotypes=["h0", "h1"], preset=preset, seed=0, out_dir=f"{tmp}/ad",
                          splits_dir=f"{tmp}/sp", baselines=())
    # MECHA ridge inputs are rows of the DNN output matrix, never hidden activations
    pos = {s: i for i, s in enumerate(su.subject_ids)}
    ids10 = D.load_id_list(f"{tmp}/sp/support_k10_rep0.txt")
    X0, scale0 = seen[0]
    assert X0.shape[1] == 3 and not scale0
    np.testing.assert_allclose(X0, expected[[pos[s] for s in ids10]], rtol=1e-6)
    assert len(seen) == 2 * 2 * 2         # one adaptation model per (K, draw, held-out phenotype)


def test_baselines_only_run_matches_and_compare_checks_fingerprints(world, tmp):
    preset, table = _small_mecha(world, tmp)
    kw = dict(checkpoint=f"{tmp}/tr/mecha_dnn.pt", support_conn=D.load_connectomes(f"{tmp}/su.npz"),
              test_conn=D.load_connectomes(f"{tmp}/te.npz"), table=table, test_phenotypes=["h0", "h1"],
              preset=preset, seed=0, splits_dir=f"{tmp}/sp")
    ME.adapt_and_evaluate(out_dir=f"{tmp}/m", baselines=(), **kw)
    ME.adapt_and_evaluate(out_dir=f"{tmp}/b", baselines=("ridge", "dnn"), include_mecha=False, **kw)
    ME.adapt_and_evaluate(out_dir=f"{tmp}/all", **kw)
    m, b, a = (pd.read_csv(f"{tmp}/{d}/per_phenotype.csv") for d in ("m", "b", "all"))
    assert set(m.method) == {"MECHA"} and set(b.method) == {"Ridge", "DNN"}
    # separately run baselines equal the jointly run ones (same draws and seeds)
    key = ["method", "k", "repeat", "phenotype"]
    j = b.merge(a, on=key, suffixes=("_b", "_a"))
    np.testing.assert_allclose(j.r_b, j.r_a)
    res = ME.compare_matched(a)                       # joint run
    res2 = ME.compare_matched(m, b)                   # separate baseline run
    assert len(res["summary"]) == 2 and (res["summary"].n_draws == 2).all()
    np.testing.assert_allclose(res["summary"].r_MECHA_mean, res2["summary"].r_MECHA_mean)
    bad = b.copy()
    bad.loc[bad.index[0], "support_fingerprint"] = "deadbeef"
    with pytest.raises(D.DataContractError, match="fingerprint"):
        ME.compare_matched(m, bad)
    bad = a.copy()
    bad.loc[bad.method == "Ridge", "test_fingerprint"] = "deadbeef"
    with pytest.raises(D.DataContractError, match="fingerprint"):
        ME.compare_matched(bad)


def test_validation_labels_select_but_do_not_train(world, tmp):
    """Changing validation labels may change the selected epoch but never the
    parameter trajectory: with patience off, every epoch's weights are identical."""
    ids = world["ids"]
    _sub(world["fc"]["B"], ids[:80], f"{tmp}/mt.npz")
    _sub(world["fc"]["B"], ids[80:100], f"{tmp}/va.npz")
    table = D.load_phenotypes(world["ph"])
    df = pd.read_csv(world["ph"], dtype={"subject_id": str})
    df.loc[df.subject_id.isin(ids[80:100]), "t0"] *= -3
    df.to_csv(f"{tmp}/ph2.csv", index=False)
    preset = P.override(P.MECHA_PRESETS["preflight"], max_epochs=3)
    hist = []
    for tab, d in ((table, "a"), (D.load_phenotypes(f"{tmp}/ph2.csv"), "b")):
        ME.train_dnn(train_conn=D.load_connectomes(f"{tmp}/mt.npz"), val_conn=D.load_connectomes(f"{tmp}/va.npz"),
                     table=tab, train_phenotypes=["t0"], mapping_path=None, rank=5, preset=preset, seed=0,
                     out_dir=f"{tmp}/{d}")
        hist.append(pd.read_csv(f"{tmp}/{d}/training_history.csv"))
    np.testing.assert_allclose(hist[0].train_loss, hist[1].train_loss)     # training unaffected
    assert not np.allclose(hist[0].val_loss, hist[1].val_loss)              # validation is what changed


def test_synthetic_mecha_generator_roles_and_columns(tmp):
    from atlas_interop import synthetic_mecha as SM
    man = SM.generate(tmp, counts=dict(reference=3, meta_training=30, validation=10, support=12, test=8))
    tab = D.load_phenotypes(f"{tmp}/phenotypes.csv")
    assert len(SM.TRAIN_COLUMNS) == 16 and len(SM.HELDOUT_COLUMNS) == 10
    assert not set(SM.TRAIN_COLUMNS) & set(SM.HELDOUT_COLUMNS)
    assert list(tab.values.columns) == SM.TRAIN_COLUMNS + SM.HELDOUT_COLUMNS
    assert tab.values.notna().all().all()
    ids = {}
    for role, atlas in (("meta_training", "synS"), ("validation", "synS"), ("support", "synT"), ("test", "synT")):
        c = D.load_connectomes(f"{tmp}/{role}_fc_{atlas}.npz")
        assert c.atlas == atlas and c.n_regions == SM.ATLASES[atlas]
        ids[role] = c.subject_ids
    ids["reference"] = D.load_timeseries(f"{tmp}/reference/timeseries_synS.npz").subject_ids
    D.assert_disjoint_subjects(ids)
    assert man["design"]["factors_shared_A"] == [0, 1, 2, 3]
    assert "subject_ids" not in json.dumps(man)
