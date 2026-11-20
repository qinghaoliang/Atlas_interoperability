"""Atlas mapping: orientation, group-cost reuse, Sinkhorn, artifact format, signed reconstruction."""

import numpy as np
import pytest

from atlas_interop import data as D
from atlas_interop import transform as T


def _paired(n_subj=3, n_s=7, n_t=4, n_time=15, seed=0):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n_subj, 10, n_time))
    Ws = rng.random((n_s, 10))
    Wt = rng.random((n_t, 10))
    src = np.einsum("rl,slt->srt", Ws, latent) + 0.1 * rng.normal(size=(n_subj, n_s, n_time))
    tgt = np.einsum("rl,slt->srt", Wt, latent) + 0.1 * rng.normal(size=(n_subj, n_t, n_time))
    return src, tgt


def test_orientation_and_normalization_unequal_sizes():
    src, tgt = _paired(n_s=7, n_t=4)
    M, _ = T.fit_mapping_arrays(tgt, src)
    assert M.shape == (4, 7)                     # target x source
    assert (M >= 0).all()
    np.testing.assert_allclose(M.sum(axis=0), 1.0, atol=1e-9)   # each source column sums to one
    assert not np.allclose(M.sum(axis=1), 1.0)   # rows are not the normalized axis


def test_participant_cost_matches_independent_formula():
    src, tgt = _paired(n_subj=1)
    c = T.participant_cost(tgt[0], src[0])
    r = np.array([[np.corrcoef(tgt[0, q], src[0, p])[0, 1] for p in range(src.shape[1])]
                  for q in range(tgt.shape[1])])
    expected = 1 - (r - r.min()) / (r.max() - r.min())
    np.testing.assert_allclose(c, expected, atol=1e-12)
    assert c.shape == (tgt.shape[1], src.shape[1])


def test_group_cost_is_reused_for_every_participant(monkeypatch):
    src, tgt = _paired(n_subj=3, n_time=5)
    G = T.group_cost(tgt, src)
    seen = []
    real = T.sinkhorn_plan

    def spy(a, b, M, s):
        seen.append(M.copy())
        return real(a, b, M, s)

    monkeypatch.setattr(T, "sinkhorn_plan", spy)
    T.fit_mapping_arrays(tgt, src)
    assert len(seen) == 3 * 5
    for M in seen:
        np.testing.assert_array_equal(M, G)
    # and the group cost differs from any single participant's cost
    assert not np.allclose(G, T.participant_cost(tgt[0], src[0]))


def _numpy_sinkhorn(a, b, M, reg, n_iter=20000):
    K = np.exp(-M / reg)
    u = np.ones_like(a)
    for _ in range(n_iter):
        v = b / (K.T @ u)
        u = a / (K @ v)
    return u[:, None] * K * v[None, :]


def test_sinkhorn_against_independent_numpy_solver():
    rng = np.random.default_rng(3)
    M = rng.random((4, 6))
    a = T.activity_distribution(rng.normal(size=4), 1e-5)
    b = T.activity_distribution(rng.normal(size=6), 1e-5)
    s = T.FitSettings(reg=0.1)
    P, _ = T.sinkhorn_plan(a, b, M, s)
    Q = _numpy_sinkhorn(a, b, M, 0.1)
    np.testing.assert_allclose(P, Q, atol=1e-7)
    np.testing.assert_allclose(P.sum(1), a, atol=1e-7)   # target marginal
    np.testing.assert_allclose(P.sum(0), b, atol=1e-7)   # source marginal


def test_full_fit_matches_hand_computation():
    src, tgt = _paired(n_subj=2, n_s=5, n_t=3, n_time=4, seed=5)
    s = T.FitSettings(reg=0.05, window_size=2)
    M, _ = T.fit_mapping_arrays(tgt, src, s)
    G = T.group_cost(tgt, src)
    subj = []
    for i in range(2):
        wins = []
        for start in (0, 2):
            acc = 0
            for t in (start, start + 1):
                P = _numpy_sinkhorn(T.activity_distribution(tgt[i, :, t], s.shift),
                                    T.activity_distribution(src[i, :, t], s.shift), G, s.reg)
                acc = acc + P / P.sum(0, keepdims=True)
            wins.append(acc / 2)
        subj.append(np.mean(wins, 0))
    np.testing.assert_allclose(M, np.mean(subj, 0), atol=1e-6)


def test_activity_distribution():
    d = T.activity_distribution(np.array([2.0, 4.0, 3.0]), 1e-5)
    assert np.isclose(d.sum(), 1) and (d > 0).all() and d[1] > d[2] > d[0]
    u = T.activity_distribution(np.array([1.0, 1.0]), 1e-5)
    np.testing.assert_allclose(u, [0.5, 0.5])


def test_mapping_roundtrip_and_no_participant_data(tmp):
    src, tgt = _paired()
    M, info = T.fit_mapping_arrays(tgt, src)
    m = T.AtlasMapping(M, "S", "T", fit_info=info)
    p = T.save_mapping(f"{tmp}/m.npz", m)
    with np.load(p, allow_pickle=False) as z:
        assert int(z["format_version"]) == 2
        assert str(z["orientation"]) == "target_by_source"
        assert str(z["normalization"]) == "source_columns_sum_to_one"
        assert not {"subject_ids", "eval_scores"} & set(z.files)
    m2 = T.load_mapping(p)
    np.testing.assert_array_equal(m2.T, M)
    assert (m2.source_atlas, m2.target_atlas, m2.n_source, m2.n_target) == ("S", "T", 7, 4)


def test_legacy_mapping_is_rejected_with_explanation(tmp):
    np.savez(f"{tmp}/old.npz", mapping_matrix=np.full((7, 4), 0.25), source_atlas="S", target_atlas="T")
    with pytest.raises(D.DataContractError, match="(?i)legacy.*cannot be transposed"):
        T.load_mapping(f"{tmp}/old.npz")


def test_tampered_mapping_is_rejected(tmp):
    src, tgt = _paired()
    M, _ = T.fit_mapping_arrays(tgt, src)
    T.save_mapping(f"{tmp}/m.npz", T.AtlasMapping(M, "S", "T"))
    with np.load(f"{tmp}/m.npz") as z:
        d = dict(z)
    d["mapping"] = d["mapping"].T.copy()
    np.savez(f"{tmp}/bad.npz", **d)
    with pytest.raises(D.DataContractError):
        T.load_mapping(f"{tmp}/bad.npz")


def test_signed_reconstruction_regression():
    """Indefinite matrix with known spectrum; identity mapping must reproduce
    the rank-k signed approximation exactly, including negative components."""
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    lam = np.array([5.0, -4.0, 3.0, -0.5, 0.2, 0.1])
    C = Q @ np.diag(lam) @ Q.T
    Ct, n_pos, n_neg = T.transform_connectome(C, np.eye(6), k=3)
    assert (n_pos, n_neg) == (2, 1)              # ranked by |eigenvalue|: 5, -4, 3
    keep = [0, 1, 2]
    expected = Q[:, keep] @ np.diag(lam[keep]) @ Q[:, keep].T
    np.testing.assert_allclose(Ct, expected, atol=1e-10)
    full, _, _ = T.transform_connectome(C, np.eye(6), k=6)
    np.testing.assert_allclose(full, C, atol=1e-10)
    # dropping the negative part would be wrong
    Up, Un = T.signed_embedding(C, 3)
    assert not np.allclose(Up @ Up.T, expected)


def test_projection_through_mapping():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(5, 5))
    C = (A + A.T) / 2
    Tm = rng.random((3, 5))
    Tm /= Tm.sum(0, keepdims=True)
    Ct, _, _ = T.transform_connectome(C, Tm, k=5)
    np.testing.assert_allclose(Ct, Tm @ C @ Tm.T, atol=1e-10)   # full rank: T C T^T


def test_rank_validated_against_source_size():
    C = np.eye(4)
    with pytest.raises(ValueError, match="rank"):
        T.signed_embedding(C, 5)
    cs = D.ConnectomeSet(np.repeat(C[:, :, None], 2, 2), np.array(["a", "b"]), None, "S")
    m = T.AtlasMapping(np.full((3, 4), 1 / 3), "S", "T")
    with pytest.raises(ValueError, match="rank"):
        T.apply_mapping(cs, m, k=5)


def test_apply_rejects_wrong_dimension_and_atlas():
    m = T.AtlasMapping(np.full((3, 4), 1 / 3), "S", "T")
    cs = D.ConnectomeSet(np.zeros((5, 5, 2)), np.array(["a", "b"]), None, "S")
    with pytest.raises(D.DataContractError, match="regions"):
        T.apply_mapping(cs, m, k=2)
    cs = D.ConnectomeSet(np.zeros((4, 4, 2)), np.array(["a", "b"]), None, "Other")
    with pytest.raises(D.DataContractError, match="atlas"):
        T.apply_mapping(cs, m, k=2)


def test_apply_preserves_ids_and_order():
    rng = np.random.default_rng(2)
    ids = np.array(["007", "003", "010"])
    A = rng.normal(size=(4, 4, 3))
    cs = D.ConnectomeSet((A + A.transpose(1, 0, 2)) / 2, ids, None, "S")
    m = T.AtlasMapping(np.full((3, 4), 1 / 3), "S", "T")
    out, rep = T.apply_mapping(cs, m, k=2)
    assert list(out.subject_ids) == ["007", "003", "010"]
    assert out.mats.shape == (3, 3, 3) and out.atlas == "T"
    np.testing.assert_allclose(out.mats, out.mats.transpose(1, 0, 2))


def test_fit_aligns_shuffled_target_timeseries(tmp):
    """Fitting with the target file's rows shuffled (IDs moving with them) must
    give the same mapping; a positional pairing would not."""
    src, tgt = _paired(n_subj=4, n_time=6, seed=9)
    ids = np.array(["s01", "s02", "s03", "s04"])
    S = D.TimeSeriesSet(src, ids, None, "S")
    Tt = D.TimeSeriesSet(tgt, ids, None, "T")
    perm = np.array([2, 0, 3, 1])
    Tshuf = D.TimeSeriesSet(tgt[perm], ids[perm], None, "T")
    m1, _ = T.fit_mapping(S, Tt, source_atlas="S", target_atlas="T")
    m2, _ = T.fit_mapping(S, Tshuf, source_atlas="S", target_atlas="T")
    np.testing.assert_allclose(m1.T, m2.T, atol=1e-12)
    m_wrong, _ = T.fit_mapping_arrays(tgt[perm], src)   # positional pairing
    assert not np.allclose(m1.T, m_wrong, atol=1e-6)


def test_fit_rejects_unpaired_reference():
    src, tgt = _paired(n_subj=3)
    S = D.TimeSeriesSet(src, np.array(["a", "b", "c"]), None, "S")
    Tt = D.TimeSeriesSet(tgt, np.array(["a", "b", "x"]), None, "T")
    with pytest.raises(D.DataContractError, match="missing"):
        T.fit_mapping(S, Tt, source_atlas="S", target_atlas="T")
    Tt = D.TimeSeriesSet(tgt[:, :, :10], np.array(["a", "b", "c"]), None, "T")
    with pytest.raises(D.DataContractError, match="time points"):
        T.fit_mapping(S, Tt, source_atlas="S", target_atlas="T")


def test_reconstruction_similarity_matches_by_id():
    rng = np.random.default_rng(4)
    A = rng.normal(size=(4, 4, 3))
    A = (A + A.transpose(1, 0, 2)) / 2
    ids = np.array(["a", "b", "c"])
    mapped = D.ConnectomeSet(A, ids, None, "T")
    native = D.ConnectomeSet(A[:, :, [2, 0, 1]], ids[[2, 0, 1]], None, "T")
    keys, r = T.reconstruction_similarity(mapped, native)
    np.testing.assert_allclose(r, 1.0)
    assert list(keys) == ["a", "b", "c"]
