"""Data contract: ID handling, alignment, shuffled rows, negative cases."""

import numpy as np
import pandas as pd
import pytest

from atlas_interop import data as D

from conftest import make_conn, write_table


def test_leading_zeros_preserved_through_csv_and_npz(tmp):
    ids = ["00123", "0007", "42"]
    write_table(f"{tmp}/p.csv", ids, {"y": [1.0, 2.0, 3.0]})
    t = D.load_phenotypes(f"{tmp}/p.csv")
    assert list(t.subject_ids) == ids
    c = make_conn(3, ids)
    D.save_connectomes(f"{tmp}/c.npz", c)
    assert list(D.load_connectomes(f"{tmp}/c.npz").subject_ids) == ids


def test_numeric_ids_rejected(tmp):
    np.savez(f"{tmp}/c.npz", mats=np.zeros((3, 3, 2)), subject_ids=np.array([1, 2]))
    with pytest.raises(D.DataContractError, match="strings"):
        D.load_connectomes(f"{tmp}/c.npz")


def test_missing_subject_ids_rejected(tmp):
    np.savez(f"{tmp}/c.npz", mats=np.zeros((3, 3, 2)))
    with pytest.raises(D.DataContractError, match="subject_ids"):
        D.load_connectomes(f"{tmp}/c.npz")


def test_shuffled_phenotype_rows_align_by_id(tmp):
    """y is an exact function of each subject's connectome. The table is
    written in deranged order, so positional matching would fail every row."""
    n = 40
    ids = np.array([f"s{i:03d}" for i in range(n)])
    c = make_conn(6, ids, seed=1)
    y_true = c.edges().mean(1)
    perm = np.random.default_rng(0).permutation(n)
    while (perm == np.arange(n)).any():
        perm = np.random.default_rng(int(perm[0]) + 1).permutation(n)
    write_table(f"{tmp}/p.csv", ids[perm], {"y": y_true[perm]})
    t = D.load_phenotypes(f"{tmp}/p.csv")
    y = D.phenotypes_for(c, t, ["y"])[:, 0]
    np.testing.assert_allclose(y, y_true)
    assert not np.allclose(t.values["y"].to_numpy(), y_true)   # positional would be wrong


def test_duplicate_ids_rejected(tmp):
    write_table(f"{tmp}/p.csv", ["a", "b", "a"], {"y": [1, 2, 3]})
    with pytest.raises(D.DataContractError, match="duplicate"):
        D.load_phenotypes(f"{tmp}/p.csv")
    D.save_connectomes(f"{tmp}/c.npz", make_conn(3, ["a", "a"]))
    with pytest.raises(D.DataContractError, match="duplicate"):
        D.load_connectomes(f"{tmp}/c.npz")


def test_missing_ids_rejected(tmp):
    write_table(f"{tmp}/p.csv", ["a", "b"], {"y": [1, 2]})
    c = make_conn(3, ["a", "b", "c"])
    with pytest.raises(D.DataContractError, match="missing"):
        D.phenotypes_for(c, D.load_phenotypes(f"{tmp}/p.csv"), ["y"])


def test_exact_pairing_rejects_extra_keys():
    with pytest.raises(D.DataContractError, match="not in"):
        D.align_to(["a", "b"], ["b", "a", "c"], ref_name="ref", other_name="other", allow_extra=False)
    idx = D.align_to(["a", "b"], ["b", "a", "c"], ref_name="ref", other_name="other")
    assert list(idx) == [1, 0]


def test_incompatible_dimensions_rejected(tmp):
    np.savez(f"{tmp}/c.npz", mats=np.zeros((3, 4, 2)), subject_ids=np.array(["a", "b"]))
    with pytest.raises(D.DataContractError, match="n_regions, n_regions"):
        D.load_connectomes(f"{tmp}/c.npz")
    np.savez(f"{tmp}/c2.npz", mats=np.zeros((3, 3, 2)), subject_ids=np.array(["a", "b", "c"]))
    with pytest.raises(D.DataContractError, match="samples on axis 2"):
        D.load_connectomes(f"{tmp}/c2.npz")
    np.savez(f"{tmp}/t.npz", mats=np.zeros((5, 3, 10)), subject_ids=np.array(["a", "b"]))
    with pytest.raises(D.DataContractError, match="n_samples, n_regions, n_timepoints"):
        D.load_timeseries(f"{tmp}/t.npz")


def test_asymmetric_connectome_rejected(tmp):
    D.save_connectomes(f"{tmp}/c.npz", make_conn(4, ["a", "b"], symmetric=False))
    with pytest.raises(D.DataContractError, match="symmetric"):
        D.load_connectomes(f"{tmp}/c.npz")


def test_nonfinite_rejected(tmp):
    c = make_conn(3, ["a", "b"])
    c.mats[0, 1, 1] = c.mats[1, 0, 1] = np.nan
    D.save_connectomes(f"{tmp}/c.npz", c)
    with pytest.raises(D.DataContractError, match="non-finite"):
        D.load_connectomes(f"{tmp}/c.npz")


def test_region_order_checked_when_available():
    c = make_conn(3, ["a"])
    c.region_ids = np.array(["r1", "r2", "r3"])
    D.check_atlas(c, expected_atlas="A", n_regions=3, region_ids=np.array(["r1", "r2", "r3"]), context="t")
    with pytest.raises(D.DataContractError, match="region order"):
        D.check_atlas(c, expected_atlas="A", n_regions=3, region_ids=np.array(["r2", "r1", "r3"]), context="t")
    c.region_ids = None
    note = D.check_atlas(c, expected_atlas="A", n_regions=3, region_ids=np.array(["r1", "r2", "r3"]), context="t")
    assert "NOT verified" in note


def test_participant_overlap_rejected():
    with pytest.raises(D.DataContractError, match="overlap"):
        D.assert_disjoint_subjects({"support": ["a", "b"], "test": ["b", "c"]})
    D.assert_disjoint_subjects({"support": ["a"], "test": ["c"]})


def test_sessions_form_visit_keys(tmp):
    df = pd.DataFrame({"subject_id": ["a", "a", "b"], "session_id": ["1", "2", "1"], "y": [1, 2, 3]})
    df.to_csv(f"{tmp}/p.csv", index=False)
    t = D.load_phenotypes(f"{tmp}/p.csv")
    c = D.ConnectomeSet(np.zeros((2, 2, 3)), np.array(["b", "a", "a"]), np.array(["1", "2", "1"]), "A")
    y = D.phenotypes_for(c, t, ["y"])[:, 0]
    assert list(y) == [3, 2, 1]
    c_nosess = make_conn(2, ["a", "b"])
    with pytest.raises(D.DataContractError, match="session"):
        D.phenotypes_for(c_nosess, t, ["y"])


def test_missing_value_tokens_and_bad_values(tmp):
    write_table(f"{tmp}/p.csv", ["a", "b", "c"], {"y": ["1.5", "NA", ""]})
    t = D.load_phenotypes(f"{tmp}/p.csv")
    assert t.values["y"].iloc[0] == 1.5 and t.values["y"].isna().sum() == 2
    write_table(f"{tmp}/q.csv", ["a"], {"y": ["oops"]})
    with pytest.raises(D.DataContractError, match="non-numeric"):
        D.load_phenotypes(f"{tmp}/q.csv")
