import numpy as np
import pytest

from atlas_interop import data as D


def make_conn(n_regions, ids, seed=0, atlas="A", symmetric=True):
    rng = np.random.default_rng(seed)
    mats = rng.normal(size=(n_regions, n_regions, len(ids)))
    if symmetric:
        mats = (mats + mats.transpose(1, 0, 2)) / 2
    return D.ConnectomeSet(mats, np.asarray(ids, dtype=str), None, atlas)


@pytest.fixture
def tmp(tmp_path):
    return str(tmp_path)


def write_table(path, ids, cols: dict):
    import pandas as pd
    df = pd.DataFrame({"subject_id": list(ids), **cols})
    df.to_csv(path, index=False)
    return path
