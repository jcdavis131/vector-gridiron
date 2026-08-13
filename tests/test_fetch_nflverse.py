"""
Offline tests for pipeline/fetch_nflverse.py column-mapping / matrix build.

NO network: everything runs against the in-repo CSV fixture. Asserts the built
matrix has the exact shape/keys/dtypes that pipeline/train_mtnn.py consumes and
that the family layout matches model.DEFAULT_FAM_DIMS.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

import fetch_nflverse as fx  # noqa: E402

# DEFAULT_FAM_DIMS is re-exported by the fetcher (torch-free); this is the same
# dict model.py defines and train_mtnn.py slices against.
DEFAULT_FAM_DIMS = fx.DEFAULT_FAM_DIMS

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "nflverse_player_stats_sample.csv"


def _trainer():
    """Import the trainer lazily — it requires torch, which the data pipeline does not."""
    pytest.importorskip("torch", reason="train_mtnn imports torch (model.py)")
    import train_mtnn

    return train_mtnn


# The exact key set the synthetic generator emits (train_mtnn.synthetic_matrix).
SYNTH_KEYS = {"X", "M", "fpts_next", "pos", "seasons", "season_ids", "features", "player_ids"}


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return pd.read_csv(FIXTURE)


@pytest.fixture
def payload(raw_df: pd.DataFrame) -> dict:
    return fx.build_training_matrix(raw_df)


def test_feature_layout_matches_model_families():
    """160-wide, family names present in sorted order with correct dims."""
    feats = fx.build_feature_names()
    assert len(feats) == fx.F_TOTAL == 160
    # each family occupies a contiguous block of DEFAULT_FAM_DIMS[fam] names
    offset = 0
    for fam in sorted(DEFAULT_FAM_DIMS):
        dim = DEFAULT_FAM_DIMS[fam]
        assert feats[offset : offset + dim] == fx.FAMILY_FEATURES[fam]
        offset += dim
    # trainer's slice reconstruction agrees on dims
    tm = _trainer()
    _, fam_dims_calc = tm.family_slices_from_dims(feats, DEFAULT_FAM_DIMS)
    assert fam_dims_calc == DEFAULT_FAM_DIMS


def test_payload_keys_match_synthetic_schema(payload: dict):
    assert set(payload.keys()) == SYNTH_KEYS


def test_matrix_shapes_and_dtypes(payload: dict):
    n = payload["X"].shape[0]
    assert n > 0
    assert payload["X"].shape == (n, 160)
    assert payload["M"].shape == (n, 160)
    assert payload["X"].dtype == np.float32
    assert payload["M"].dtype == np.float32
    assert payload["fpts_next"].shape == (n,)
    assert payload["fpts_next"].dtype == np.float32
    assert payload["pos"].shape == (n,)
    assert payload["pos"].dtype == np.int64
    assert payload["season_ids"].shape == (n,)
    assert payload["seasons"].shape == (n,)
    assert payload["player_ids"].shape == (n,)
    assert len(payload["features"]) == 160


def test_mask_is_binary_and_zero_where_masked(payload: dict):
    m = payload["M"]
    assert set(np.unique(m)).issubset({0.0, 1.0})
    # X must be exactly 0 wherever the mask is 0 (RealMLP cat([x*m, m]) invariant)
    assert np.all(payload["X"][m == 0.0] == 0.0)


def test_sourced_families_have_real_values(payload: dict):
    """usage/rushing/form/rest columns carry real (non-masked) data."""
    feats = list(payload["features"])
    m = payload["M"]
    # a couple of concrete real columns from the fixture
    for name in ("targets", "carries", "rushing_yards", "fpts_lag1", "week"):
        j = feats.index(name)
        assert m[:, j].sum() > 0, f"expected real values for {name!r}"


def test_masked_families_are_fully_masked(payload: dict):
    """Families sourced from other nflverse releases stay masked (all zero)."""
    feats = list(payload["features"])
    m = payload["M"]
    for name in ("temp", "spread", "snap_pct", "age", "rz_targets"):
        j = feats.index(name)
        assert m[:, j].sum() == 0, f"expected {name!r} fully masked"


def test_next_game_target_is_leakage_safe(raw_df: pd.DataFrame, payload: dict):
    """fpts_next must equal the player's following-week PPR points."""
    eng = fx.engineer_features(raw_df)
    # Alpha Back week-1 -> next game (week 2) PPR = 17.2 in the fixture
    row = eng[(eng["_name"] == "Alpha Back") & (eng["week"] == 1)]
    assert len(row) == 1
    assert row["fpts_next"].iloc[0] == pytest.approx(17.2)
    # last game of a player-season has no next game -> dropped (finite targets only)
    assert np.all(np.isfinite(payload["fpts_next"]))


def test_positions_mapped_to_codes(payload: dict):
    codes = {int(p) for p in payload["pos"]}
    assert codes.issubset({0, 1, 2, 3})  # QB/RB/WR/TE


def test_roundtrip_through_train_mtnn_load_bundle(tmp_path, payload: dict):
    """The written npz loads cleanly through the trainer's own loader."""
    tm = _trainer()
    out = tmp_path / "train_matrix.npz"
    np.savez_compressed(out, **payload)
    X, M, y, _pos, pids, _season_ids, _seasons, feats, _fam_dims, _uids = tm.load_bundle(out)
    n = X.shape[0]
    assert X.shape == (n, 160)
    assert M.shape == (n, 160)
    assert len(feats) == 160
    assert y.shape == (n,)
    assert len(pids) == n
    # trainer can build family slices from these features
    slices, _dims = tm.family_slices_from_dims(feats, DEFAULT_FAM_DIMS)
    assert sum(len(v) for v in slices.values()) == 150  # 10 families, pad excluded
