"""Build the REAL-data multi-target benchmark dataset for vector-gridiron.

Fetches nflverse weekly player stats (public CSVs on the nflverse-data GitHub
releases — real data, no API key) via the repo's committed pipeline
(``pipeline/fetch_nflverse.py``) and constructs the three registry targets
declared by vector-bench's gridiron DomainSpec:

- ``next_game_fpts``  — PPR fantasy points in the player's NEXT game
- ``next_game_yards`` — rushing + receiving + passing yards in the NEXT game
- ``next_game_tds``   — rushing + receiving + passing TDs in the NEXT game

Label construction (leakage-safe, forward-shifted)
--------------------------------------------------
Entities are players (nflverse ``player_id``); time is (season, week). For a
row (player, game g) the label is the stat in that player's next recorded game
*within the same season* (``groupby(player, season).shift(-1)``). Rows whose
player has no following game in that season are DROPPED (identical to the
committed pipeline's ``fpts_next`` handling), so every emitted row has all
three labels observed; the per-target masks in the npz are all-ones and exist
for exchange-schema compatibility. Features for row (player, g) are built by
``pipeline.fetch_nflverse.engineer_features`` from information available up to
and including game g only (current-game stat lines + lagged/rolling form +
schedule) — the label always lives strictly in the future (week(g+1) > week(g)).

FPTS is nflverse's ``fantasy_points_ppr`` column directly (NaN -> 0.0 before
shifting, as in the committed pipeline); no formula is applied in-repo.

Alignment guarantee: the extra labels are computed by re-applying the exact
filter/sort/group transform ``engineer_features`` uses, then asserted row-for-row
identical on (player_id, season, week) AND on the fpts_next values themselves.

Split spec (temporal, strictly forward)
---------------------------------------
- time_key = season * 100 + week
- train = seasons 2019-2022, val = 2023, test = 2024 (indices in the npz)
- the vector-bench harness cut is time_cut = 202400: baselines fit on
  everything strictly before 2024 (train+val), test on 2024. The MTNN trains
  on 2019-2022 only and early-stops on 2023 — it never sees a test row.

Usage
-----
    python bench/build_dataset.py                          # writes bench/data/
    python bench/build_dataset.py --seasons 2019 ... 2024 --out <path>

Honesty: this script fetches REAL data or fails loudly. There is no synthetic
fallback here, deliberately.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

from fetch_nflverse import (  # noqa: E402  (repo pipeline, path-inserted above)
    FEATURES,
    KEEP_POS,
    NFLVERSE_URL_TEMPLATES,
    POS_ORDER,
    engineer_features,
    fetch_player_stats,
)

DEFAULT_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]
TRAIN_SEASONS = (2019, 2022)  # inclusive range
VAL_SEASON = 2023
TEST_SEASON = 2024
TIME_CUT = 202400  # harness temporal cut: train < cut <= test
TARGETS = ("next_game_fpts", "next_game_yards", "next_game_tds")

OUT_DIR = ROOT / "bench" / "data"
DEFAULT_OUT = OUT_DIR / "gridiron_bench_dataset.npz"
DEFAULT_DATASHEET = OUT_DIR / "datasheet.json"


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """Numeric column or all-NaN if absent (mirrors fetch_nflverse._col)."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def build_label_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Re-apply engineer_features' exact row transform and compute all labels.

    Returns the KEPT rows (those with a next game) with columns:
    _pid, _name, _pos, season, week, cur_fpts, cur_yards, cur_tds,
    y_fpts, y_yards, y_tds.
    """
    df = raw.copy()

    pos_col = "position" if "position" in df.columns else ("position_group" if "position_group" in df.columns else None)
    if pos_col is None:
        raise ValueError("nflverse frame has no 'position' or 'position_group' column")
    df["_pos"] = df[pos_col].astype(str).str.upper()
    df = df[df["_pos"].isin(KEEP_POS)].copy()

    id_col = next(
        (c for c in ("player_id", "gsis_id", "player_display_name", "player_name") if c in df.columns),
        None,
    )
    if id_col is None:
        raise ValueError("nflverse frame has no player id/name column")
    name_col = next(
        (c for c in ("player_display_name", "player_name", "player_id") if c in df.columns),
        id_col,
    )
    df["_pid"] = df[id_col].astype(str)
    df["_name"] = df[name_col].astype(str)

    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["season", "week"]).copy()
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)

    # fantasy points — prefer PPR, exactly as engineer_features does
    fpts = _num(df, "fantasy_points_ppr")
    if fpts.isna().all():
        fpts = _num(df, "fantasy_points")
    df["_fpts"] = fpts.fillna(0.0)

    # current-game component stats (NaN == stat not applicable -> 0)
    df["_yards"] = (
        _num(df, "rushing_yards").fillna(0.0)
        + _num(df, "receiving_yards").fillna(0.0)
        + _num(df, "passing_yards").fillna(0.0)
    )
    df["_tds"] = (
        _num(df, "rushing_tds").fillna(0.0)
        + _num(df, "receiving_tds").fillna(0.0)
        + _num(df, "passing_tds").fillna(0.0)
    )

    df = df.sort_values(["_pid", "season", "week"]).reset_index(drop=True)
    g = df.groupby(["_pid", "season"], sort=False)

    df["y_fpts"] = g["_fpts"].shift(-1)
    df["y_yards"] = g["_yards"].shift(-1)
    df["y_tds"] = g["_tds"].shift(-1)
    df = df.rename(columns={"_fpts": "cur_fpts", "_yards": "cur_yards", "_tds": "cur_tds"})

    keep = df["y_fpts"].notna()
    # y_fpts is NaN exactly at each (player, season)'s last game; _yards/_tds
    # were fillna(0) before shifting, so the three label masks coincide.
    assert bool((df["y_yards"].notna() == keep).all())
    assert bool((df["y_tds"].notna() == keep).all())
    return df.loc[keep].reset_index(drop=True)


def build(seasons: list[int]) -> tuple[dict[str, np.ndarray], dict]:
    t0 = time.time()
    print(f"[build] fetching REAL nflverse weekly stats for seasons {seasons} ...")
    raw = fetch_player_stats(seasons)
    print(f"[build] fetched {len(raw)} raw weekly rows in {time.time() - t0:.1f}s")

    # Feature matrix comes from the repo's committed pipeline, unchanged.
    eng = engineer_features(raw)
    lab = build_label_frame(raw)

    # --- row-for-row alignment assertions (the leakage/consistency guard) ---
    if len(eng) != len(lab):
        raise AssertionError(f"alignment failure: eng has {len(eng)} rows, labels {len(lab)}")
    if not np.array_equal(eng["_pid"].to_numpy(), lab["_pid"].to_numpy()):
        raise AssertionError("alignment failure: player ids differ")
    if not np.array_equal(eng["season"].to_numpy(), lab["season"].to_numpy()):
        raise AssertionError("alignment failure: seasons differ")
    if not np.array_equal(eng["week"].to_numpy().astype(int), lab["week"].to_numpy()):
        raise AssertionError("alignment failure: weeks differ")
    if not np.allclose(eng["fpts_next"].to_numpy(dtype=float), lab["y_fpts"].to_numpy(dtype=float)):
        raise AssertionError("alignment failure: fpts_next values differ")
    print(f"[build] alignment OK: {len(eng)} rows, labels match engineer_features exactly")

    n = len(eng)
    x = np.zeros((n, len(FEATURES)), dtype=np.float32)
    m = np.zeros((n, len(FEATURES)), dtype=np.float32)
    for j, feat in enumerate(FEATURES):
        if feat in eng.columns:
            col = pd.to_numeric(eng[feat], errors="coerce").to_numpy(dtype=np.float64)
            valid = np.isfinite(col)
            x[valid, j] = col[valid].astype(np.float32)
            m[valid, j] = 1.0

    season = lab["season"].to_numpy(dtype=np.int64)
    week = lab["week"].to_numpy(dtype=np.int64)
    time_key = season * 100 + week
    pos = np.array([POS_ORDER.get(str(p).upper(), 1) for p in lab["_pos"]], dtype=np.int64)

    train_idx = np.where((season >= TRAIN_SEASONS[0]) & (season <= TRAIN_SEASONS[1]))[0]
    val_idx = np.where(season == VAL_SEASON)[0]
    test_idx = np.where(season == TEST_SEASON)[0]
    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise ValueError(
            f"split produced an empty side (train={len(train_idx)}, "
            f"val={len(val_idx)}, test={len(test_idx)}) for seasons {sorted(set(season))}"
        )

    y = {
        "next_game_fpts": lab["y_fpts"].to_numpy(dtype=np.float32),
        "next_game_yards": lab["y_yards"].to_numpy(dtype=np.float32),
        "next_game_tds": lab["y_tds"].to_numpy(dtype=np.float32),
    }
    cur = {
        "next_game_fpts": lab["cur_fpts"].to_numpy(dtype=np.float32),
        "next_game_yards": lab["cur_yards"].to_numpy(dtype=np.float32),
        "next_game_tds": lab["cur_tds"].to_numpy(dtype=np.float32),
    }

    payload: dict[str, np.ndarray] = {
        "X": x,
        "M": m,
        "features": np.array(FEATURES, dtype=object),
        "entity_ids": lab["_pid"].to_numpy(dtype=object),
        "player_names": lab["_name"].to_numpy(dtype=object),
        "pos": pos,
        "season": season,
        "week": week,
        "time_key": time_key,
        "train_idx": train_idx.astype(np.int64),
        "val_idx": val_idx.astype(np.int64),
        "test_idx": test_idx.astype(np.int64),
    }
    for t in TARGETS:
        payload[f"y_{t}"] = y[t]
        payload[f"mask_{t}"] = np.ones(n, dtype=np.uint8)  # invalid rows were dropped
        payload[f"cur_{t}"] = cur[t]

    def _stats(a: np.ndarray) -> dict:
        return {
            "observed": int(a.shape[0]),
            "mean": round(float(a.mean()), 4),
            "std": round(float(a.std()), 4),
            "min": round(float(a.min()), 4),
            "max": round(float(a.max()), 4),
        }

    datasheet = {
        "domain": "gridiron",
        "source": "nflverse-data GitHub releases (player_stats weekly CSVs) — REAL data",
        "source_url_templates": NFLVERSE_URL_TEMPLATES,
        "license": "nflverse data — CC-BY 4.0 (see docs/DATA_SOURCES.md)",
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seasons": seasons,
        "rows": int(n),
        "entities": int(len(set(lab["_pid"]))),
        "entity_kind": "NFL player (nflverse player_id), positions QB/RB/WR/TE",
        "time_range": {
            "min_time_key": int(time_key.min()),
            "max_time_key": int(time_key.max()),
            "time_key": "season*100 + week",
        },
        "features": (
            f"{len(FEATURES)}-dim per-game vector from pipeline/fetch_nflverse.py "
            "(families usage/rushing/form/rest real; snaps/age/weather/vegas/"
            "def_vs_pos/redzone masked M=0). X is RAW (unscaled); M is the "
            "observability mask. Features for row (player, g) use information "
            "up to and including game g only."
        ),
        "targets": {
            "next_game_fpts": {
                "construction": (
                    "y = nflverse fantasy_points_ppr of the player's next recorded "
                    "game in the same season (groupby(player, season).shift(-1); "
                    "NaN->0 before shift, as in the committed pipeline). Rows with "
                    "no next game are dropped."
                ),
                **_stats(y["next_game_fpts"]),
            },
            "next_game_yards": {
                "construction": (
                    "y = rushing_yards + receiving_yards + passing_yards (each "
                    "NaN->0) of the player's next recorded game in the same season."
                ),
                **_stats(y["next_game_yards"]),
            },
            "next_game_tds": {
                "construction": (
                    "y = rushing_tds + receiving_tds + passing_tds (each NaN->0) "
                    "of the player's next recorded game in the same season."
                ),
                **_stats(y["next_game_tds"]),
            },
        },
        "split": {
            "kind": "temporal",
            "train_seasons": list(range(TRAIN_SEASONS[0], TRAIN_SEASONS[1] + 1)),
            "val_season": VAL_SEASON,
            "test_season": TEST_SEASON,
            "harness_time_cut": TIME_CUT,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "note": (
                "Harness baselines fit on train+val (< time_cut); the MTNN fits "
                "on train only and early-stops on val. Test (2024) is never "
                "visible to any method at fit time."
            ),
        },
        "leakage_note": (
            "Labels are strictly forward-shifted within (player, season); the "
            "label for row (player, week w) uses only the game at week > w. "
            "Form features are lagged/rolling over games <= w (shift(1) based). "
            "Preprocessing for the MTNN is fit on train rows only."
        ),
    }
    return payload, datasheet


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seasons", type=int, nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--datasheet", type=str, default=str(DEFAULT_DATASHEET))
    args = ap.parse_args(argv)

    if TEST_SEASON not in args.seasons or VAL_SEASON not in args.seasons:
        raise SystemExit(f"--seasons must include val season {VAL_SEASON} and test season {TEST_SEASON}")

    payload, datasheet = build(sorted(args.seasons))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    ds_path = Path(args.datasheet)
    ds_path.write_text(json.dumps(datasheet, indent=2) + "\n", encoding="utf-8")
    size_mb = out.stat().st_size / 1e6
    print(f"[build] wrote {out} ({size_mb:.1f} MB) and {ds_path}")
    print(
        f"[build] rows={datasheet['rows']} entities={datasheet['entities']} "
        f"train/val/test={datasheet['split']['n_train']}/"
        f"{datasheet['split']['n_val']}/{datasheet['split']['n_test']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
