"""
Vector Gridiron — real nflverse data pipeline.

Fetches nflverse public *player weekly stats* (plain HTTP CSV from the
``nflverse-data`` GitHub releases — no API key, no auth) and maps the columns
onto the exact feature families / training-matrix schema that
``pipeline/train_mtnn.py`` consumes, so training runs unchanged on real data.

What it produces
----------------
An ``.npz`` bundle at ``pipeline/data/train_matrix.npz`` (the path the trainer
reads) with the SAME keys the synthetic generator emits
(:func:`pipeline.train_mtnn.synthetic_matrix`)::

    X            [N, 160] float32   robust feature matrix (family-ordered)
    M            [N, 160] float32   mask 0/1 (1 = real value present)
    fpts_next    [N]      float32   next-game PPR fantasy points (target)
    pos          [N]      int64     0..3 -> QB/RB/WR/TE
    seasons      [N]      str        e.g. "2023"
    season_ids   [N]      int64      season - min(season)
    features     [160]    str        feature names (family-ordered + padding)
    player_ids   [N]      str        player display name / gsis id

Feature families (must match ``model.DEFAULT_FAM_DIMS`` and the contiguous,
alphabetically-sorted slice layout that ``train_mtnn.family_slices_from_dims``
rebuilds). Families we can source from player weekly stats are filled with real
values; families that live in other nflverse releases (snaps, weather, vegas,
def_vs_pos, redzone, age) are emitted as masked columns (value 0, mask 0) so the
schema and the RealMLP ``cat([x*m, m])`` masking path stay identical to synthetic.

  usage      real   receiving/passing/target usage
  rushing    real   carries / rushing yards / epa / ...
  form       real   lagged & rolling fantasy-point form (leakage-safe)
  rest       real   schedule-derived (week, weeks-since-last, playoffs, ...)
  snaps      masked (separate nflverse snap_counts release)
  age        masked (separate rosters release)
  weather    masked (separate games/weather source)
  vegas      masked (separate betting lines source)
  def_vs_pos masked (derived matchup source)
  redzone    masked (separate pbp aggregation)

Usage
-----
    python pipeline/fetch_nflverse.py --seasons 2021 2022 2023
    python pipeline/fetch_nflverse.py --dry-run          # tiny slice, prints shape, writes nothing
    python pipeline/train_mtnn.py                         # trains on the real matrix if present

Offline / errors: prints a clear message and exits non-zero (does not write a
partial matrix). The synthetic fallback in ``train_mtnn.py`` remains available.

Provenance & license: see ``docs/DATA_SOURCES.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

# Feature-family dims are the single source of truth in model.py, but that module
# imports torch. The data pipeline must run with only numpy+pandas, so import it
# defensively and fall back to a torch-free literal copy (kept in sync with model).
try:  # pragma: no cover - exercised implicitly by import environment
    from .model import DEFAULT_FAM_DIMS
except Exception:  # torch (model.py dep) may be missing/broken
    try:
        from model import DEFAULT_FAM_DIMS
    except Exception:
        # torch (a model.py dependency) unavailable — use the known dims.
        DEFAULT_FAM_DIMS = {
            "usage": 16,
            "snaps": 12,
            "age": 8,
            "weather": 10,
            "vegas": 8,
            "rest": 10,
            "def_vs_pos": 16,
            "form": 20,
            "rushing": 30,
            "redzone": 20,
        }

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "pipeline" / "data"
DEFAULT_OUTPUT = DATA_DIR / "train_matrix.npz"
META_PATH = DATA_DIR / "train_matrix.meta.json"

# nflverse-data public release URL templates for weekly player stats.
# The project has renamed these files over time; we try each candidate in order.
# All are plain HTTPS CSV on GitHub releases — no API key required.
NFLVERSE_URL_TEMPLATES: list[str] = [
    "https://github.com/nflverse/nflverse-data/releases/download/player_stats/stats_player_week_{season}.csv",
    "https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_{season}.csv",
    "https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_{season}.csv.gz",
]

POS_ORDER = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}
KEEP_POS = set(POS_ORDER)
F_TOTAL = 160  # padded matrix width (matches synthetic + model)

# ---------------------------------------------------------------------------
# Feature layout — MUST mirror the alphabetically-sorted, contiguous slicing in
# train_mtnn.family_slices_from_dims(feats, DEFAULT_FAM_DIMS). Each list length
# equals DEFAULT_FAM_DIMS[family]. Names that also appear as columns in the
# engineered frame get REAL values; all others are emitted masked.
# ---------------------------------------------------------------------------
FAMILY_FEATURES: dict[str, list[str]] = {
    "age": [  # masked (rosters release)
        "age",
        "years_exp",
        "draft_round",
        "draft_pick",
        "draft_overall",
        "rookie_flag",
        "entry_year",
        "age_sq",
    ],
    "def_vs_pos": [f"def_vs_pos_{i:02d}" for i in range(16)],  # masked (matchup source)
    "form": [  # real — leakage-safe (prior games only)
        "fpts_lag1",
        "fpts_lag2",
        "fpts_lag3",
        "fpts_roll3_mean",
        "fpts_roll3_std",
        "fpts_roll5_mean",
        "fpts_season_mean",
        "fpts_season_std",
        "fpts_ewm3",
        "fpts_trend",
        "targets_lag1",
        "carries_lag1",
        "rec_yards_lag1",
        "rush_yards_lag1",
        "targets_roll3_mean",
        "carries_roll3_mean",
        "fpts_season_max",
        "fpts_season_min",
        "games_to_date",
        "fpts_lag1_ppr",
    ],
    "redzone": (  # masked (pbp aggregation)
        [
            "rz_targets",
            "rz_carries",
            "rz_receptions",
            "rz_rush_tds",
            "rz_rec_tds",
            "rz_target_share",
            "rz_touches",
            "rz_looks",
            "rz_conv_rate",
            "rz_air_yards",
        ]
        + [f"rz_{i:02d}" for i in range(10)]
    ),
    "rest": [  # real — schedule-derived
        "week",
        "week_norm",
        "weeks_since_last",
        "short_week",
        "bye_prev",
        "is_early_season",
        "is_late_season",
        "season_progress",
        "games_this_season",
        "is_playoffs",
    ],
    "rushing": (  # real (first 9) then masked padding to 30
        [
            "carries",
            "rushing_yards",
            "rushing_tds",
            "rushing_epa",
            "rushing_first_downs",
            "rushing_fumbles",
            "rushing_fumbles_lost",
            "rushing_2pt_conversions",
            "yards_per_carry",
        ]
        + [f"rush_{i:02d}" for i in range(21)]
    ),
    "snaps": (  # masked (snap_counts release)
        ["snap_pct", "off_snaps", "routes_run", "route_pct", "snap_share", "routes_per_target"]
        + [f"snap_{i:02d}" for i in range(6)]
    ),
    "usage": [  # real
        "targets",
        "receptions",
        "target_share",
        "air_yards_share",
        "wopr",
        "racr",
        "receiving_air_yards",
        "receiving_yards_after_catch",
        "attempts",
        "completions",
        "passing_yards",
        "passing_air_yards",
        "receiving_yards",
        "receiving_first_downs",
        "passing_first_downs",
        "special_teams_tds",
    ],
    "vegas": [  # masked (betting lines source)
        "spread",
        "team_total",
        "game_total",
        "implied_team_total",
        "is_favorite",
        "spread_abs",
        "total_over",
        "win_prob",
    ],
    "weather": [  # masked (games/weather source)
        "temp",
        "wind",
        "humidity",
        "precip",
        "is_dome",
        "is_outdoor",
        "wind_chill",
        "altitude",
        "roof_closed",
        "surface_turf",
    ],
}

# Families whose columns are engineered from player weekly stats (real values).
SOURCED_FAMILIES = ("usage", "rushing", "form", "rest")


def build_feature_names() -> list[str]:
    """Contiguous, alphabetically-sorted family order + padding to 160."""
    feats: list[str] = []
    for fam in sorted(DEFAULT_FAM_DIMS):
        cols = FAMILY_FEATURES[fam]
        if len(cols) != DEFAULT_FAM_DIMS[fam]:
            raise ValueError(f"family {fam!r} has {len(cols)} names but DEFAULT_FAM_DIMS says {DEFAULT_FAM_DIMS[fam]}")
        feats.extend(cols)
    n_pad = F_TOTAL - len(feats)
    if n_pad < 0:
        raise ValueError(f"family features sum to {len(feats)} > {F_TOTAL}")
    feats.extend(f"pad_{i:02d}" for i in range(n_pad))
    return feats


FEATURES = build_feature_names()


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
def _http_get(url: str, timeout: float = 60.0) -> bytes:
    """GET raw bytes, honoring env proxies / CA bundle via urllib defaults."""
    req = urllib.request.Request(url, headers={"User-Agent": "vector-gridiron/fetch_nflverse"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_season(season: int, timeout: float = 60.0) -> pd.DataFrame:
    """Fetch one season of nflverse weekly player stats, trying known URL templates."""
    errors: list[str] = []
    for tmpl in NFLVERSE_URL_TEMPLATES:
        url = tmpl.format(season=season)
        try:
            raw = _http_get(url, timeout=timeout)
        except urllib.error.HTTPError as e:  # 404 for a template that no longer exists
            errors.append(f"{url} -> HTTP {e.code}")
            continue
        except urllib.error.URLError as e:  # offline / DNS / TLS
            raise ConnectionError(
                f"Could not reach nflverse ({url}): {e.reason}. "
                "Are you offline? See docs/DATA_SOURCES.md. "
                "Use `python pipeline/train_mtnn.py --synthetic` for an offline smoke run."
            ) from e
        if url.endswith(".gz"):
            import gzip

            raw = gzip.decompress(raw)
        df = pd.read_csv(StringIO(raw.decode("utf-8")), low_memory=False)
        df["season"] = season
        return df
    raise FileNotFoundError(
        f"No nflverse weekly-stats file found for season {season}. Tried:\n  " + "\n  ".join(errors)
    )


def fetch_player_stats(seasons: list[int], timeout: float = 60.0) -> pd.DataFrame:
    """Fetch and concatenate weekly player stats for the given seasons."""
    frames = [fetch_season(s, timeout=timeout) for s in seasons]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Column mapping / feature engineering  (NO network — unit-testable)
# ---------------------------------------------------------------------------
def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return column as float, or an all-NaN series if the column is absent."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take raw nflverse weekly rows -> one engineered frame with columns named
    exactly like the SOURCED feature names, plus id/target/meta columns.

    Leakage-safe: form features use only prior games; the target `fpts_next` is
    the *following* game's PPR points. Rows with no following game are dropped.
    """
    df = df.copy()

    # position (accept `position` or `position_group`)
    pos_col = "position" if "position" in df.columns else ("position_group" if "position_group" in df.columns else None)
    if pos_col is None:
        raise ValueError("nflverse frame has no 'position' or 'position_group' column")
    df["_pos"] = df[pos_col].astype(str).str.upper()
    df = df[df["_pos"].isin(KEEP_POS)].copy()

    # player id / name
    id_col = next((c for c in ("player_id", "gsis_id", "player_display_name", "player_name") if c in df.columns), None)
    if id_col is None:
        raise ValueError("nflverse frame has no player id/name column")
    name_col = next((c for c in ("player_display_name", "player_name", "player_id") if c in df.columns), id_col)
    df["_pid"] = df[id_col].astype(str)
    df["_name"] = df[name_col].astype(str)

    # season / week
    if "season" not in df.columns or "week" not in df.columns:
        raise ValueError("nflverse frame needs 'season' and 'week' columns")
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["season", "week"]).copy()
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)

    # fantasy points — prefer PPR
    fpts = _col(df, "fantasy_points_ppr")
    if fpts.isna().all():
        fpts = _col(df, "fantasy_points")
    df["_fpts"] = fpts.fillna(0.0)

    df = df.sort_values(["_pid", "season", "week"]).reset_index(drop=True)
    g = df.groupby(["_pid", "season"], sort=False)

    out = pd.DataFrame(index=df.index)

    # --- target: next game PPR fantasy points ---
    out["fpts_next"] = g["_fpts"].shift(-1)

    # --- usage (real) ---
    usage_direct = [
        "targets",
        "receptions",
        "target_share",
        "air_yards_share",
        "wopr",
        "racr",
        "receiving_air_yards",
        "receiving_yards_after_catch",
        "attempts",
        "completions",
        "passing_yards",
        "passing_air_yards",
        "receiving_yards",
        "receiving_first_downs",
        "passing_first_downs",
        "special_teams_tds",
    ]
    for c in usage_direct:
        out[c] = _col(df, c)

    # --- rushing (real) ---
    rushing_direct = [
        "carries",
        "rushing_yards",
        "rushing_tds",
        "rushing_epa",
        "rushing_first_downs",
        "rushing_fumbles",
        "rushing_fumbles_lost",
        "rushing_2pt_conversions",
    ]
    for c in rushing_direct:
        out[c] = _col(df, c)
    carries = _col(df, "carries")
    out["yards_per_carry"] = np.where(carries > 0, _col(df, "rushing_yards") / carries, np.nan)

    # --- form (real, leakage-safe: prior games only) ---
    fp = df["_fpts"]
    out["fpts_lag1"] = g["_fpts"].shift(1)
    out["fpts_lag2"] = g["_fpts"].shift(2)
    out["fpts_lag3"] = g["_fpts"].shift(3)
    out["fpts_lag1_ppr"] = out["fpts_lag1"]
    out["fpts_roll3_mean"] = g["_fpts"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    out["fpts_roll3_std"] = g["_fpts"].transform(lambda s: s.shift(1).rolling(3, min_periods=2).std())
    out["fpts_roll5_mean"] = g["_fpts"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    out["fpts_season_mean"] = g["_fpts"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    out["fpts_season_std"] = g["_fpts"].transform(lambda s: s.shift(1).expanding(min_periods=2).std())
    out["fpts_season_max"] = g["_fpts"].transform(lambda s: s.shift(1).expanding(min_periods=1).max())
    out["fpts_season_min"] = g["_fpts"].transform(lambda s: s.shift(1).expanding(min_periods=1).min())
    out["fpts_ewm3"] = g["_fpts"].transform(lambda s: s.shift(1).ewm(span=3, min_periods=1).mean())
    out["fpts_trend"] = out["fpts_lag1"] - out["fpts_lag3"]
    out["games_to_date"] = g["_fpts"].cumcount().astype(float)
    tgt = _col(df, "targets")
    car = _col(df, "carries")
    out["targets_lag1"] = tgt.groupby([df["_pid"], df["season"]]).shift(1)
    out["carries_lag1"] = car.groupby([df["_pid"], df["season"]]).shift(1)
    out["rec_yards_lag1"] = _col(df, "receiving_yards").groupby([df["_pid"], df["season"]]).shift(1)
    out["rush_yards_lag1"] = _col(df, "rushing_yards").groupby([df["_pid"], df["season"]]).shift(1)
    out["targets_roll3_mean"] = tgt.groupby([df["_pid"], df["season"]]).transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    out["carries_roll3_mean"] = car.groupby([df["_pid"], df["season"]]).transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    _ = fp  # (kept for readability of the form block)

    # --- rest (real, schedule-derived) ---
    week = df["week"].astype(float)
    out["week"] = week
    out["week_norm"] = week / 18.0
    wsl = g["week"].diff()
    out["weeks_since_last"] = wsl
    out["short_week"] = np.nan  # exact rest-days need a games source; masked here
    out["bye_prev"] = (wsl > 1).astype(float).where(wsl.notna(), np.nan)
    out["is_early_season"] = (week <= 4).astype(float)
    out["is_late_season"] = (week >= 14).astype(float)
    out["season_progress"] = week / 18.0
    out["games_this_season"] = g["week"].cumcount().astype(float) + 1.0
    out["is_playoffs"] = (week > 18).astype(float)

    # meta
    out["_pid"] = df["_pid"].to_numpy()
    out["_name"] = df["_name"].to_numpy()
    out["_pos"] = df["_pos"].to_numpy()
    out["season"] = df["season"].to_numpy()

    # keep only rows with a real next-game target
    out = out[out["fpts_next"].notna()].reset_index(drop=True)
    return out


def build_training_matrix(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Map raw nflverse weekly rows to the trainer's npz payload.

    Returns a dict with the exact keys the synthetic generator writes:
    X, M, fpts_next, pos, seasons, season_ids, features, player_ids.
    """
    eng = engineer_features(df)
    n = len(eng)
    if n == 0:
        raise ValueError("no usable rows after feature engineering (need >=2 games per player-season)")

    x = np.zeros((n, F_TOTAL), dtype=np.float32)
    m = np.zeros((n, F_TOTAL), dtype=np.float32)
    for j, feat in enumerate(FEATURES):
        if feat in eng.columns:
            col = pd.to_numeric(eng[feat], errors="coerce").to_numpy(dtype=np.float64)
            valid = np.isfinite(col)
            x[valid, j] = col[valid].astype(np.float32)
            m[valid, j] = 1.0
        # else: reserved/masked family -> stays 0 / mask 0

    fpts_next = eng["fpts_next"].to_numpy(dtype=np.float32)
    pos = np.array([POS_ORDER.get(str(p).upper(), 1) for p in eng["_pos"]], dtype=np.int64)
    seasons = np.array([str(int(s)) for s in eng["season"]], dtype=object)
    season_int = eng["season"].to_numpy(dtype=np.int64)
    season_ids = (season_int - int(season_int.min())).astype(np.int64)
    player_ids = np.array([str(p) for p in eng["_name"]], dtype=object)
    features = np.array(FEATURES, dtype=object)

    return {
        "X": x,
        "M": m,
        "fpts_next": fpts_next,
        "pos": pos,
        "seasons": seasons,
        "season_ids": season_ids,
        "features": features,
        "player_ids": player_ids,
    }


def write_matrix(payload: dict[str, np.ndarray], out_path: Path, seasons: list[int]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    meta = {
        "source": "nflverse",
        "dataset": "player_stats (weekly)",
        "seasons": seasons,
        "n_rows": int(payload["X"].shape[0]),
        "n_features": int(payload["X"].shape[1]),
        "sourced_families": list(SOURCED_FAMILIES),
        "url_templates": NFLVERSE_URL_TEMPLATES,
        "license": "nflverse data — CC-BY 4.0 (see docs/DATA_SOURCES.md)",
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch real nflverse weekly stats -> gridiron training matrix")
    ap.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=[2021, 2022, 2023],
        help="seasons to fetch (default: 2021 2022 2023)",
    )
    ap.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="output .npz path")
    ap.add_argument("--timeout", type=float, default=60.0, help="per-request HTTP timeout (s)")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="fetch only the first season, take a tiny slice, print the matrix shape, write nothing",
    )
    args = ap.parse_args(argv)

    try:
        if args.dry_run:
            season = args.seasons[0]
            print(f"[fetch] dry-run: fetching a slice of {season} ...")
            df = fetch_season(season, timeout=args.timeout).head(500)
            payload = build_training_matrix(df)
            print(
                f"[fetch] dry-run OK  X={payload['X'].shape} M={payload['M'].shape} "
                f"features={len(payload['features'])} rows={payload['X'].shape[0]}"
            )
            print(f"[fetch] mask coverage (mean over sourced cols): {float(payload['M'].mean()):.3f}")
            print("[fetch] dry-run: nothing written.")
            return 0

        print(f"[fetch] fetching nflverse weekly stats for seasons {args.seasons} ...")
        df = fetch_player_stats(args.seasons, timeout=args.timeout)
        print(f"[fetch] fetched {len(df)} raw weekly rows")
        payload = build_training_matrix(df)
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        write_matrix(payload, out_path, args.seasons)
        print(
            f"[fetch] wrote {out_path}  X={payload['X'].shape}  target mean "
            f"{float(payload['fpts_next'].mean()):.2f}"
        )
        print(f"[fetch] provenance stamped at {META_PATH}")
        print("[fetch] now run:  python pipeline/train_mtnn.py   (trains on the real matrix)")
        return 0
    except (ConnectionError, FileNotFoundError, ValueError) as e:
        print(f"[fetch] ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
