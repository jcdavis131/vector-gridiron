"""Red-zone opportunity from nflverse play-by-play.

Aggregates player-week:
  rz_tgt_share   — player's RZ (yardline_100<=20) targets / team RZ targets
  rz_carry_share — player's RZ carries / team RZ carries
  inside5_share  — player's inside-5 carries / team inside-5 carries

Writes pipeline/data/rz_index.json for build_features (leakage-safe: callers
must use PRIOR weeks only).

Run:  python pipeline/build_rz.py [--offline]
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import nfl_data as nfl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "pipeline" / "data"
COLS = [
    "season", "week", "season_type", "yardline_100", "play_type",
    "passer_player_id", "receiver_player_id", "rusher_player_id",
    "posteam",
]


def aggregate_year(year: int, offline: bool = False) -> dict:
    path = nfl.pbp_parquet_path(year, offline)
    if path is None:
        print(f"  pbp {year}: missing")
        return {}
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=COLS)
    df = table.to_pandas()
    df = df[df["season_type"] == "REG"].copy()
    df["yardline_100"] = df["yardline_100"].fillna(99)
    df["week"] = df["week"].astype(int)

    # Per (week, team) denominators + per (week, player) numerators
    team_rz_tgt = defaultdict(float)
    team_rz_car = defaultdict(float)
    team_i5_car = defaultdict(float)
    pl_rz_tgt = defaultdict(float)
    pl_rz_car = defaultdict(float)
    pl_i5_car = defaultdict(float)

    for row in df.itertuples(index=False):
        wk = int(row.week)
        team = row.posteam or ""
        if not team:
            continue
        yl = float(row.yardline_100)
        in_rz = yl <= 20
        in_i5 = yl <= 5
        pt = row.play_type or ""
        if pt == "pass" and in_rz and row.receiver_player_id:
            team_rz_tgt[(wk, team)] += 1
            pl_rz_tgt[(wk, row.receiver_player_id)] += 1
        if pt == "run" and in_rz and row.rusher_player_id:
            team_rz_car[(wk, team)] += 1
            pl_rz_car[(wk, row.rusher_player_id)] += 1
        if pt == "run" and in_i5 and row.rusher_player_id:
            team_i5_car[(wk, team)] += 1
            pl_i5_car[(wk, row.rusher_player_id)] += 1

    # Need team for each player-week — from any play they appeared on
    player_team = {}
    for row in df.itertuples(index=False):
        wk = int(row.week)
        team = row.posteam or ""
        for pid in (row.receiver_player_id, row.rusher_player_id, row.passer_player_id):
            if pid and team:
                player_team[(wk, pid)] = team

    out = {}
    players = set(pl_rz_tgt) | set(pl_rz_car) | set(pl_i5_car)
    for (wk, pid) in players:
        team = player_team.get((wk, pid), "")
        if not team:
            continue
        tt = team_rz_tgt.get((wk, team), 0.0) or 0.0
        tc = team_rz_car.get((wk, team), 0.0) or 0.0
        ti = team_i5_car.get((wk, team), 0.0) or 0.0
        out[f"{pid}|{year}|{wk}"] = {
            "rz_tgt_share": (pl_rz_tgt.get((wk, pid), 0.0) / tt) if tt else 0.0,
            "rz_carry_share": (pl_rz_car.get((wk, pid), 0.0) / tc) if tc else 0.0,
            "inside5_share": (pl_i5_car.get((wk, pid), 0.0) / ti) if ti else 0.0,
            "_has": 1.0 if (tt or tc or ti) else 0.0,
        }
    print(f"  pbp {year}: {len(out)} player-weeks with RZ activity")
    return out


def build(last_season: int | None = None, offline: bool = False) -> dict:
    last = last_season or nfl.latest_stats_season(offline)
    seasons = list(range(nfl.FIRST_SEASON, last + 1))
    print(f"building RZ index from pbp {seasons[0]}-{seasons[-1]} ...")
    t0 = time.time()
    idx = {}
    for y in seasons:
        idx.update(aggregate_year(y, offline))
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "rz_index.json"
    meta = {
        "built": time.strftime("%Y-%m-%d"),
        "seasons": seasons,
        "n_rows": len(idx),
        "keys": ["rz_tgt_share", "rz_carry_share", "inside5_share"],
        "source": "nflverse play_by_play parquet",
        "rows": idx,
    }
    path.write_text(json.dumps(meta), encoding="utf-8")
    print(f"wrote {path} ({len(idx)} rows, {time.time() - t0:.0f}s)")
    return meta


def load_index() -> dict:
    path = DATA / "rz_index.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("rows", {})


if __name__ == "__main__":
    raise SystemExit(0 if build(offline="--offline" in sys.argv) else 1)
