"""Vector Gridiron pipeline: nflverse weekly player stats (NFL, 2016-2025) ->
per-game statistical-profile vectors -> PCA map + named archetypes ->
assets/vectors.json. Football sibling of Vector Hoops / Vector Pitch --
same shape, same philosophy, bent toward a fantasy-football cockpit.

Design (mirrors the hoops/pitch builds):
- Per-GAME rates from weekly box-score data (games-played-adjusted at the door).
- Season normalization: z-score every feature WITHIN its season, across all
  qualified skill players (QB/RB/WR/TE together). Context-honest -- a 2016
  volume isn't compared against a 2025 mean -- and because positions have
  near-disjoint stat support (QBs pass, WRs receive), position SEPARATES
  naturally in the space instead of being z-normalized away. Position is then
  carried as a color/emergent-cluster label, not a modeled input.
- PCA(3) for the 3D map; k-means(K=8) archetypes named from centroids.
- Fantasy points (Standard / Half-PPR / PPR) computed per player-season so the
  UI can toggle scoring; PPR is the vector feature (fantasy's lingua franca).

Data source (free, permissive): nflverse-data on GitHub releases, one CSV per
season under
  https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_<season>.csv
Each row is a player-week regular-season box score with passing/rushing/
receiving detail plus precomputed fantasy_points / fantasy_points_ppr and
usage shares (target_share, air_yards_share, wopr).

Every season CSV is cached under pipeline/cache/, resumable -- re-running the
script skips anything already on disk. --offline rebuilds from cache only.

Run:  python pipeline/build_vectors.py [--offline]
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

import nfl_data as nfl
from nfl_data import num

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "vectors.json"

# Full nflverse history (1999) .. current calendar year; seasons without a
# published file are skipped, so this auto-includes each new season the moment
# nflverse posts it. Grades/archetypes need only box-score + fantasy columns,
# which exist for every season back to 1999.
SEASONS = list(range(1999, date.today().year + 1))
SKILL_POS = {"QB", "RB", "WR", "TE"}
MIN_GAMES = 6            # season-level qualification (a real fantasy sample)

# ---------------------------------------------------------------------------
# Feature contract (18 dims, per-game unless noted). Order is the contract the
# game/UI indexes -- mirrors GAME_FEATURES in vector-hoops / vector-pitch.
# ---------------------------------------------------------------------------

FEATURES = [
    "PASS_ATT_PG", "PASS_YDS_PG", "PASS_TD_PG", "INT_PG",
    "CARRY_PG", "RUSH_YDS_PG", "RUSH_TD_PG",
    "TGT_PG", "REC_PG", "REC_YDS_PG", "REC_TD_PG",
    "AIR_YDS_PG", "YAC_PG", "TGT_SHARE",
    "TOUCH_PG", "TOTAL_TD_PG", "EPA_PG", "FPTS_PPR_PG",
]
LABELS = {
    "PASS_ATT_PG": "pass volume",
    "PASS_YDS_PG": "passing yards",
    "PASS_TD_PG": "passing touchdowns",
    "INT_PG": "interceptions thrown",
    "CARRY_PG": "rush volume",
    "RUSH_YDS_PG": "rushing yards",
    "RUSH_TD_PG": "rushing touchdowns",
    "TGT_PG": "targets",
    "REC_PG": "receptions",
    "REC_YDS_PG": "receiving yards",
    "REC_TD_PG": "receiving touchdowns",
    "AIR_YDS_PG": "downfield target depth",
    "YAC_PG": "yards after catch",
    "TGT_SHARE": "team target share",
    "TOUCH_PG": "touches (carries+catches)",
    "TOTAL_TD_PG": "total touchdowns",
    "EPA_PG": "expected points added",
    "FPTS_PPR_PG": "fantasy points (PPR)",
}


# ---------------------------------------------------------------------------
# Per-season aggregation: sum weekly boxes into a player-season, /games-played
# ---------------------------------------------------------------------------

class PlayerAgg:
    __slots__ = (
        "name", "pos", "team", "headshot", "games",
        "pass_att", "pass_yds", "pass_td", "ints",
        "carries", "rush_yds", "rush_td",
        "tgt", "rec", "rec_yds", "rec_td", "air_yds", "yac",
        "tgt_share_sum", "epa", "total_td",
        "fpts_std", "fpts_ppr",
    )

    def __init__(self, name, pos, team, headshot):
        self.name = name
        self.pos = pos
        self.team = team
        self.headshot = headshot
        self.games = 0
        self.pass_att = self.pass_yds = self.pass_td = self.ints = 0.0
        self.carries = self.rush_yds = self.rush_td = 0.0
        self.tgt = self.rec = self.rec_yds = self.rec_td = 0.0
        self.air_yds = self.yac = self.tgt_share_sum = 0.0
        self.epa = self.total_td = 0.0
        self.fpts_std = self.fpts_ppr = 0.0


def aggregate_season(rows: list[dict]) -> dict[str, PlayerAgg]:
    agg: dict[str, PlayerAgg] = {}
    for r in rows:
        if r.get("season_type") != "REG":
            continue
        pos = (r.get("position") or "").strip()
        if pos not in SKILL_POS:
            continue
        pid = r.get("player_id") or r.get("player_display_name")
        a = agg.get(pid)
        if a is None:
            a = PlayerAgg(
                r.get("player_display_name") or r.get("player_name") or pid,
                pos, r.get("team", ""), r.get("headshot_url", ""))
            agg[pid] = a
        # A row exists only for weeks the player recorded stats -> games played.
        a.games += 1
        a.pass_att += num(r, "attempts")
        a.pass_yds += num(r, "passing_yards")
        a.pass_td += num(r, "passing_tds")
        a.ints += num(r, "passing_interceptions")
        a.carries += num(r, "carries")
        a.rush_yds += num(r, "rushing_yards")
        a.rush_td += num(r, "rushing_tds")
        a.tgt += num(r, "targets")
        a.rec += num(r, "receptions")
        a.rec_yds += num(r, "receiving_yards")
        a.rec_td += num(r, "receiving_tds")
        a.air_yds += num(r, "receiving_air_yards")
        a.yac += num(r, "receiving_yards_after_catch")
        a.tgt_share_sum += num(r, "target_share")
        a.epa += (num(r, "passing_epa") + num(r, "rushing_epa")
                  + num(r, "receiving_epa"))
        a.total_td += (num(r, "passing_tds") + num(r, "rushing_tds")
                       + num(r, "receiving_tds"))
        a.fpts_std += num(r, "fantasy_points")
        a.fpts_ppr += num(r, "fantasy_points_ppr")
        a.team = r.get("team") or a.team  # latest team seen
    return agg


def feature_row(a: PlayerAgg) -> dict:
    g = a.games
    return {
        "PASS_ATT_PG": a.pass_att / g,
        "PASS_YDS_PG": a.pass_yds / g,
        "PASS_TD_PG": a.pass_td / g,
        "INT_PG": a.ints / g,
        "CARRY_PG": a.carries / g,
        "RUSH_YDS_PG": a.rush_yds / g,
        "RUSH_TD_PG": a.rush_td / g,
        "TGT_PG": a.tgt / g,
        "REC_PG": a.rec / g,
        "REC_YDS_PG": a.rec_yds / g,
        "REC_TD_PG": a.rec_td / g,
        "AIR_YDS_PG": a.air_yds / g,
        "YAC_PG": a.yac / g,
        "TGT_SHARE": a.tgt_share_sum / g,   # mean weekly team target share
        "TOUCH_PG": (a.carries + a.rec) / g,
        "TOTAL_TD_PG": a.total_td / g,
        "EPA_PG": a.epa / g,
        "FPTS_PPR_PG": a.fpts_ppr / g,
    }


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def main() -> int:
    t_start = time.time()
    offline = "--offline" in sys.argv

    rows: list[dict] = []
    seasons_used: list[int] = []
    for season in SEASONS:
        data = nfl.weekly_stats(season, offline)
        if not data:
            print(f"WARNING: no data for {season} -- skipping")
            continue
        agg = aggregate_season(data)
        kept = 0
        for a in agg.values():
            if a.games < MIN_GAMES:
                continue
            feats = feature_row(a)
            rows.append({
                "name": a.name, "season": season, "pos": a.pos, "team": a.team,
                "games": a.games, "headshot": a.headshot,
                "fpts_std_pg": a.fpts_std / a.games,
                "fpts_half_pg": (a.fpts_std + a.fpts_ppr) / 2 / a.games,
                "fpts_ppr_pg": a.fpts_ppr / a.games,
                **feats,
            })
            kept += 1
        seasons_used.append(season)
        print(f"  {season}: {kept} qualified skill players "
              f"(>= {MIN_GAMES} games)")

    if not rows:
        raise SystemExit("no player-seasons aggregated -- aborting honestly "
                         "(network wall or empty cache; the cache is per-season "
                         "and resumable, re-run without --offline)")

    n, d = len(rows), len(FEATURES)
    X = np.array([[r[f] for f in FEATURES] for r in rows], dtype=np.float64)

    # ---- season z-scores (context-honest: each season normalized separately) --
    season_idx: dict[int, list[int]] = {}
    for i, r in enumerate(rows):
        season_idx.setdefault(r["season"], []).append(i)
    Z = np.zeros_like(X)
    for idxs in season_idx.values():
        block = X[idxs]
        mu = block.mean(axis=0)
        sd = block.std(axis=0)
        sd[sd == 0] = 1.0
        Z[idxs] = (block - mu) / sd
    Z = np.clip(Z, -4, 4)

    # ---- PCA(3) map (SVD on the centered z-matrix) ----
    C = Z - Z.mean(0)
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    P = U[:, :3] * S[:3]
    P = (P - P.min(0)) / (P.max(0) - P.min(0)).max()

    # ---- k-means(8) archetypes (numpy, seeded, mirrors the siblings) ----
    K = 8
    rng = np.random.default_rng(7)
    cent = Z[rng.choice(n, K, replace=False)].copy()
    lab = np.zeros(n, dtype=int)
    for _ in range(80):
        dist = ((Z[:, None, :] - cent[None]) ** 2).sum(-1)
        lab = dist.argmin(1)
        for k in range(K):
            if (lab == k).any():
                cent[k] = Z[lab == k].mean(0)

    def name_cluster(c: np.ndarray) -> str:
        top = np.argsort(-c)[:2]
        low = int(np.argsort(c)[0])
        a, b = LABELS[FEATURES[top[0]]], LABELS[FEATURES[top[1]]]
        if c[top[1]] > 0.35:
            return f"{a} + {b}".title()
        return f"{a} (low {LABELS[FEATURES[low]]})".title()

    cluster_names = [name_cluster(cent[k]) for k in range(K)]

    # dominant position per cluster (for UI grouping / sanity)
    cluster_pos: list[str] = []
    for k in range(K):
        poss = [rows[i]["pos"] for i in range(n) if lab[i] == k]
        cluster_pos.append(max(set(poss), key=poss.count) if poss else "?")

    players = []
    for i, r in enumerate(rows):
        players.append({
            "id": i,
            "name": r["name"], "season": r["season"], "team": r["team"],
            "pos": r["pos"], "games": r["games"], "headshot": r["headshot"],
            "ppg": {
                "std": round(r["fpts_std_pg"], 2),
                "half": round(r["fpts_half_pg"], 2),
                "ppr": round(r["fpts_ppr_pg"], 2),
            },
            "v": [round(float(z), 3) for z in Z[i]],
            "raw": [round(float(x), 2) for x in X[i]],
            "x": round(float(P[i, 0]), 4), "y": round(float(P[i, 1]), 4),
            "z": round(float(P[i, 2]), 4),
            "c": int(lab[i]),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "built": time.strftime("%Y-%m-%d"),
        "seasons": seasons_used,
        "normalization": "per-game rates, z-scored within season across QB/RB/WR/TE (context-honest)",
        "features": FEATURES, "featureLabels": LABELS,
        "clusters": cluster_names, "clusterPos": cluster_pos,
        "count": len(players),
        "players": players,
        "attribution": "Data: nflverse (nflverse.com) player_stats -- free, permissive; built with numpy, no third-party data libs",
    }, separators=(",", ":")), encoding="utf-8")

    # ---- audit assertions: never ship a dirty file ----
    assert all(len(p["v"]) == d for p in players), "vector length"
    assert all(all(-4.0001 <= v <= 4.0001 for v in p["v"]) for p in players), "clip"
    assert all(0 <= p["x"] <= 1 and 0 <= p["y"] <= 1 and 0 <= p["z"] <= 1
               for p in players), "map range"
    assert all(p["pos"] in SKILL_POS for p in players), "skill positions only"

    var_explained = (S[:3] ** 2).sum() / (S ** 2).sum()
    elapsed = time.time() - t_start
    print(f"\nwrote {OUT.name}: {len(players)} player-seasons across "
          f"{len(seasons_used)} seasons, {K} archetypes, {d} features "
          f"(PCA3 explains {var_explained:.0%} of variance, {elapsed:.0f}s)")
    for k, nm in enumerate(cluster_names):
        print(f"  cluster {k}: {nm} [{cluster_pos[k]}] "
              f"({int((lab == k).sum())} players)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
