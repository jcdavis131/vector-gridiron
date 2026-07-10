"""Opportunity layer for MTNN v2 — ffopportunity EP + derived usage shares.

Cites: ffopportunity / ffverse (CC-BY-SA 4.0). Downloads ep_weekly_<year>.csv via
nfl_data.ep_weekly and indexes by (gsis_id, season, week).

Also derives simple team-share helpers used by the opportunity tower:
  ep_fpts, ep_diff (EPOE), rec_attempt, rush_attempt, td_exp.

Run:  python pipeline/build_opportunity.py [--offline]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import nfl_data as nfl
from nfl_data import num

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "pipeline" / "data"
SKILL = {"QB", "RB", "WR", "TE"}


def load_ep_index(seasons: list[int], offline: bool = False) -> dict:
    """(gsis, season, week) -> opportunity feature dict."""
    idx = {}
    for y in seasons:
        rows = nfl.ep_weekly(y, offline)
        if not rows:
            print(f"  ep_weekly {y}: missing")
            continue
        n = 0
        for r in rows:
            pos = (r.get("position") or "").strip()
            if pos not in SKILL:
                continue
            gsis = (r.get("player_id") or "").strip()
            if not gsis:
                continue
            week = int(num(r, "week"))
            key = (gsis, y, week)
            idx[key] = {
                "ep_fpts": num(r, "total_fantasy_points_exp"),
                "ep_diff": num(r, "total_fantasy_points_diff"),
                "ep_actual": num(r, "total_fantasy_points"),
                "rec_attempt": num(r, "rec_attempt"),
                "rush_attempt": num(r, "rush_attempt"),
                "pass_attempt": num(r, "pass_attempt"),
                "td_exp": num(r, "total_touchdown_exp"),
                "rec_air_yards": num(r, "rec_air_yards"),
                "pass_air_yards": num(r, "pass_air_yards"),
            }
            n += 1
        print(f"  ep_weekly {y}: {n} skill rows")
    return idx


def trailing_mean(hist: list[dict], key: str, default: float = 0.0) -> float:
    if not hist:
        return default
    vals = [h.get(key, default) for h in hist]
    return sum(vals) / len(vals)


def build(last_season: int | None = None, offline: bool = False) -> dict:
    last = last_season or nfl.latest_stats_season(offline)
    seasons = list(range(nfl.FIRST_SEASON, last + 1))
    print(f"building opportunity index {seasons[0]}-{seasons[-1]} ...")
    idx = load_ep_index(seasons, offline)
    DATA.mkdir(parents=True, exist_ok=True)
    # compact JSON for debugging (not the train matrix)
    sample = []
    for i, ((g, s, w), v) in enumerate(idx.items()):
        if i >= 20:
            break
        sample.append({"gsis": g, "season": s, "week": w, **v})
    meta = {
        "built": time.strftime("%Y-%m-%d"),
        "seasons": seasons,
        "n_rows": len(idx),
        "source": "ffopportunity latest-data ep_weekly (CC-BY-SA 4.0)",
        "sample": sample,
    }
    (DATA / "opportunity_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  indexed {len(idx)} player-weeks -> pipeline/data/opportunity_meta.json")
    return {"index": idx, "meta": meta, "seasons": seasons}


def main() -> int:
    offline = "--offline" in sys.argv
    d = build(offline=offline)
    assert d["meta"]["n_rows"] > 10000, "opportunity index too small"
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
