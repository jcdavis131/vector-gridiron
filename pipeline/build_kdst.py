"""Kicker (K) and Team Defense/Special-Teams (DST) projections.

The skill-position MTNN deliberately ignores K/DST (different scoring, different
signal). But they're real lineup slots, so we project them here with simple,
honest, data-grounded models and ship them alongside the skill board:

  Kickers  -- nflverse zeroes fantasy_points for kickers, so we score them from
              the FG-by-distance + PAT columns (3/4/5 pts by bucket, 1 per PAT).
              Season ppg per kicker; project next season = recency-weighted
              average of the last two seasons.
  DST      -- aggregate each team-week's def_sacks / interceptions / fumble
              recoveries / def+ST TDs / safeties from the player rows, add the
              points-allowed bucket from games.csv, and average to a per-game
              DST score. Project next season = recency-weighted last two seasons.

Every K/DST also carries a per-season history so the Lookback / what-if replays
can draft them for prior seasons too. Output: assets/kdst.json.

  python pipeline/build_kdst.py
"""

from __future__ import annotations

import json
import re
import time
from datetime import date
from pathlib import Path

import nfl_data as nfl
from nfl_data import num

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

FG_BUCKETS = [("fg_made_0_19", 3), ("fg_made_20_29", 3), ("fg_made_30_39", 3),
              ("fg_made_40_49", 4), ("fg_made_50_59", 5), ("fg_made_60_", 5)]


def norm_key(name, pos):
    s = (name or "").lower()
    s = re.sub(r"[.'`]", "", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s)
    s = re.sub(r"[^a-z ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return f"{s}|{pos}"


def pa_bucket(pa):
    if pa <= 0: return 10.0
    if pa <= 6: return 7.0
    if pa <= 13: return 4.0
    if pa <= 20: return 1.0
    if pa <= 27: return 0.0
    if pa <= 34: return -1.0
    return -4.0


def kicker_points(r):
    pts = num(r, "pat_made")
    for col, v in FG_BUCKETS:
        pts += v * num(r, col)
    return pts


def points_allowed_table(offline):
    """(season, week, team) -> points allowed that game."""
    pa = {}
    for g in nfl.games(offline):
        s, w = int(num(g, "season")), int(num(g, "week"))
        hs, as_ = num(g, "home_score", None), num(g, "away_score", None)
        home, away = g.get("home_team", ""), g.get("away_team", "")
        if g.get("home_score") in ("", "NA", None):
            continue
        pa[(s, w, home)] = as_        # home defense allowed the away score
        pa[(s, w, away)] = hs
    return pa


def build():
    offline = False
    last = nfl.latest_stats_season()
    seasons = [y for y in range(nfl.FIRST_SEASON, last + 1) if nfl.weekly_stats(y)]
    pa = points_allowed_table(offline)

    # current team for kickers (offseason moves)
    kteam = {}
    for r in nfl.players():
        if (r.get("position") or "") == "K":
            kteam[r.get("gsis_id")] = (r.get("latest_team") or "").strip()

    k_hist, k_name = {}, {}          # gsis -> {season: ppg}
    dst_hist = {}                    # team -> {season: ppg}
    for s in seasons:
        rows = nfl.weekly_stats(s)
        kagg, dweek = {}, {}
        for r in rows:
            if r.get("season_type") != "REG":
                continue
            pos = (r.get("position") or "").strip()
            team = r.get("team", "")
            wk = int(num(r, "week"))
            if pos == "K":
                g = r.get("player_id")
                a = kagg.setdefault(g, {"g": 0, "pts": 0.0, "name": r.get("player_display_name", "")})
                a["g"] += 1; a["pts"] += kicker_points(r)
            # accumulate team defense box from every player row
            d = dweek.setdefault((wk, team), {"sacks": 0.0, "int": 0.0, "fr": 0.0, "td": 0.0, "sfty": 0.0})
            d["sacks"] += num(r, "def_sacks")
            d["int"] += num(r, "def_interceptions")
            d["fr"] += num(r, "fumble_recovery_opp")
            d["td"] += num(r, "def_tds") + num(r, "special_teams_tds")
            d["sfty"] += num(r, "def_safeties")
        for g, a in kagg.items():
            if a["g"] >= 4:
                k_hist.setdefault(g, {})[s] = round(a["pts"] / a["g"], 2)
                k_name[g] = a["name"]
        # DST per team-season
        teamweek = {}
        for (wk, team), d in dweek.items():
            allowed = pa.get((s, wk, team))
            if allowed is None:
                continue
            pts = d["sacks"] + 2 * d["int"] + 2 * d["fr"] + 6 * d["td"] + 2 * d["sfty"] + pa_bucket(allowed)
            teamweek.setdefault(team, []).append(pts)
        for team, arr in teamweek.items():
            if team and len(arr) >= 4:
                dst_hist.setdefault(team, {})[s] = round(sum(arr) / len(arr), 2)

    def project(hist):
        """recency-weighted mean of the last two seasons present."""
        ys = sorted(hist)
        if not ys:
            return None
        if len(ys) == 1:
            return hist[ys[-1]]
        a, b = hist[ys[-1]], hist[ys[-2]]
        return round(0.65 * a + 0.35 * b, 2)

    kickers = []
    for g, hist in k_hist.items():
        proj = project(hist)
        if proj is None:
            continue
        kickers.append({"key": norm_key(k_name[g], "K"), "name": k_name[g], "pos": "K",
                        "team": kteam.get(g, ""), "proj": proj, "history": hist})
    kickers.sort(key=lambda x: -x["proj"])
    for i, k in enumerate(kickers):
        k["rank_pos"] = i + 1

    dst = []
    for team, hist in dst_hist.items():
        proj = project(hist)
        if proj is None:
            continue
        dst.append({"key": f"{team.lower()}|DST", "name": f"{team} DST", "pos": "DST",
                    "team": team, "proj": proj, "history": hist})
    dst.sort(key=lambda x: -x["proj"])
    for i, d in enumerate(dst):
        d["rank_pos"] = i + 1

    ASSETS.mkdir(exist_ok=True)
    (ASSETS / "kdst.json").write_text(json.dumps({
        "built": time.strftime("%Y-%m-%d"), "proj_season": last + 1,
        "seasons": seasons, "kickers": kickers, "dst": dst,
    }, separators=(",", ":")), encoding="utf-8")
    print(f"wrote kdst.json: {len(kickers)} kickers, {len(dst)} DST, "
          f"seasons {seasons[0]}-{seasons[-1]}, projecting {last + 1}")
    print("  top K:", ", ".join(f"{k['name']} {k['proj']}" for k in kickers[:3]))
    print("  top DST:", ", ".join(f"{d['name']} {d['proj']}" for d in dst[:3]))


if __name__ == "__main__":
    build()
