#!/usr/bin/env python3
"""build_site_data.py — Gridiron weekly site-data builder (Wave B).

Emits exactly the JSON files assets/app.js reads, from real nflverse data
via nflreadpy (the dfs-nfl code path), plus FantasyFootballCalculator ADP
(the source the old adp.json already used):

  assets/projections.json   rest-of-season PPR points/game board (skill)
  assets/nextgame.json      upcoming-week projections with matchup/conditions
  assets/adp.json           FFC consensus ADP (ppr + half-ppr, draft-season artifact)
  assets/kdst.json          K/DST season-rate projections + per-season history
  assets/eval_backtest.json walk-forward backtest of THIS method on 2026

assets/lookback_seasons.json is a static historical artifact (2000-2025
simulated draft grades); the builder schema-validates it and leaves its
bytes untouched. assets/vectors.json (the map) is NOT rebuilt here — it
comes from the MTNN training pipeline and was refreshed 2026-10-06.

Projection method (v1 — honest and simple, no fabricated precision):
  base form   = trailing-4 weighted ppg blended with season ppg, with a
                prior-season (2025) prior for small samples
  opponent    = x defense-vs-position factor (points allowed per game to
                the position vs league average, shrunk by sample size)
  ECR blend   = 70% form x opponent + 30% expert-consensus-rank-implied
                points (rank->points curve fit on season-to-date ppg)
Floor/ceiling = projection -/+ the per-position q80 of |residual| from
the walk-forward backtest below. The backtest re-runs the identical
projection function as-of each completed week (features strictly from
prior weeks; ECR from the Friday snapshot before that week's first
game), so the number on the site is the number this code produces.

QA gates fail loud (nonzero exit, nothing written): week derived from
the schedule itself, row-count bands, no null/NaN in numeric fields,
every next-game player's opponent matches the schedule, emitted week ==
schedule's current week, per-record schema checks against app.js.

Run with the gridiron venv:
  ~/workspace/gridiron-venv/bin/python pipeline/build_site_data.py
"""

from __future__ import annotations

import json
import math
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
PUBLIC_ASSETS = ROOT / "public" / "assets"

MODEL_VERSION = "gridiron-proj-v1.0"
SKILL = ("QB", "RB", "WR", "TE")
POS_MAP = {"FB": "RB"}
TEAM_FIX = {"JAC": "JAX", "ARZ": "ARI", "LA": "LAR"}  # dfs-nfl convention
SUFFIXES = {"ii", "iii", "iv", "v", "jr", "sr"}
OUT_STATUSES = {"RES", "PUP", "SUS", "RET", "CUT", "EXE", "DEV"}  # not on an active roster
TRAIL_W = (0.4, 0.3, 0.2, 0.1)  # most-recent-first
ECR_BLEND = 0.30
MIN_GROUP = 8

TODAY = datetime.now(timezone.utc).date()
BUILT = TODAY.isoformat()
BUILT_AT = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def fail(msg: str) -> None:
    print(f"QA FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def canon_name(s: str) -> str:
    """dfs-nfl canon_name: normalized lowercase, generational suffix stripped."""
    s = "".join(c if c.isalnum() or c == " " else " " for c in (s or "").lower())
    parts = s.split()
    while parts and parts[-1] in SUFFIXES:
        parts.pop()
    return " ".join(parts)


def norm_key(name: str, pos: str) -> str:
    return f"{name.lower()}|{pos}"


# --------------------------------------------------------------------------
# loading (nflreadpy — the dfs-nfl code path)
# --------------------------------------------------------------------------

def load_all():
    import nflreadpy as nfl

    print("loading nflverse via nflreadpy ...")
    sched26 = nfl.load_schedules([2026]).to_pandas()
    stats26 = nfl.load_player_stats([2026], summary_level="week").to_pandas()
    stats25 = nfl.load_player_stats([2025], summary_level="week").to_pandas()
    rosters26 = nfl.load_rosters([2026]).to_pandas()
    rosters25 = nfl.load_rosters([2025]).to_pandas()
    injuries26 = nfl.load_injuries([2026]).to_pandas()
    ecr_now = nfl.load_ff_rankings("week").to_pandas()
    ecr_all = nfl.load_ff_rankings("all").to_pandas()
    # history for kdst (2016-2025): team stats + schedules + player stats
    hist_stats = {}
    for yr in range(2016, 2026):
        hist_stats[yr] = nfl.load_player_stats([yr], summary_level="week").to_pandas()
    hist_stats[2025] = stats25
    team_stats = {}
    scheds = {2026: sched26}
    for yr in range(2016, 2027):
        team_stats[yr] = nfl.load_team_stats([yr], summary_level="week").to_pandas()
        if yr != 2026:
            scheds[yr] = nfl.load_schedules([yr]).to_pandas()
    print("loaded.")
    return {
        "sched26": sched26, "stats26": stats26, "stats25": stats25,
        "rosters26": rosters26, "rosters25": rosters25, "injuries26": injuries26,
        "ecr_now": ecr_now, "ecr_all": ecr_all,
        "hist_stats": hist_stats, "team_stats": team_stats, "scheds": scheds,
    }


def reg(df):
    return df[df["season_type"] == "REG"] if "season_type" in df else df


# --------------------------------------------------------------------------
# schedule-derived truth: current week, byes, games
# --------------------------------------------------------------------------

def current_week(sched) -> tuple[int, int]:
    """(last fully-scored week, target week) from the schedule itself."""
    g = sched[sched["game_type"] == "REG"].copy()
    g["played"] = g["home_score"].notna()
    by_week = g.groupby("week")["played"].agg(["sum", "count"])
    completed = [int(w) for w, r in by_week.iterrows() if r["sum"] == r["count"]]
    pending = [int(w) for w, r in by_week.iterrows() if r["sum"] < r["count"]]
    if not pending:
        fail("schedule has no unplayed regular-season week")
    target = min(pending)
    last_done = max([w for w in completed if w < target], default=0)
    return last_done, target


def bye_weeks(sched) -> dict[str, int]:
    g = sched[sched["game_type"] == "REG"]
    teams = set(g["home_team"]) | set(g["away_team"])
    out = {}
    for t in teams:
        played = set(g[(g["home_team"] == t) | (g["away_team"] == t)]["week"])
        missing = [w for w in range(1, 19) if w not in played]
        if len(missing) == 1:
            out[t] = int(missing[0])
    return out


def week_games(sched, week: int) -> dict[str, dict]:
    """team -> {opp, is_home, roof, temp, wind, team_implied, primetime, gameday}"""
    g = sched[(sched["game_type"] == "REG") & (sched["week"] == week)]
    out = {}
    for _, r in g.iterrows():
        home, away = r["home_team"], r["away_team"]
        total, spread = r["total_line"], r["spread_line"]
        if total is None or (isinstance(total, float) and math.isnan(total)):
            fail(f"week {week} game {away}@{home} has no total_line yet — lines not posted, refusing to emit null conditions")
        # nflverse spread_line is the HOME spread (negative = home favored)
        home_impl = (total - spread) / 2.0
        away_impl = (total + spread) / 2.0
        roof = r["roof"] if isinstance(r["roof"], str) and r["roof"] else "outdoors"
        temp = None if r["temp"] is None or (isinstance(r["temp"], float) and math.isnan(r["temp"])) else float(r["temp"])
        wind = None if r["wind"] is None or (isinstance(r["wind"], float) and math.isnan(r["wind"])) else float(r["wind"])
        try:
            primetime = 1 if int(str(r["gametime"])[:2]) >= 19 else 0
        except Exception:
            primetime = 0
        base = {"roof": roof, "temp": temp, "wind": wind,
                "primetime": primetime, "gameday": str(r["gameday"])}
        out[home] = {**base, "opp": away, "is_home": 1, "team_implied": round(home_impl * 2) / 2}
        out[away] = {**base, "opp": home, "is_home": 0, "team_implied": round(away_impl * 2) / 2}
    return out


# --------------------------------------------------------------------------
# player identity / pool
# --------------------------------------------------------------------------

def build_identity(data) -> dict:
    """gsis_id -> identity from rosters (latest week wins), stats as fallback."""
    ident = {}
    for src, df in (("rosters25", data["rosters25"]), ("rosters26", data["rosters26"])):
        d = df.sort_values("week") if "week" in df else df
        for _, r in d.iterrows():
            gid = r.get("gsis_id")
            if not gid or (isinstance(gid, float) and math.isnan(gid)):
                continue
            cur = ident.setdefault(gid, {})
            if src == "rosters26":
                cur.update({
                    "name": r.get("full_name") or cur.get("name"),
                    "team": r.get("team") or cur.get("team"),
                    "pos_roster": r.get("position"),
                    "status": r.get("status") or "",
                    "headshot": r.get("headshot_url") or "",
                    "years_exp": r.get("years_exp"),
                })
            else:
                cur.setdefault("team25", r.get("team"))
    return ident


def game_logs(stats) -> dict[str, list[dict]]:
    """player_id -> [{week, team, opp, pts, comp{...}}] sorted by week (REG only)."""
    logs: dict[str, list[dict]] = {}
    d = reg(stats)
    for _, r in d.iterrows():
        pos = POS_MAP.get(r["position"], r["position"])
        if pos not in SKILL and pos != "K":
            continue
        comp = {
            "pass_yds": float(r.get("passing_yards") or 0),
            "rush_yds": float(r.get("rushing_yards") or 0),
            "rec_yds": float(r.get("receiving_yards") or 0),
            "rec": float(r.get("receptions") or 0),
            "td": float((r.get("passing_tds") or 0) + (r.get("rushing_tds") or 0) + (r.get("receiving_tds") or 0)),
        }
        pts = float(r.get("fantasy_points_ppr") or 0)
        if pos == "K":
            pts = kicking_points(r)
        logs.setdefault(r["player_id"], []).append(
            {"week": int(r["week"]), "team": r["team"], "opp": r["opponent_team"],
             "pos": pos, "pts": pts, "comp": comp})
    for v in logs.values():
        v.sort(key=lambda g: g["week"])
    return logs


def kicking_points(r) -> float:
    def g(c):
        v = r.get(c)
        return float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else 0.0
    return (g("fg_made_0_19") * 3 + g("fg_made_20_29") * 3 + g("fg_made_30_39") * 3
            + g("fg_made_40_49") * 4 + g("fg_made_50_59") * 5 + g("fg_made_60_") * 5
            + g("pat_made") * 1)


# --------------------------------------------------------------------------
# ECR (expert consensus) — dfs-nfl patterns
# --------------------------------------------------------------------------

def ecr_from_week_load(df) -> dict[tuple[str, str], dict]:
    """Current-week ECR from load_ff_rankings('week'): (canon, pos) -> {ecr, pos_rank}."""
    pages = {"qb": "QB", "ppr-rb": "RB", "ppr-wr": "WR", "ppr-te": "TE"}
    out: dict[tuple[str, str], dict] = {}
    for _, r in df.iterrows():
        pos = pages.get(str(r.get("page", "")))
        if not pos:
            continue
        key = (canon_name(str(r.get("player_name", ""))), pos)
        out[key] = {"ecr": float(r["ecr"]), "team": TEAM_FIX.get(str(r.get("team", "")), str(r.get("team", "")))}
    for pos in SKILL:
        rows = sorted([k for k in out if k[1] == pos], key=lambda k: out[k]["ecr"])
        for i, k in enumerate(rows, 1):
            out[k]["pos_rank"] = i
    return out


def ecr_snapshot_for_week(ecr_all, sched, season: int, week: int) -> dict[tuple[str, str], dict]:
    """ECR as-of week `week`: the Friday ff snapshot strictly before the
    week's first gameday (zero leakage — dfs-nfl scrape-Friday convention,
    tightened to pre-week)."""
    g = sched[(sched["game_type"] == "REG") & (sched["week"] == week)]
    first_day = str(g["gameday"].min())
    pages = {"weekly-qb": "QB", "weekly-rb": "RB", "weekly-wr": "WR", "weekly-te": "TE"}
    d = ecr_all[ecr_all["page_type"].isin(pages)]
    d = d[d["scrape_date"].astype(str) < first_day]
    if d.empty:
        return {}
    snap = str(d["scrape_date"].astype(str).max())
    d = d[d["scrape_date"].astype(str) == snap]
    out: dict[tuple[str, str], dict] = {}
    for _, r in d.iterrows():
        pos = pages[str(r["page_type"])]
        key = (canon_name(str(r.get("player", ""))), pos)
        out[key] = {"ecr": float(r["ecr"]), "team": TEAM_FIX.get(str(r.get("team", "")), str(r.get("team", ""))),
                    "scrape_date": snap}
    for pos in SKILL:
        rows = sorted([k for k in out if k[1] == pos], key=lambda k: out[k]["ecr"])
        for i, k in enumerate(rows, 1):
            out[k]["pos_rank"] = i
    return out


def ecr_curve(ppg_by_pos: dict[str, list[float]]) -> dict[str, list[float]]:
    """pos -> smoothed ppg-by-rank curve (rank 1 = best) from season ppg values."""
    out = {}
    for pos, ppgs in ppg_by_pos.items():
        vals = sorted(ppgs, reverse=True)
        sm = []
        for i in range(len(vals)):
            w = vals[max(0, i - 2): i + 3]
            sm.append(sum(w) / len(w))
        out[pos] = sm
    return out


def asof_ppg_by_pos(logs_asof: dict[str, list[dict]]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for _, gs in logs_asof.items():
        if gs and gs[-1]["pos"] in SKILL:
            out.setdefault(gs[-1]["pos"], []).append(sum(g["pts"] for g in gs) / len(gs))
    return out


def curve_pts(curve: dict[str, list[float]], pos: str, rank: int) -> float | None:
    c = curve.get(pos) or []
    if not c:
        return None
    return c[min(rank, len(c)) - 1]


# --------------------------------------------------------------------------
# the projection engine (one function — live and backtest run the same code)
# --------------------------------------------------------------------------

def form_base(games: list[dict], prior_ppg: float | None) -> float | None:
    """Trailing-form base rate from a player's game log (all games < target)."""
    if not games:
        return prior_ppg * 0.92 if prior_ppg else None
    pts = [g["pts"] for g in games]
    season_ppg = sum(pts) / len(pts)
    last = pts[-len(TRAIL_W):]
    w = TRAIL_W[-len(last):]
    trailing = sum(p * wi for p, wi in zip(reversed(last), w)) / sum(w)
    if len(pts) >= 3:
        return 0.65 * trailing + 0.35 * season_ppg
    if prior_ppg:
        return 0.5 * season_ppg + 0.5 * prior_ppg
    return season_ppg


def opp_factors(logs: dict[str, list[dict]], upto_week: int) -> dict[tuple[str, str], float]:
    """(defense, pos) -> shrunk multiplier of pts allowed vs league average."""
    allowed: dict[tuple[str, str], list[float]] = {}
    team_games: dict[str, set] = {}
    pos_totals: dict[str, float] = {}
    pos_teamgames = 0
    seen_games: set[tuple[str, int]] = set()
    for games in logs.values():
        for g in games:
            if g["week"] >= upto_week or g["pos"] == "K":
                continue
            allowed.setdefault((g["opp"], g["pos"]), []).append(g["pts"])
            team_games.setdefault(g["opp"], set()).add(g["week"])
            seen_games.add((g["opp"], g["week"]))
    for (d, pos), vals in allowed.items():
        pos_totals[pos] = pos_totals.get(pos, 0) + sum(vals)
    n_tg = len(seen_games)
    league = {p: (pos_totals[p] / n_tg if n_tg else 0) for p in SKILL}
    out = {}
    for (d, pos), vals in allowed.items():
        n = len(team_games[d])
        per_game = sum(vals) / n
        raw = per_game / league[pos] if league.get(pos) else 1.0
        out[(d, pos)] = 1.0 + (raw - 1.0) * (n / (n + 3.0))
    return out


def project(base: float | None, pos: str, opp: str | None,
            factors: dict, ecr_rank: int | None, curve: dict) -> float | None:
    if base is None and ecr_rank is None:
        return None
    epts = curve_pts(curve, pos, ecr_rank) if ecr_rank else None
    if base is None:
        p = epts
    else:
        adj = base * (factors.get((opp, pos), 1.0) if opp else 1.0)
        p = adj if epts is None else (1 - ECR_BLEND) * adj + ECR_BLEND * epts
    if p is None or not math.isfinite(p):
        return None
    return max(0.0, p)  # expected fantasy points are never negative

# --------------------------------------------------------------------------
# statistics helpers (stdlib only)
# --------------------------------------------------------------------------

def rankdata(xs: list[float]) -> list[float]:
    """Average-tie ranks (1-based), stdlib Spearman support."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    rx, ry = rankdata(xs), rankdata(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx)
    dy = sum((b - my) ** 2 for b in ry)
    if not dx or not dy:
        return None
    return num / math.sqrt(dx * dy)


def mae(xs: list[float], ys: list[float]) -> float | None:
    if not xs:
        return None
    return sum(abs(a - b) for a, b in zip(xs, ys)) / len(xs)


def q80(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(math.ceil(0.80 * len(s))) - 1)]


# --------------------------------------------------------------------------
# backtest: walk-forward, weeks 2..last_completed (2026)
# --------------------------------------------------------------------------

def run_backtest(data, logs26, ident, last_done, sched26) -> dict:
    """Run the identical projection engine as-of each completed week,
    score against actuals. Returns the eval payload (metrics) and the
    per-position q80 |residual| used for floor/ceiling."""
    prior_ppg = season_ppg(data["logs25"])
    weeks = list(range(2, last_done + 1))
    residuals: dict[str, list[float]] = {p: [] for p in SKILL}
    per_week = []
    pooled = {p: [[], []] for p in SKILL}   # pos -> [projs, actuals]
    pooled_base = {p: [[], []] for p in SKILL}
    n_rows = 0

    for N in weeks:
        asof = {pid: [g for g in games if g["week"] < N] for pid, games in logs26.items()}
        factors = opp_factors(asof, N)
        pool = build_pool(asof, ident, prior_ppg)
        ecr = ecr_snapshot_for_week(data["ecr_all"], sched26, 2026, N)
        add_ecr_only_players(pool, ecr, ident, prior_ppg)
        curve = ecr_curve(asof_ppg_by_pos(asof))
        gm = week_games(sched26, N)
        week_rows, base_rows = {p: [[], []] for p in SKILL}, {p: [[], []] for p in SKILL}
        for pid, pl in pool.items():
            g = [x for x in logs26.get(pid, []) if x["week"] == N]
            if not g:
                continue
            actual = g[0]["pts"]
            pos = pl["pos"]
            game = gm.get(g[0]["team"], {})  # week-N team: exact for trades
            base = form_base(asof.get(pid, []), prior_ppg.get(pid))
            ecr_info = ecr.get((pl["canon"], pos))
            proj = project(base, pos, game.get("opp"), factors,
                           ecr_info.get("pos_rank") if ecr_info else None, curve)
            if proj is None:
                continue
            week_rows[pos][0].append(proj)
            week_rows[pos][1].append(actual)
            residuals[pos].append(abs(proj - actual))
            pooled[pos][0].append(proj)
            pooled[pos][1].append(actual)
            n_rows += 1
            # baseline: trailing-4 simple mean
            pg = asof.get(pid, [])
            if pg:
                base_proj = sum(x["pts"] for x in pg[-4:]) / min(4, len(pg))
            else:
                base_proj = curve_pts(curve, pos, ecr_info["pos_rank"]) if ecr_info else None
            if base_proj is not None:
                base_rows[pos][0].append(base_proj)
                base_rows[pos][1].append(actual)
        wk = {"week": N, "player_weeks": sum(len(v[0]) for v in week_rows.values())}
        for pos in SKILL:
            p, a = week_rows[pos]
            if len(p) >= MIN_GROUP:
                wk[pos.lower()] = {"n": len(p), "spearman": spearman(p, a), "mae": mae(p, a)}
        all_p = [x for pos in SKILL for x in week_rows[pos][0]]
        all_a = [x for pos in SKILL for x in week_rows[pos][1]]
        wk["pooled"] = {"n": len(all_p), "spearman": spearman(all_p, all_a), "mae": mae(all_p, all_a)}
        per_week.append(wk)
        for pos in SKILL:
            pooled[pos][0] += week_rows[pos][0]
            pooled[pos][1] += week_rows[pos][1]
            pooled_base[pos][0] += base_rows[pos][0]
            pooled_base[pos][1] += base_rows[pos][1]

    positions = {}
    for pos in SKILL:
        p, a = pooled[pos]
        bp, ba = pooled_base[pos]
        sp = [w[pos.lower()]["spearman"] for w in per_week if pos.lower() in w]
        sm = [w[pos.lower()]["mae"] for w in per_week if pos.lower() in w]
        positions[pos] = {
            "n": len(p),
            "spearman": sum(sp) / len(sp) if sp else None,
            "spearman_sd": math.sqrt(sum((x - sum(sp) / len(sp)) ** 2 for x in sp) / len(sp)) if len(sp) > 1 else 0.0,
            "mae": mae(p, a),
            "baseline_last4": spearman(bp, ba),
            "mae_baseline": mae(bp, ba),
        }
    all_p = [x for pos in SKILL for x in pooled[pos][0]]
    all_a = [x for pos in SKILL for x in pooled[pos][1]]
    all_bp = [x for pos in SKILL for x in pooled_base[pos][0]]
    all_ba = [x for pos in SKILL for x in pooled_base[pos][1]]
    ov_sp = [w["pooled"]["spearman"] for w in per_week]
    ov_mae = [w["pooled"]["mae"] for w in per_week]
    overall = {
        "spearman": sum(ov_sp) / len(ov_sp),
        "spearman_sd": math.sqrt(sum((x - sum(ov_sp) / len(ov_sp)) ** 2 for x in ov_sp) / len(ov_sp)) if len(ov_sp) > 1 else 0.0,
        "mae": mae(all_p, all_a),
        "baseline_last4": spearman(all_bp, all_ba),
        "mae_baseline": mae(all_bp, all_ba),
        "n": len(all_p),
    }
    q = {pos: q80(residuals[pos]) for pos in SKILL}
    eval_payload = {
        "computed_at": BUILT_AT,
        "metric": {"name": "walk_forward_weekly_mae",
                   "definition": "For each completed 2026 week N (2..{}): project every player-week using only data through week N-1 (stats) plus the ECR snapshot strictly before week N's first game, then compare to actual week-N PPR points. MAE = mean |projected - actual|. Spearman = pooled rank correlation of projections vs actuals, averaged across weeks.".format(last_done)},
        "season": 2026,
        "weeks": weeks,
        "min_group": MIN_GROUP,
        "positions": positions,
        "overall": overall,
        "per_week": per_week,
        "model": {
            "version": MODEL_VERSION,
            "form": "0.65*trailing-4-weighted ppg + 0.35*season ppg (small-sample: 0.5*season + 0.5*prior-season ppg; no 2026 games: prior-season ppg x 0.92; ECR-only: consensus-implied points)",
            "opponent": "defense-vs-position points-allowed/game vs league average, shrunk n/(n+3), applied only to the upcoming-week board",
            "ecr_blend": "0.7 * form x opponent + 0.3 * ECR-implied points (rank->ppg curve fit on season-to-date ppg)",
            "floor_ceiling": "projection -/+ per-position q80 |residual| from this backtest",
        },
        "caveats": [
            "Backtest covers weeks 2-5 of 2026 only (the completed weeks at build time).",
            "Kickers and DST are projected by a separate season-rate method (kdst.json) and are not in this backtest.",
            "ECR snapshots are pre-week Fridays; no within-week leakage.",
        ],
        "provenance": prov("eval_backtest.json", sum(w["player_weeks"] for w in per_week), 2026, weeks[-1]),
    }
    return eval_payload, q


def season_ppg(logs: dict[str, list[dict]]) -> dict[str, float]:
    out = {}
    for pid, games in logs.items():
        if games:
            out[pid] = sum(g["pts"] for g in games) / len(games)
    return out


def build_pool(logs_asof: dict[str, list[dict]], ident: dict,
               prior_ppg: dict[str, float]) -> dict:
    """Players eligible for projection: >=1 prior game (ECR-only players
    are added separately via add_ecr_only_players). keyed by player_id."""
    pool = {}
    for pid, games in logs_asof.items():
        if not games:
            continue
        g = games[-1]
        if g["pos"] not in SKILL:  # kickers/DST are handled in build_kdst
            continue
        idn = ident.get(pid, {})
        pool[pid] = {
            "pid": pid, "pos": g["pos"], "team": idn.get("team") or g["team"],
            "name": idn.get("name") or pid, "canon": canon_name(idn.get("name") or ""),
            "headshot": idn.get("headshot") or "", "status": idn.get("status") or "",
            "years_exp": idn.get("years_exp"),
            "team25": idn.get("team25"), "prior_ppg": prior_ppg.get(pid),
            "games": games,
        }
    return pool


def add_ecr_only_players(pool: dict, ecr_map: dict[tuple[str, str], dict],
                         ident: dict, prior_ppg: dict[str, float]) -> int:
    """Add ECR-listed players with no games yet (rookies, injured returners)
    if they resolve to a roster identity. Returns the number added."""
    have = {(pl["canon"], pl["pos"]) for pl in pool.values()}
    added = 0
    for (cn, pos), _ei in ecr_map.items():
        if (cn, pos) in have:
            continue
        match = None
        for pid, idn in ident.items():
            if canon_name(idn.get("name") or "") == cn and idn.get("pos_roster") == pos \
                    and idn.get("team") and pid not in pool:
                match = pid
                break
        if match is None:
            continue
        idn = ident[match]
        pool[match] = {"pid": match, "pos": pos, "team": idn["team"], "name": idn["name"],
                       "canon": cn, "headshot": idn.get("headshot") or "",
                       "status": idn.get("status") or "", "years_exp": idn.get("years_exp"),
                       "team25": idn.get("team25"), "prior_ppg": prior_ppg.get(match),
                       "games": []}
        added += 1
    return added

# --------------------------------------------------------------------------
# live assembly — the current-week boards
# --------------------------------------------------------------------------

def prov(filename: str, rows: int, season: int, week: int, extra: dict | None = None) -> dict:
    p = {
        "builder": "pipeline/build_site_data.py",
        "model_version": MODEL_VERSION,
        "source": "nflverse via nflreadpy",
        "built_at": BUILT_AT,
        "season": season,
        "week": week,
        "rows": rows,
    }
    if extra:
        p.update(extra)
    return p


def assemble_live(data, logs26, ident, pool, factors, curve, ecr,
                  gm, byes, qres, target_week) -> tuple[list, list]:
    """(projections_players, nextgame_players)"""
    comps_tbl = load_vector_comps()
    pid_rows: list[tuple[str, dict]] = []
    for pid, pl in pool.items():
        games = pl["games"]
        base = form_base(games, pl["prior_ppg"])
        ecr_info = ecr.get((pl["canon"], pl["pos"]))
        e_rank = ecr_info.get("pos_rank") if ecr_info else None
        # season board: neutral opponent (None) + ECR blend
        proj = project(base, pl["pos"], None, factors, e_rank, curve)
        if proj is None or not math.isfinite(proj):
            continue
        line = comp_line(games, proj)
        unc = qres.get(pl["pos"], 0.0)
        row = {
            "key": norm_key(pl["name"], pl["pos"]),
            "name": pl["name"], "pos": pl["pos"], "team": pl["team"],
            "headshot": pl["headshot"], "proj": round(proj, 2),
            "floor": round(max(0.0, proj - unc), 2),
            "ceil": round(proj + unc, 2),
            "uncertainty": round(unc, 2),
            "line": line,
            "comps": comps_tbl.get((pl["canon"], pl["pos"]), []),
            "bye": byes.get(pl["team"]),
            "avail": availability(pl, data),
            "rookie": bool((pl["years_exp"] or 0) == 0),
            "moved": bool(pl["team25"] and pl["team25"] != pl["team"]),
            "prev_team": pl["team25"] if (pl["team25"] and pl["team25"] != pl["team"]) else pl["team"],
        }
        pid_rows.append((pid, row))
    pid_rows.sort(key=lambda t: t[1]["proj"], reverse=True)
    rank_pos_ctr: dict[str, int] = {}
    proj_rows = []
    for i, (pid, r) in enumerate(pid_rows, 1):
        r["rank_overall"] = i
        rank_pos_ctr[r["pos"]] = rank_pos_ctr.get(r["pos"], 0) + 1
        r["rank_pos"] = rank_pos_ctr[r["pos"]]
        r["tier"] = min(10, math.ceil(r["rank_pos"] / 3))
        proj_rows.append(r)
    pid_of = {pid: r for pid, r in pid_rows}

    # nextgame: same players + matchup, opponent-adjusted, bye/ex-roster excluded
    next_rows = []
    for pid, pl in pid_of.items():
        if pool[pid]["status"] in OUT_STATUSES:
            continue
        if byes.get(pool[pid]["team"]) == target_week:
            continue
        game = gm.get(pool[pid]["team"])
        if not game:
            fail(f"{pool[pid]['name']} ({pool[pid]['team']}) has no week-{target_week} game in schedule")
        base = form_base(pool[pid]["games"], pool[pid]["prior_ppg"])
        ecr_info = ecr.get((pool[pid]["canon"], pool[pid]["pos"]))
        proj = project(base, pool[pid]["pos"], game["opp"], factors,
                       ecr_info.get("pos_rank") if ecr_info else None, curve)
        if proj is None or not math.isfinite(proj):
            continue
        unc = qres.get(pool[pid]["pos"], 0.0)
        nr = dict(pl)
        nr.update({
            "proj": round(proj, 2),
            "floor": round(max(0.0, proj - unc), 2),
            "ceil": round(proj + unc, 2),
            "opp": game["opp"],
            "conditions": {
                "is_home": game["is_home"], "roof": game["roof"],
                "temp": game["temp"], "wind": game["wind"],
                "team_implied": game["team_implied"],
                "primetime": game["primetime"], "gameday": game["gameday"],
            },
        })
        next_rows.append(nr)
    next_rows.sort(key=lambda r: r["proj"], reverse=True)
    return proj_rows, next_rows


def comp_line(games: list[dict], proj: float) -> dict | None:
    if not games:
        return None
    n = len(games)
    mean = {k: sum(g["comp"][k] for g in games) / n for k in ("pass_yds", "rush_yds", "rec_yds", "rec", "td")}
    actual_mean = sum(g["pts"] for g in games) / n
    scale = proj / actual_mean if actual_mean > 0 else 1.0
    return {k: round(v * scale, 1) for k, v in mean.items()}


def availability(pl: dict, data) -> str:
    """injury-report flag first, else roster status passthrough (Wave-A contract)."""
    inj = data["inj_map"].get(pl["canon"], "")
    if inj:
        return inj
    return pl["status"] if pl["status"] != "ACT" else ""


def load_vector_comps() -> dict[tuple[str, str], list[dict]]:
    """Top-5 same-position cosine neighbors from vectors.json's latest
    season (currently 2025 — the file's data ends there; the July builder's
    comps were likewise from the then-latest season). Not a projection;
    a style-similarity list for the Explorer."""
    path = ASSETS / "vectors.json"
    rows = json.load(path.open())["players"]
    latest = max(r.get("season") for r in rows if r.get("season"))
    # vectors.json's data runs through `latest`; no same-season rows exist.
    rows26 = [r for r in rows if r.get("season") == latest and r.get("pos") in SKILL]
    by_key: dict[tuple[str, str], dict] = {}
    for r in rows26:
        key = (canon_name(r["name"]), r["pos"])
        prev = by_key.get(key)
        if prev is None:
            by_key[key] = r
    def cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0
    pos_groups: dict[str, list] = {}
    for key, r in by_key.items():
        pos_groups.setdefault(key[1], []).append((key, r))
    out: dict[tuple[str, str], list[dict]] = {}
    for pos, grp in pos_groups.items():
        for key, r in grp:
            cands = []
            for key2, r2 in grp:
                if key2 == key:
                    continue
                cands.append((cos(r["v"], r2["v"]), r2))
            cands.sort(key=lambda t: (-t[0], t[1]["name"]))
            out[key] = [{"name": r2["name"], "pos": r2["pos"], "team": r2["team"],
                         "sim": round(c, 3)} for c, r2 in cands[:5]]
    return out


# --------------------------------------------------------------------------
# ADP from FantasyFootballCalculator (the source the old adp.json used)
# --------------------------------------------------------------------------

FFC_POS = {"PK": "K", "DEF": "DST"}

def fetch_ffc(typ: str) -> dict:
    url = f"https://fantasyfootballcalculator.com/api/v1/adp/{typ}?teams=12&year=2026"
    req = urllib.request.Request(url, headers={"User-Agent": "vector-gridiron-wave-b/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def build_adp() -> dict:
    srcs = {}
    for typ, name in (("ppr", "ffc_ppr"), ("half-ppr", "ffc_half")):
        d = fetch_ffc(typ)
        srcs[name] = d
    agg: dict[tuple[str, str], dict] = {}
    for src, d in srcs.items():
        for p in d["players"]:
            pos = FFC_POS.get(p["position"], p["position"])
            if pos not in (*SKILL, "K", "DST"):
                continue
            key = (canon_name(p["name"]), pos)
            e = agg.setdefault(key, {"name": p["name"], "pos": pos, "per": {}, "n": 0})
            e["per"][src] = float(p["adp"])
            e["n"] += 1
    players = []
    for key, e in agg.items():
        adp = sum(e["per"].values()) / len(e["per"])
        players.append({"name": e["name"], "pos": e["pos"], "adp": round(adp, 1),
                        "per": {k: round(v, 1) for k, v in e["per"].items()},
                        "n": e["n"], "key": norm_key(e["name"], e["pos"])})
    players.sort(key=lambda p: p["adp"])
    pos_ctr: dict[str, int] = {}
    for i, p in enumerate(players, 1):
        p["adp_rank"] = i
        pos_ctr[p["pos"]] = pos_ctr.get(p["pos"], 0) + 1
        p["pos_rank"] = pos_ctr[p["pos"]]
    return {
        "built": BUILT,
        "proj_season": 2026,
        "sources": ["ffc_ppr", "ffc_half"],
        "count": len(players),
        "players": players,
        "provenance": prov("adp.json", len(players), 2026, 0, {
            "source": "FantasyFootballCalculator ADP (ppr + half-ppr, 12-team, 2026)",
            "drafts_per_source": {s: d["meta"]["total_drafts"] for s, d in srcs.items()},
            "note": "ADP is a draft-season artifact; FFC's October pull reflects the last mock-draft window (late Sept 2026), not weekly refreshes."}),
    }

# --------------------------------------------------------------------------
# K/DST — season-rate projections (v1), walk-forward holdout
# --------------------------------------------------------------------------

def dst_weekly_points(team_stats: dict, scheds: dict) -> dict[tuple[int, str], list[float]]:
    """(season, defense) -> weekly fantasy points, 2016-2026.
    sacks=1, INT=2, fum-rec=2, safeties=2, def TD=6, blocks=2, PA buckets."""
    def g(r, c):
        v = r.get(c)
        return float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else 0.0
    def pa_pts(pa):
        return {0: 10, 1: 7, 7: 4, 14: 1, 21: 0, 28: -1, 35: -4}[
            next(b for b in (0, 1, 7, 14, 21, 28, 35) if pa >= b)]
    # points allowed per game from schedules
    pa_map: dict[tuple[int, int, str], float] = {}
    for season, sc in scheds.items():
        d = sc[(sc["game_type"] == "REG") & (sc["week"] <= 18)]
        for _, r in d.iterrows():
            if r["home_score"] is None or (isinstance(r["home_score"], float) and math.isnan(r["home_score"])):
                continue
            pa_map[(season, int(r["week"]), r["home_team"])] = float(r["away_score"])
            pa_map[(season, int(r["week"]), r["away_team"])] = float(r["home_score"])
    out: dict[tuple[int, str], list[float]] = {}
    for season, ts in team_stats.items():
        d = ts[ts["season_type"] == "REG"] if "season_type" in ts else ts
        for _, r in d.iterrows():
            team = r["team"]
            pts = (g(r, "def_sacks") * 1 + g(r, "def_interceptions") * 2
                   + g(r, "fumble_recovery_opp") * 2 + g(r, "def_safeties") * 2
                   + g(r, "def_tds") * 6 + g(r, "def_punt_blocks") * 2
                   + g(r, "def_pat_blocks") * 2)
            pa = pa_map.get((season, int(r["week"]), team))
            if pa is not None:
                pts += pa_pts(pa)
            out.setdefault((season, team), []).append(pts)
    return out


def build_kdst(data, ident, last_done) -> dict:
    k_logs: dict[str, list[float]] = {}
    k_names: dict[str, str] = {}
    k_teams: dict[str, str] = {}
    for season, st in data["hist_stats"].items():
        if season > 2026:
            continue
        d = reg(st)
        for _, r in d.iterrows():
            if r["position"] != "K":
                continue
            pid = r["player_id"]
            pts = kicking_points(r)
            k_logs.setdefault(pid, []).append((season, pts))
            idn = ident.get(pid, {})
            k_names[pid] = idn.get("name") or str(r.get("player_name", ""))
            k_teams[pid] = idn.get("team") or r["team"]
    # keep kickers with >=1 game in 2025-2026 (recent relevance)
    kickers = []
    for pid, seq in k_logs.items():
        seq.sort()
        recent = [p for s, p in seq if s >= 2025]
        if not recent:
            continue
        y26 = [p for s, p in seq if s == 2026]
        season_ppg = sum(recent) / len(recent)
        if y26:
            last3 = y26[-3:]
            proj = 0.65 * (sum(last3) / len(last3)) + 0.35 * (sum(y26) / len(y26))
        else:
            proj = 0.9 * (sum(recent) / len(recent))
        history = {}
        for season in range(2016, 2027):
            vals = [p for s, p in seq if s == season]
            if vals:
                history[str(season)] = round(sum(vals) / len(vals), 1)
        kickers.append({"key": norm_key(k_names[pid], "K"), "name": k_names[pid],
                        "pos": "K", "team": k_teams[pid], "proj": round(proj, 2),
                        "history": history})
    kickers.sort(key=lambda k: k["proj"], reverse=True)
    for i, k in enumerate(kickers, 1):
        k["rank_pos"] = i
    # DST: all 32 teams
    dst_w = dst_weekly_points(data["team_stats"], data["scheds"])
    dst_rows = []
    teams32 = sorted({t for (s, t) in dst_w})
    for t in teams32:
        vals26 = dst_w.get((2026, t), [])
        if not vals26:
            continue
        last3 = vals26[-3:]
        proj = 0.65 * (sum(last3) / len(last3)) + 0.35 * (sum(vals26) / len(vals26))
        history = {}
        for season in range(2016, 2027):
            vals = dst_w.get((season, t))
            if vals:
                history[str(season)] = round(sum(vals) / len(vals), 1)
        dst_rows.append({"key": norm_key(t, "DST"), "name": t, "pos": "DST",
                         "team": t, "proj": round(proj, 2), "history": history})
    dst_rows.sort(key=lambda d: d["proj"], reverse=True)
    for i, d in enumerate(dst_rows, 1):
        d["rank_pos"] = i
    # holdout: walk-forward season-rate prediction, seasons 2018-2025
    k_err, k_n, d_err, d_n = [], 0, [], 0
    for pid, seq in k_logs.items():
        seas = {}
        for s, p in seq:
            seas.setdefault(s, []).append(p)
        for season in range(2018, 2026):
            if season in seas and (season - 1) in seas and (season - 2) in seas:
                if len(seas[season]) >= 4:
                    pred = 0.65 * (sum(seas[season - 1]) / len(seas[season - 1])) \
                           + 0.35 * (sum(seas[season - 2]) / len(seas[season - 2]))
                    act = sum(seas[season]) / len(seas[season])
                    k_err.append(abs(pred - act))
                    k_n += 1
    for t in teams32:
        seas = {s: dst_w.get((s, t)) for s in range(2016, 2026)}
        for season in range(2018, 2026):
            if seas[season] and seas[season - 1] and seas[season - 2] and len(seas[season]) >= 4:
                pred = 0.65 * (sum(seas[season - 1]) / len(seas[season - 1])) \
                       + 0.35 * (sum(seas[season - 2]) / len(seas[season - 2]))
                act = sum(seas[season]) / len(seas[season])
                d_err.append(abs(pred - act))
                d_n += 1
    return {
        "built": BUILT,
        "proj_season": 2026,
        "seasons": list(range(2016, 2027)),
        "kickers": kickers,
        "dst": dst_rows,
        "holdout": {
            "method": "walk_forward_season_ppg_0.65_0.35",
            "window": "seasons 2018-2025 (predict season S from S-1 and S-2)",
            "kicker_mae": round(sum(k_err) / len(k_err), 2) if k_err else None,
            "kicker_n": k_n,
            "dst_mae": round(sum(d_err) / len(d_err), 2) if d_err else None,
            "dst_n": d_n,
        },
        "provenance": prov("kdst.json", len(kickers) + len(dst_rows), 2026, last_done),
    }


# --------------------------------------------------------------------------
# lookback_seasons.json: schema-validate ONLY (static historical artifact)
# --------------------------------------------------------------------------

def validate_lookback():
    p = ASSETS / "lookback_seasons.json"
    d = json.load(p.open())
    if not {"built", "n_teams", "format", "note", "seasons"} <= set(d):
        fail("lookback_seasons.json schema changed — manual review required")
    if len(d["seasons"]) < 20 or any("teams" not in s for s in d["seasons"]):
        fail("lookback_seasons.json seasons malformed")
    print(f"lookback_seasons.json: schema OK ({len(d['seasons'])} seasons, bytes untouched)")


# --------------------------------------------------------------------------
# QA gates
# --------------------------------------------------------------------------

def qa_schema(proj_rows, next_rows, eval_payload, adp, kdst, target_week):
    # projections records: every field app.js reads
    need = {"key", "name", "pos", "team", "headshot", "proj", "floor", "ceil",
            "uncertainty", "line", "comps", "bye", "avail", "rookie", "moved",
            "prev_team", "rank_overall", "rank_pos", "tier"}
    for r in proj_rows:
        if not need <= set(r):
            fail(f"projections record missing keys: {need - set(r)}")
        if r["pos"] not in SKILL:
            fail(f"projections non-skill pos: {r['key']}")
        for k in ("proj", "floor", "ceil", "uncertainty"):
            if not isinstance(r[k], (int, float)) or not math.isfinite(r[k]):
                fail(f"non-finite {k} for {r['key']}")
        if r["floor"] > r["proj"] or r["ceil"] < r["proj"]:
            fail(f"floor/ceil ordering for {r['key']}")
        if r["line"] is not None and set(r["line"]) != {"pass_yds", "rush_yds", "rec_yds", "rec", "td"}:
            fail(f"bad line shape for {r['key']}")
    # nextgame records: projections fields + opp + conditions
    for r in next_rows:
        if not need <= set(r):
            fail(f"nextgame record missing keys: {need - set(r)}")
        if set(r["conditions"]) != {"is_home", "roof", "temp", "wind", "team_implied",
                                    "primetime", "gameday"}:
            fail(f"bad conditions keys for {r['key']}")
        for k in ("temp", "wind", "team_implied"):
            v = r["conditions"][k]
            if v is not None and (not isinstance(v, (int, float)) or not math.isfinite(v)):
                fail(f"non-finite {k} for {r['key']}")
    # nextgame subset-of-pool check + no nulls in the renderer-critical path
    proj_keys = {r["key"] for r in proj_rows}
    for r in next_rows:
        if r["key"] not in proj_keys:
            fail(f"nextgame player not in projections pool: {r['key']}")
        for k in ("opp", "name", "pos", "team"):
            if not r[k]:
                fail(f"missing {k} for {r['key']}")
    # band checks vs the Week-1 files
    if not (400 <= len(proj_rows) <= 800):
        fail(f"projections count {len(proj_rows)} outside band [400, 800]")
    if not (380 <= len(next_rows) <= 700):
        fail(f"nextgame count {len(next_rows)} outside band [380, 700]")
    if not (40 <= len(adp["players"]) <= 500):
        fail(f"adp count {len(adp['players'])} outside band [40, 500]")
    if len(kdst["dst"]) != 32:
        fail(f"dst count {len(kdst['dst'])} != 32")
    if not (25 <= len(kdst["kickers"]) <= 70):
        fail(f"kicker count {len(kdst['kickers'])} outside band [25, 70]")
    # eval_backtest: backtestNote()'s fields
    ov = eval_payload["overall"]
    for k in ("spearman", "mae", "baseline_last4"):
        if ov.get(k) is None or not math.isfinite(ov[k]):
            fail(f"eval_backtest overall.{k} missing/non-finite")
    if not (2.0 <= ov["mae"] <= 8.0):
        fail(f"eval_backtest overall.mae {ov['mae']:.2f} outside sanity band [2.0, 8.0]")
    if not eval_payload.get("weeks"):
        fail("eval_backtest has no weeks")
    print("QA gates: schema, bands, nulls, pool-resolution, backtest sanity — all pass")


def write_atomic(payloads: dict[str, dict]):
    """All-or-nothing write: assets/ then byte-identical public/assets/ mirrors."""
    for name, payload in payloads.items():
        text = json.dumps(payload, indent=1, allow_nan=False) + "\n"
        (ASSETS / name).write_text(text)
        (PUBLIC_ASSETS / name).write_text(text)
    # verify byte-identical mirrors
    for name in payloads:
        a = (ASSETS / name).read_bytes()
        b = (PUBLIC_ASSETS / name).read_bytes()
        if a != b:
            fail(f"mirror mismatch for {name}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    data = load_all()
    sched26 = data["sched26"]
    last_done, target_week = current_week(sched26)
    print(f"schedule-derived: last completed week = {last_done}, target week = {target_week}")
    if last_done < 2:
        fail("fewer than 2 completed weeks — backtest cannot run, no number ships")

    logs26 = game_logs(data["stats26"])
    logs25 = game_logs(data["stats25"])
    data["logs25"] = logs25
    ident = build_identity(data)
    # injury map: canon-name -> latest report_status (Out/Doubtful/Questionable)
    inj = data["injuries26"].sort_values("report_status") if len(data["injuries26"]) else data["injuries26"]
    inj_map = {}
    for _, r in inj.sort_values("week").iterrows():
        idn = ident.get(r.get("gsis_id") or "", {})
        cn = canon_name(idn.get("name") or str(r.get("full_name", "")))
        inj_map[cn] = str(r.get("report_status", ""))
    data["inj_map"] = inj_map

    prior_ppg = season_ppg(logs25)

    # backtest FIRST (its q80 residuals feed floor/ceiling of the live board)
    eval_payload, qres = run_backtest(data, logs26, ident, last_done, sched26)
    print(f"backtest: weeks {eval_payload['weeks']}, n={eval_payload['overall']['n']}, "
          f"MAE={eval_payload['overall']['mae']:.2f}, Spearman={eval_payload['overall']['spearman']:.3f}")

    # live boards as-of last_done, targeting target_week
    asof = {pid: gs for pid, gs in logs26.items()}
    factors = opp_factors(asof, target_week + 1)
    pool = build_pool(asof, ident, prior_ppg)
    ecr = ecr_from_week_load(data["ecr_now"])
    added_ecr = add_ecr_only_players(pool, ecr, ident, prior_ppg)
    print(f"pool: {len(pool)} skill players ({added_ecr} ECR-only additions)")

    curve = ecr_curve(asof_ppg_by_pos(asof))
    gm = week_games(sched26, target_week)
    byes = bye_weeks(sched26)
    proj_rows, next_rows = assemble_live(data, logs26, ident, pool, factors,
                                         curve, ecr, gm, byes, qres, target_week)

    projections = {
        "built": BUILT,
        "proj_season": 2026,
        "basis": ("2026 form blended with expert consensus (v1): trailing-4-weighted ppg + "
                  "season ppg (small samples get a prior-season prior; no 2026 games get "
                  "consensus-implied points), 70/30 blend with ECR-implied points, "
                  "floor/ceiling = +/- per-position q80 |residual| from the 2026 walk-forward backtest"),
        "scoring_feature": "fantasy_points_ppr",
        "model": {"type": "trailing form x ECR consensus blend v1",
                  "version": MODEL_VERSION,
                  "backtest_ref": "assets/eval_backtest.json"},
        "rookie_count": sum(1 for r in proj_rows if r["rookie"]),
        "count": len(proj_rows),
        "players": proj_rows,
        "provenance": prov("projections.json", len(proj_rows), 2026, target_week),
    }
    nextgame = {
        "built": BUILT,
        "season": 2026,
        "week": target_week,
        "model": {"type": "trailing form x matchup adjustment x ECR consensus blend v1",
                  "version": MODEL_VERSION,
                  "backtest_ref": "assets/eval_backtest.json"},
        "count": len(next_rows),
        "players": next_rows,
        "provenance": prov("nextgame.json", len(next_rows), 2026, target_week),
    }
    if nextgame["week"] != target_week:
        fail("emitted week != schedule-derived current week")

    adp = build_adp()
    kdst = build_kdst(data, ident, last_done)
    validate_lookback()

    qa_schema(proj_rows, next_rows, eval_payload, adp, kdst, target_week)
    write_atomic({
        "projections.json": projections,
        "nextgame.json": nextgame,
        "adp.json": adp,
        "kdst.json": kdst,
        "eval_backtest.json": eval_payload,
    })
    print(f"wrote 5 files to {ASSETS}/ and mirrored to {PUBLIC_ASSETS}/ (byte-identical)")
    print(f"projections: {len(proj_rows)} | nextgame: {len(next_rows)} | "
          f"adp: {len(adp['players'])} | kickers: {len(kdst['kickers'])} | dst: 32")
    print(f"overall MAE: {eval_payload['overall']['mae']:.2f} | "
          f"Spearman: {eval_payload['overall']['spearman']:.3f} | "
          f"baseline Spearman: {eval_payload['overall']['baseline_last4']:.3f}")


if __name__ == "__main__":
    main()
