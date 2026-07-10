"""Holistic multi-family player-week feature builder for Vector Gridiron MTNN v2.

Leakage-safe: every form/usage/opportunity/NGS/PFR trailing feature uses PRIOR
weeks only. Emits a masked family matrix for multi-tower training:

  form, usage, opportunity, role, availability, meta, conditions, market,
  defense, ngs, pfr_adv, context, pedigree

Also builds upcoming-week rows for nextgame / season projection export.

Run:  python pipeline/build_features.py [--offline]
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from math import log1p
from pathlib import Path

import numpy as np

import build_opportunity as opp
import nfl_data as nfl
from nfl_data import num

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "pipeline" / "data"
SKILL = {"QB", "RB", "WR", "TE"}
TRAIL = 4
MIN_PRIOR = 1
# Hill-climb expand (EWMA / pass-rush DvP / snapΔ / RZ proxy / WR1) — held-out 2025
# MAE 4.278–4.378 vs promoted 4.268. Keep code path; default OFF until it earns promote.
HILL_CLIMB_FEATURES = False
# True pbp red-zone shares (nflverse) — separate bet from the failed expand bundle.
RZ_FEATURES = True

# ---------------------------------------------------------------------------
# Family feature specs (order = column order within family)
# ---------------------------------------------------------------------------

FORM_KEYS = [
    "fpts_ppr", "rec_yds", "rush_yds", "pass_yds", "receptions", "total_td",
    "epa", "pass_att", "targets", "carries",
]
USAGE_KEYS = [
    "snap_pct", "target_share", "air_yards_share", "wopr", "touches",
    "racr", "pacr", "cpoe",
] + (["snap_delta"] if HILL_CLIMB_FEATURES else [])
OPP_KEYS = [
    "ep_fpts", "ep_diff", "rec_attempt", "rush_attempt", "td_exp", "rec_air_yards",
] + (["td_rate", "rz_proxy"] if HILL_CLIMB_FEATURES else []) + (
    ["rz_tgt_share", "rz_carry_share", "inside5_share"] if RZ_FEATURES else []
)
ROLE_KEYS = ["depth_rank", "is_starter", "depth_ahead"]
AVAIL_KEYS = ["inj_out", "inj_doubtful", "inj_questionable", "games_missed_4"]
META_KEYS = ["age", "exp", "height", "weight", "is_QB", "is_RB", "is_WR", "is_TE"]
COND_KEYS = [
    "is_home", "rest_days", "is_div", "is_indoor", "is_grass", "temp", "wind",
    "kick_hour", "is_primetime", "is_thu", "is_mon", "week_no",
]
MARKET_KEYS = ["team_implied", "opp_implied", "spread_team", "total_line"]
DEF_KEYS = ["dvp_allowed", "dvp_roll4"] + (
    ["dvp_pass", "dvp_rush", "dvp_pass_roll4"] if HILL_CLIMB_FEATURES else []
)
NGS_KEYS = [
    "ngs_sep", "ngs_cushion", "ngs_yac_oe", "ngs_air_share", "ngs_ryoe",
    "ngs_eff", "ngs_cpoe", "ngs_ttt", "ngs_aggr",
]
PFR_KEYS = [
    "pfr_ybc_avg", "pfr_yac_avg", "pfr_broken", "pfr_drop_pct",
    "pfr_pressure_pct", "pfr_bad_throw_pct",
]
CTX_KEYS = ["qb_ep_fpts", "team_pass_rate", "committee_hhi"] + (
    ["wr1_target_share"] if HILL_CLIMB_FEATURES else []
)
PED_KEYS = ["draft_pick_log", "draft_round", "combine_forty", "combine_vertical"]

_FORM_EXTRA = (
    ["ewma_ppr_3", "ewma_ppr_5", "ewma_ppr_8"] if HILL_CLIMB_FEATURES else []
)
FAMILIES: dict[str, list[str]] = {
    "form": ["f_" + k for k in FORM_KEYS] + ["std_ppr", "games_played", "prior_ppg"] + _FORM_EXTRA,
    "usage": ["u_" + k for k in USAGE_KEYS],
    "opportunity": ["o_" + k for k in OPP_KEYS],
    "role": ["r_" + k for k in ROLE_KEYS],
    "availability": ["a_" + k for k in AVAIL_KEYS],
    "meta": META_KEYS,
    "conditions": COND_KEYS,
    "market": MARKET_KEYS,
    "defense": DEF_KEYS,
    "ngs": ["n_" + k for k in NGS_KEYS],
    "pfr_adv": ["p_" + k for k in PFR_KEYS],
    "context": ["c_" + k for k in CTX_KEYS],
    "pedigree": ["d_" + k for k in PED_KEYS],
}

FEATURE_NAMES: list[str] = []
FAMILY_OF: dict[str, str] = {}
for fam, cols in FAMILIES.items():
    for c in cols:
        FEATURE_NAMES.append(c)
        FAMILY_OF[c] = fam

TARGET_NAMES = ["fpts_ppr", "rec_yds", "rush_yds", "pass_yds", "receptions", "total_td"]
USAGE_RECON_NAMES = ["target_share", "snap_pct", "carries"]


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------

def player_meta(offline: bool) -> dict:
    meta = {}
    for r in nfl.players(offline):
        gsis = r.get("gsis_id")
        if not gsis:
            continue
        meta[gsis] = {
            "pfr": r.get("pfr_id", ""),
            "birth": r.get("birth_date", ""),
            "rookie": num(r, "rookie_season", 0),
            "height": num(r, "height", 73),
            "weight": num(r, "weight", 210),
            "exp": num(r, "years_of_experience", 3),
            "latest_team": (r.get("latest_team") or "").strip(),
            "status": (r.get("status") or "").strip(),
            "draft_pick": num(r, "draft_pick", 0),
            "draft_round": num(r, "draft_round", 0),
        }
    return meta


def parse_date(s: str):
    try:
        y, m, d = s.split("-")[:3]
        return date(int(y), int(m), int(d))
    except Exception:
        return None


def team_byes(season: int, offline: bool) -> dict:
    weeks_by_team: dict[str, set] = {}
    max_week = 0
    for r in nfl.games(offline):
        if int(num(r, "season")) != season:
            continue
        w = int(num(r, "week"))
        max_week = max(max_week, w)
        for t in (r.get("home_team", ""), r.get("away_team", "")):
            weeks_by_team.setdefault(t, set()).add(w)
    byes = {}
    for t, played in weeks_by_team.items():
        miss = [w for w in range(1, max_week + 1) if w not in played]
        if miss:
            byes[t] = miss[0]
    return byes


def game_context(offline: bool) -> dict:
    ctx = {}
    for r in nfl.games(offline):
        season = int(num(r, "season"))
        week = int(num(r, "week"))
        home, away = r.get("home_team", ""), r.get("away_team", "")
        total = num(r, "total_line", 44.0)
        spread = num(r, "spread_line", 0.0)
        roof = (r.get("roof") or "outdoors").lower()
        surface = (r.get("surface") or "grass").lower()
        indoor = 1.0 if roof in ("dome", "closed") else 0.0
        temp = num(r, "temp", 68.0 if indoor else 60.0)
        wind = num(r, "wind", 0.0)
        gametime = r.get("gametime", "") or ""
        hour = int(gametime.split(":")[0]) if ":" in gametime else 13
        weekday = (r.get("weekday") or "Sunday")
        primetime = 1.0 if (hour >= 19 or weekday in ("Thursday", "Monday", "Saturday")) else 0.0
        gday = r.get("gameday", "")
        home_imp = total / 2 + spread / 2
        away_imp = total / 2 - spread / 2
        base = dict(
            week=week, roof=roof, is_indoor=indoor,
            is_grass=1.0 if "grass" in surface else 0.0,
            temp=temp, wind=wind, kick_hour=float(hour), is_primetime=primetime,
            is_thu=1.0 if weekday == "Thursday" else 0.0,
            is_mon=1.0 if weekday == "Monday" else 0.0,
            total_line=total, div=num(r, "div_game", 0.0), gday=gday,
            home_qb=r.get("home_qb_id", ""), away_qb=r.get("away_qb_id", ""),
        )
        ctx[(season, week, home)] = {
            **base, "is_home": 1.0, "opp": away, "rest": num(r, "home_rest", 7),
            "team_implied": home_imp, "opp_implied": away_imp, "spread_team": spread,
            "qb_id": r.get("home_qb_id", ""),
        }
        ctx[(season, week, away)] = {
            **base, "is_home": 0.0, "opp": home, "rest": num(r, "away_rest", 7),
            "team_implied": away_imp, "opp_implied": home_imp, "spread_team": -spread,
            "qb_id": r.get("away_qb_id", ""),
        }
    return ctx


def weekly_record(r: dict, snap_pct: float) -> dict:
    rec_td = num(r, "receiving_tds")
    rush_td = num(r, "rushing_tds")
    pass_td = num(r, "passing_tds")
    rec_yds = num(r, "receiving_yards")
    rush_yds = num(r, "rushing_yards")
    pass_yds = num(r, "passing_yards")
    receptions = num(r, "receptions")
    # Component fantasy (half-ish) for DvP pass vs rush splits.
    recv_fpts = rec_yds / 10.0 + receptions + rec_td * 6.0
    rush_fpts = rush_yds / 10.0 + rush_td * 6.0
    pass_fpts = pass_yds / 25.0 + pass_td * 4.0
    return {
        "gsis": r.get("player_id"),
        "name": r.get("player_display_name") or r.get("player_name"),
        "pos": (r.get("position") or "").strip(),
        "team": r.get("team", ""),
        "opp": r.get("opponent_team", ""),
        "week": int(num(r, "week")),
        "headshot": r.get("headshot_url", ""),
        "fpts_ppr": num(r, "fantasy_points_ppr"),
        "targets": num(r, "targets"),
        "carries": num(r, "carries"),
        "receptions": receptions,
        "touches": num(r, "carries") + receptions,
        "rec_yds": rec_yds,
        "rush_yds": rush_yds,
        "pass_yds": pass_yds,
        "pass_att": num(r, "attempts"),
        "target_share": num(r, "target_share"),
        "air_yards_share": num(r, "air_yards_share"),
        "wopr": num(r, "wopr"),
        "racr": num(r, "racr"),
        "pacr": num(r, "pacr"),
        "cpoe": num(r, "passing_cpoe"),
        "epa": num(r, "passing_epa") + num(r, "rushing_epa") + num(r, "receiving_epa"),
        "total_td": pass_td + rush_td + rec_td,
        "recv_fpts": recv_fpts,
        "rush_fpts": rush_fpts,
        "pass_fpts": pass_fpts,
        "snap_pct": snap_pct,
    }


def index_depth(year: int, offline: bool) -> dict:
    """(week, gsis) -> {depth_rank, is_starter, depth_ahead} for offense."""
    rows = nfl.depth_charts(year, offline)
    by_team_week_pos: dict[tuple, list] = {}
    out = {}
    for r in rows:
        gsis = (r.get("gsis_id") or "").strip()
        if not gsis:
            continue
        form = (r.get("formation") or "").lower()
        if form and form not in ("offense", "o", ""):
            # keep offense-ish; some dumps use empty formation
            if "def" in form or form in ("special teams", "st"):
                continue
        week = int(num(r, "week", 0))
        if week <= 0:
            continue
        team = r.get("club_code") or r.get("team") or ""
        pos = (r.get("depth_position") or r.get("position") or "").strip().upper()
        rank = int(num(r, "depth_team", 3))
        key = (week, team, pos)
        by_team_week_pos.setdefault(key, []).append((rank, gsis))
        out[(week, gsis)] = {"depth_rank": float(rank), "is_starter": 1.0 if rank == 1 else 0.0}
    for (week, team, pos), arr in by_team_week_pos.items():
        arr.sort()
        for i, (rank, gsis) in enumerate(arr):
            if (week, gsis) in out:
                out[(week, gsis)]["depth_ahead"] = float(i)
    return out


def index_injuries(year: int, offline: bool) -> dict:
    """(week, gsis) -> status string."""
    out = {}
    for r in nfl.injuries(year, offline):
        gsis = r.get("gsis_id") or r.get("player_id") or ""
        if not gsis:
            continue
        week = int(num(r, "week"))
        st = (r.get("report_status") or r.get("injury_status") or "").strip()
        if st:
            out[(week, gsis)] = st
    return out


def index_ngs(year: int, offline: bool) -> dict:
    """(week, gsis) -> ngs feature dict (position-appropriate fields filled)."""
    out: dict[tuple, dict] = {}

    def ensure(week, gsis):
        k = (week, gsis)
        if k not in out:
            out[k] = {k2: 0.0 for k2 in NGS_KEYS}
            out[k]["_mask"] = 0.0
        return out[k]

    for r in nfl.ngs("receiving", year, offline):
        gsis = r.get("player_gsis_id") or ""
        week = int(num(r, "week"))
        if not gsis or week <= 0:
            continue
        d = ensure(week, gsis)
        d["ngs_sep"] = num(r, "avg_separation")
        d["ngs_cushion"] = num(r, "avg_cushion")
        d["ngs_yac_oe"] = num(r, "avg_yac_above_expectation")
        d["ngs_air_share"] = num(r, "percent_share_of_intended_air_yards")
        d["_mask"] = 1.0
    for r in nfl.ngs("rushing", year, offline):
        gsis = r.get("player_gsis_id") or ""
        week = int(num(r, "week"))
        if not gsis or week <= 0:
            continue
        d = ensure(week, gsis)
        d["ngs_ryoe"] = num(r, "rush_yards_over_expected_per_att")
        d["ngs_eff"] = num(r, "efficiency")
        d["_mask"] = 1.0
    for r in nfl.ngs("passing", year, offline):
        gsis = r.get("player_gsis_id") or ""
        week = int(num(r, "week"))
        if not gsis or week <= 0:
            continue
        d = ensure(week, gsis)
        d["ngs_cpoe"] = num(r, "completion_percentage_above_expectation")
        d["ngs_ttt"] = num(r, "avg_time_to_throw")
        d["ngs_aggr"] = num(r, "aggressiveness")
        d["_mask"] = 1.0
    return out


def index_pfr(year: int, offline: bool) -> dict:
    out: dict[tuple, dict] = {}

    def ensure(week, gsis_pfr):
        k = (week, gsis_pfr)
        if k not in out:
            out[k] = {k2: 0.0 for k2 in PFR_KEYS}
            out[k]["_mask"] = 0.0
        return out[k]

    for r in nfl.pfr_adv("rush", year, offline):
        pid = r.get("pfr_player_id") or ""
        week = int(num(r, "week"))
        if not pid or week <= 0:
            continue
        d = ensure(week, pid)
        d["pfr_ybc_avg"] = num(r, "rushing_yards_before_contact_avg")
        d["pfr_yac_avg"] = num(r, "rushing_yards_after_contact_avg")
        d["pfr_broken"] = num(r, "rushing_broken_tackles")
        d["_mask"] = 1.0
    for r in nfl.pfr_adv("rec", year, offline):
        pid = r.get("pfr_player_id") or ""
        week = int(num(r, "week"))
        if not pid or week <= 0:
            continue
        d = ensure(week, pid)
        d["pfr_drop_pct"] = num(r, "receiving_drop_pct")
        d["pfr_broken"] = max(d["pfr_broken"], num(r, "receiving_broken_tackles"))
        d["_mask"] = 1.0
    for r in nfl.pfr_adv("pass", year, offline):
        pid = r.get("pfr_player_id") or ""
        week = int(num(r, "week"))
        if not pid or week <= 0:
            continue
        d = ensure(week, pid)
        d["pfr_pressure_pct"] = num(r, "times_pressured_pct")
        d["pfr_bad_throw_pct"] = num(r, "passing_bad_throw_pct")
        d["_mask"] = 1.0
    return out


def pedigree_table(offline: bool) -> dict:
    """gsis -> pedigree features."""
    out = {}
    for r in nfl.draft_picks(offline):
        gsis = r.get("gsis_id") or ""
        if not gsis:
            continue
        pick = num(r, "pick", 0)
        rnd = num(r, "round", 8)
        out[gsis] = {
            "draft_pick_log": log1p(pick) if pick > 0 else log1p(250),
            "draft_round": rnd if rnd > 0 else 8.0,
            "combine_forty": 0.0,
            "combine_vertical": 0.0,
            "_mask": 1.0 if pick > 0 else 0.0,
        }
    # combine keyed by pfr_id — join via players later in row build
    comb = {}
    for r in nfl.combine(offline):
        pid = r.get("pfr_id") or ""
        if pid:
            comb[pid] = (num(r, "forty"), num(r, "vertical"))
    return out, comb


def load_season(season: int, meta: dict, offline: bool):
    stats = nfl.weekly_stats(season, offline)
    if not stats:
        return {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    snap_by_pfr = {}
    for s in nfl.snaps(season, offline):
        if s.get("game_type") not in ("REG", "regular", ""):
            if s.get("game_type") and s.get("game_type") != "REG":
                continue
        snap_by_pfr[(int(num(s, "week")), s.get("pfr_player_id"))] = num(s, "offense_pct")
    by_player = {}
    allowed = {}
    allowed_pass = {}
    allowed_rush = {}
    team_week_touches = {}
    team_week_targets = {}
    for r in stats:
        if r.get("season_type") != "REG":
            continue
        pos = (r.get("position") or "").strip()
        if pos not in SKILL:
            continue
        gsis = r.get("player_id")
        pfr = meta.get(gsis, {}).get("pfr", "")
        wk = int(num(r, "week"))
        rec = weekly_record(r, snap_by_pfr.get((wk, pfr), 0.0))
        by_player.setdefault(gsis, []).append(rec)
        key = (wk, rec["opp"], pos)
        allowed[key] = allowed.get(key, 0.0) + rec["fpts_ppr"]
        # Pass-game DvP: QB pass fantasy + WR/TE/RB receiving fantasy vs that D.
        if pos == "QB":
            allowed_pass[key] = allowed_pass.get(key, 0.0) + rec["pass_fpts"]
        else:
            allowed_pass[key] = allowed_pass.get(key, 0.0) + rec["recv_fpts"]
        allowed_rush[key] = allowed_rush.get(key, 0.0) + rec["rush_fpts"]
        tk = (wk, rec["team"])
        team_week_touches[tk] = team_week_touches.get(tk, 0.0) + rec["touches"]
        team_week_targets[tk] = team_week_targets.get(tk, 0.0) + rec["targets"]
    for arr in by_player.values():
        arr.sort(key=lambda g: g["week"])
    depth = index_depth(season, offline)
    inj = index_injuries(season, offline)
    ngs_i = index_ngs(season, offline)
    pfr_i = index_pfr(season, offline)
    return (by_player, allowed, allowed_pass, allowed_rush, depth, inj, ngs_i, pfr_i,
            team_week_touches, team_week_targets)


def season_ppg(by_player):
    return {g: sum(x["fpts_ppr"] for x in arr) / max(1, len(arr))
            for g, arr in by_player.items()}


def dvp_prior(allowed, week, defteam, pos, league_avg):
    vals = [allowed[(w, defteam, pos)] for w in range(1, week)
            if (w, defteam, pos) in allowed]
    return sum(vals) / len(vals) if vals else league_avg


def dvp_roll4(allowed, week, defteam, pos, league_avg):
    vals = [allowed[(w, defteam, pos)] for w in range(max(1, week - 4), week)
            if (w, defteam, pos) in allowed]
    return sum(vals) / len(vals) if vals else league_avg


def ewma_series(vals: list[float], span: int) -> float:
    """Exponentially weighted mean over chronological vals (oldest→newest)."""
    if not vals:
        return 0.0
    alpha = 2.0 / (span + 1.0)
    s = float(vals[0])
    for v in vals[1:]:
        s = alpha * float(v) + (1.0 - alpha) * s
    return s


def mean_keys(hist, keys):
    if not hist:
        return [0.0] * len(keys)
    return [float(np.mean([g.get(k, 0.0) for g in hist])) for k in keys]


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------

def assemble_row(
    hist, g, c, meta_row, pos, dvp, dvp4, dvp_pass, dvp_rush, dvp_pass4,
    depth_row, inj_hist, ngs_hist, pfr_hist,
    ep_hist, rz_hist, qb_ep, team_pass_rate, committee_hhi, wr1_share, ped, prior_ppg, season,
):
    """Return (values[list], mask[list]) aligned to FEATURE_NAMES."""
    vals = {}
    mask = {}

    # form — always observed if hist non-empty
    window = hist[-TRAIL:]
    for k in FORM_KEYS:
        vals["f_" + k] = float(np.mean([x.get(k, 0.0) for x in window]))
        mask["f_" + k] = 1.0
    vals["std_ppr"] = float(np.mean([x["fpts_ppr"] for x in hist]))
    vals["games_played"] = float(len(hist))
    vals["prior_ppg"] = float(prior_ppg)
    mask["std_ppr"] = mask["games_played"] = mask["prior_ppg"] = 1.0
    if HILL_CLIMB_FEATURES:
        ppg_hist = [x["fpts_ppr"] for x in hist]
        vals["ewma_ppr_3"] = ewma_series(ppg_hist, 3)
        vals["ewma_ppr_5"] = ewma_series(ppg_hist, 5)
        vals["ewma_ppr_8"] = ewma_series(ppg_hist, 8)
        mask["ewma_ppr_3"] = mask["ewma_ppr_5"] = mask["ewma_ppr_8"] = 1.0

    # usage
    for k in USAGE_KEYS:
        if k == "snap_delta":
            continue
        vals["u_" + k] = float(np.mean([x.get(k, 0.0) for x in window]))
        mask["u_" + k] = 1.0
    if HILL_CLIMB_FEATURES:
        snaps = [x.get("snap_pct", 0.0) for x in hist]
        if len(snaps) >= 2:
            vals["u_snap_delta"] = float(snaps[-1] - np.mean(snaps[:-1][-3:]))
            mask["u_snap_delta"] = 1.0
        else:
            vals["u_snap_delta"] = 0.0
            mask["u_snap_delta"] = 0.0

    # opportunity (EP) — mask if no prior EP weeks
    ep_core = ("ep_fpts", "ep_diff", "rec_attempt", "rush_attempt", "td_exp", "rec_air_yards")
    if ep_hist:
        for k in ep_core:
            vals["o_" + k] = float(np.mean([x.get(k, 0.0) for x in ep_hist[-TRAIL:]]))
            mask["o_" + k] = 1.0
    else:
        for k in ep_core:
            vals["o_" + k] = 0.0
            mask["o_" + k] = 0.0
    if HILL_CLIMB_FEATURES:
        if hist:
            vals["o_td_rate"] = float(np.mean([x.get("total_td", 0.0) for x in window]))
            vals["o_rz_proxy"] = float(np.mean([1.0 if x.get("total_td", 0) >= 1 else 0.0 for x in window]))
            mask["o_td_rate"] = mask["o_rz_proxy"] = 1.0
        else:
            vals["o_td_rate"] = vals["o_rz_proxy"] = 0.0
            mask["o_td_rate"] = mask["o_rz_proxy"] = 0.0
    if RZ_FEATURES:
        if rz_hist:
            for k in ("rz_tgt_share", "rz_carry_share", "inside5_share"):
                vals["o_" + k] = float(np.mean([x.get(k, 0.0) for x in rz_hist[-TRAIL:]]))
                mask["o_" + k] = 1.0
        else:
            for k in ("rz_tgt_share", "rz_carry_share", "inside5_share"):
                vals["o_" + k] = 0.0
                mask["o_" + k] = 0.0

    # role
    if depth_row:
        vals["r_depth_rank"] = depth_row.get("depth_rank", 3.0)
        vals["r_is_starter"] = depth_row.get("is_starter", 0.0)
        vals["r_depth_ahead"] = depth_row.get("depth_ahead", 0.0)
        mask["r_depth_rank"] = mask["r_is_starter"] = mask["r_depth_ahead"] = 1.0
    else:
        for k in ROLE_KEYS:
            vals["r_" + k] = 0.0
            mask["r_" + k] = 0.0

    # availability — mask entire family for season > 2024
    if season > nfl.INJURIES_LAST_SEASON:
        for k in AVAIL_KEYS:
            vals["a_" + k] = 0.0
            mask["a_" + k] = 0.0
    else:
        last_st = inj_hist[-1] if inj_hist else ""
        vals["a_inj_out"] = 1.0 if last_st.lower() == "out" else 0.0
        vals["a_inj_doubtful"] = 1.0 if last_st.lower() == "doubtful" else 0.0
        vals["a_inj_questionable"] = 1.0 if last_st.lower() == "questionable" else 0.0
        vals["a_games_missed_4"] = float(sum(1 for s in inj_hist[-4:] if s.lower() == "out"))
        for k in AVAIL_KEYS:
            mask["a_" + k] = 1.0

    # meta
    bd = parse_date(meta_row.get("birth", ""))
    gd = parse_date(c.get("gday", ""))
    age = (gd - bd).days / 365.25 if (bd and gd) else 26.0
    exp = meta_row.get("exp", 3.0) or 3.0
    vals["age"] = age
    vals["exp"] = float(exp)
    vals["height"] = float(meta_row.get("height", 73))
    vals["weight"] = float(meta_row.get("weight", 210))
    for p in ("QB", "RB", "WR", "TE"):
        vals[f"is_{p}"] = 1.0 if pos == p else 0.0
    for k in META_KEYS:
        mask[k] = 1.0

    # conditions + market
    vals["is_home"] = c["is_home"]
    vals["rest_days"] = c["rest"]
    vals["is_div"] = c["div"]
    vals["is_indoor"] = c["is_indoor"]
    vals["is_grass"] = c["is_grass"]
    vals["temp"] = c["temp"]
    vals["wind"] = c["wind"]
    vals["kick_hour"] = c["kick_hour"]
    vals["is_primetime"] = c["is_primetime"]
    vals["is_thu"] = c["is_thu"]
    vals["is_mon"] = c["is_mon"]
    vals["week_no"] = float(c["week"])
    for k in COND_KEYS:
        mask[k] = 1.0
    vals["team_implied"] = c["team_implied"]
    vals["opp_implied"] = c["opp_implied"]
    vals["spread_team"] = c["spread_team"]
    vals["total_line"] = c["total_line"]
    for k in MARKET_KEYS:
        mask[k] = 1.0

    # defense
    vals["dvp_allowed"] = dvp
    vals["dvp_roll4"] = dvp4
    mask["dvp_allowed"] = mask["dvp_roll4"] = 1.0
    if HILL_CLIMB_FEATURES:
        vals["dvp_pass"] = dvp_pass
        vals["dvp_rush"] = dvp_rush
        vals["dvp_pass_roll4"] = dvp_pass4
        for k in ("dvp_pass", "dvp_rush", "dvp_pass_roll4"):
            mask[k] = 1.0

    # ngs
    if ngs_hist:
        last = ngs_hist[-TRAIL:]
        for k in NGS_KEYS:
            vals["n_" + k] = float(np.mean([x.get(k, 0.0) for x in last]))
            mask["n_" + k] = 1.0
    else:
        for k in NGS_KEYS:
            vals["n_" + k] = 0.0
            mask["n_" + k] = 0.0

    # pfr
    if pfr_hist:
        last = pfr_hist[-TRAIL:]
        for k in PFR_KEYS:
            vals["p_" + k] = float(np.mean([x.get(k, 0.0) for x in last]))
            mask["p_" + k] = 1.0
    else:
        for k in PFR_KEYS:
            vals["p_" + k] = 0.0
            mask["p_" + k] = 0.0

    # context
    vals["c_qb_ep_fpts"] = qb_ep
    vals["c_team_pass_rate"] = team_pass_rate
    vals["c_committee_hhi"] = committee_hhi
    if HILL_CLIMB_FEATURES:
        vals["c_wr1_target_share"] = wr1_share
    for k in CTX_KEYS:
        mask["c_" + k] = 1.0
        # soft-mask: always feed context defaults
        mask["c_" + k] = 1.0

    # pedigree
    if ped and ped.get("_mask", 0) > 0:
        vals["d_draft_pick_log"] = ped["draft_pick_log"]
        vals["d_draft_round"] = ped["draft_round"]
        vals["d_combine_forty"] = ped.get("combine_forty", 0.0)
        vals["d_combine_vertical"] = ped.get("combine_vertical", 0.0)
        for k in PED_KEYS:
            mask["d_" + k] = 1.0 if vals["d_" + k] or k.startswith("draft") else 0.0
        mask["d_draft_pick_log"] = mask["d_draft_round"] = 1.0
        mask["d_combine_forty"] = 1.0 if vals["d_combine_forty"] else 0.0
        mask["d_combine_vertical"] = 1.0 if vals["d_combine_vertical"] else 0.0
    else:
        for k in PED_KEYS:
            vals["d_" + k] = 0.0
            mask["d_" + k] = 0.0

    v = [vals[n] for n in FEATURE_NAMES]
    m = [mask[n] for n in FEATURE_NAMES]
    return v, m


def build(last_season: int | None = None, offline: bool = False) -> dict:
    last_season = last_season or nfl.latest_stats_season(offline)
    meta = player_meta(offline)
    ctx = game_context(offline)
    ped_by_gsis, comb_by_pfr = pedigree_table(offline)
    for gsis, m in meta.items():
        pfr = m.get("pfr", "")
        if gsis not in ped_by_gsis:
            pick = m.get("draft_pick", 0) or 0
            rnd = m.get("draft_round", 0) or 0
            ped_by_gsis[gsis] = {
                "draft_pick_log": log1p(pick) if pick > 0 else log1p(250),
                "draft_round": rnd if rnd > 0 else 8.0,
                "combine_forty": 0.0,
                "combine_vertical": 0.0,
                "_mask": 1.0 if pick > 0 else 0.0,
            }
        if pfr in comb_by_pfr:
            forty, vert = comb_by_pfr[pfr]
            ped_by_gsis[gsis]["combine_forty"] = forty
            ped_by_gsis[gsis]["combine_vertical"] = vert

    seasons = [y for y in range(nfl.FIRST_SEASON, last_season + 1)
               if nfl.weekly_stats(y, offline)]
    if not seasons:
        raise SystemExit("no weekly-stats seasons available")

    print("loading opportunity (ffopportunity EP) ...")
    ep_idx = opp.load_ep_index(seasons, offline)
    rz_idx = {}
    if RZ_FEATURES:
        import build_rz as brz
        if not (DATA / "rz_index.json").exists() and not offline:
            print("building RZ index (first run) ...")
            brz.build(last_season, offline=offline)
        elif not (DATA / "rz_index.json").exists() and offline:
            print("  rz_index.json missing and --offline — RZ features will be masked")
        else:
            # Refresh if stale seasons missing — rebuild when empty
            pass
        rz_raw = brz.load_index()
        # key "gsis|season|week" → dict
        for k, v in rz_raw.items():
            parts = k.split("|")
            if len(parts) == 3:
                gsis, ys, wk = parts[0], int(parts[1]), int(parts[2])
                rz_idx[(gsis, ys, wk)] = v
        print(f"  RZ index: {len(rz_idx)} player-weeks")

    per_season = {}
    for s in seasons:
        print(f"  loading season {s} ...")
        (bp, allowed, allowed_pass, allowed_rush, depth, inj, ngs_i, pfr_i,
         twt, twtgt) = load_season(s, meta, offline)
        per_season[s] = {
            "bp": bp, "allowed": allowed, "allowed_pass": allowed_pass,
            "allowed_rush": allowed_rush, "depth": depth, "inj": inj,
            "ngs": ngs_i, "pfr": pfr_i, "twt": twt, "twtgt": twtgt,
            "ppg": season_ppg(bp),
        }

    league_avg_pos = {}
    league_avg_pass = {}
    league_avg_rush = {}
    for s in seasons:
        allowed = per_season[s]["allowed"]
        ap = per_season[s]["allowed_pass"]
        ar = per_season[s]["allowed_rush"]
        for pos in ("QB", "RB", "WR", "TE"):
            vals = [v for (w, d, p), v in allowed.items() if p == pos]
            league_avg_pos[(s, pos)] = float(np.mean(vals)) if vals else 12.0
            pv = [v for (w, d, p), v in ap.items() if p == pos]
            league_avg_pass[(s, pos)] = float(np.mean(pv)) if pv else 8.0
            rv = [v for (w, d, p), v in ar.items() if p == pos]
            league_avg_rush[(s, pos)] = float(np.mean(rv)) if rv else 4.0

    X, M, Y, META, Y_USAGE = [], [], [], [], []

    for s in seasons:
        bundle = per_season[s]
        bp, allowed = bundle["bp"], bundle["allowed"]
        allowed_pass, allowed_rush = bundle["allowed_pass"], bundle["allowed_rush"]
        prior_sp = per_season.get(s - 1, {}).get("ppg", {})
        # Precompute context maps for this season (avoid O(n²) scans).
        qb_ep_map = {}
        team_pass_map = {}  # (week, team) -> pass rate
        rb_touches = {}     # (week, team) -> list of RB touches
        wr_targets = {}     # (week, team) -> list of WR target shares
        gsis_team_week = {}
        for gsis, games in bp.items():
            for g0 in games:
                gsis_team_week[(gsis, g0["week"])] = (g0["team"], g0["pos"])
                if g0["pos"] == "RB":
                    rb_touches.setdefault((g0["week"], g0["team"]), []).append(g0["touches"])
                if g0["pos"] == "WR":
                    wr_targets.setdefault((g0["week"], g0["team"]), []).append(g0.get("target_share", 0.0))
        team_att = {}  # (week, team) -> [passish, rush]
        for (gsis, ys, wk), ev in ep_idx.items():
            if ys != s:
                continue
            tt = gsis_team_week.get((gsis, wk))
            if not tt:
                continue
            team, pos = tt
            if pos == "QB":
                qb_ep_map[(wk, team)] = ev["ep_fpts"]
            key = (wk, team)
            pa, ra = team_att.get(key, [0.0, 0.0])
            pa += ev.get("pass_attempt", 0) + ev.get("rec_attempt", 0)
            ra += ev.get("rush_attempt", 0)
            team_att[key] = [pa, ra]
        for key, (pa, ra) in team_att.items():
            team_pass_map[key] = pa / (pa + ra) if (pa + ra) > 0 else 0.55
        hhi_map = {}
        for key, shares in rb_touches.items():
            total = sum(shares)
            hhi_map[key] = sum((x / total) ** 2 for x in shares) if total > 0 else 0.0
        wr1_map = {key: max(shares) if shares else 0.0 for key, shares in wr_targets.items()}

        for gsis, games in bp.items():
            for i in range(len(games)):
                if i < MIN_PRIOR:
                    continue
                g = games[i]
                c = ctx.get((s, g["week"], g["team"]))
                if c is None:
                    continue
                hist = games[:i]
                pos = g["pos"]
                dvp = dvp_prior(allowed, g["week"], g["team"], pos, league_avg_pos[(s, pos)])
                dvp4 = dvp_roll4(allowed, g["week"], g["team"], pos, league_avg_pos[(s, pos)])
                dvp_p = dvp_prior(allowed_pass, g["week"], g["team"], pos, league_avg_pass[(s, pos)])
                dvp_r = dvp_prior(allowed_rush, g["week"], g["team"], pos, league_avg_rush[(s, pos)])
                dvp_p4 = dvp_roll4(allowed_pass, g["week"], g["team"], pos, league_avg_pass[(s, pos)])
                depth_row = bundle["depth"].get((g["week"], gsis))
                inj_hist = [bundle["inj"].get((h["week"], gsis), "") for h in hist]
                ngs_hist = []
                for h in hist:
                    nd = bundle["ngs"].get((h["week"], gsis))
                    if nd and nd.get("_mask"):
                        ngs_hist.append(nd)
                pfr_id = meta.get(gsis, {}).get("pfr", "")
                pfr_hist = []
                for h in hist:
                    pd = bundle["pfr"].get((h["week"], pfr_id))
                    if pd and pd.get("_mask"):
                        pfr_hist.append(pd)
                ep_hist = [ep_idx[(gsis, s, h["week"])]
                           for h in hist if (gsis, s, h["week"]) in ep_idx]
                rz_hist = [rz_idx[(gsis, s, h["week"])]
                           for h in hist if (gsis, s, h["week"]) in rz_idx]
                pw = hist[-1]["week"]
                pteam = hist[-1]["team"]
                committee_hhi = hhi_map.get((pw, pteam), 0.0)
                qb_ep = qb_ep_map.get((pw, pteam), 0.0)
                team_pass = team_pass_map.get((pw, pteam), 0.55)
                wr1 = wr1_map.get((pw, pteam), 0.0)

                row_v, row_m = assemble_row(
                    hist, g, c, meta.get(gsis, {}), pos, dvp, dvp4, dvp_p, dvp_r, dvp_p4,
                    depth_row, inj_hist, ngs_hist, pfr_hist, ep_hist, rz_hist, qb_ep, team_pass,
                    committee_hhi, wr1, ped_by_gsis.get(gsis), prior_sp.get(gsis, 0.0), s,
                )
                X.append(row_v)
                M.append(row_m)
                Y.append([g[t] for t in TARGET_NAMES])
                Y_USAGE.append([g.get(t, 0.0) for t in USAGE_RECON_NAMES])
                META.append({
                    "season": s, "week": g["week"], "gsis": gsis,
                    "name": g["name"], "pos": g["pos"], "team": g["team"],
                })

    up_season = last_season + 1
    byes = team_byes(up_season, offline)
    upcoming = build_upcoming(
        up_season, per_season[last_season], ctx, meta, league_avg_pos,
        league_avg_pass, league_avg_rush, last_season, byes, ep_idx, rz_idx, ped_by_gsis,
    )

    Xa = np.array(X, dtype=np.float32)
    Ma = np.array(M, dtype=np.float32)
    Ya = np.array(Y, dtype=np.float32)
    Ua = np.array(Y_USAGE, dtype=np.float32)

    DATA.mkdir(parents=True, exist_ok=True)
    manifest = {
        "built": time.strftime("%Y-%m-%d"),
        "features": FEATURE_NAMES,
        "families": FAMILY_OF,
        "family_lists": FAMILIES,
        "targets": TARGET_NAMES,
        "usage_recon": USAGE_RECON_NAMES,
        "n_rows": int(Xa.shape[0]),
        "n_features": int(Xa.shape[1]),
        "seasons": seasons,
        "coverage": {
            fam: float(Ma[:, [FEATURE_NAMES.index(c) for c in cols]].mean())
            for fam, cols in FAMILIES.items()
        },
    }
    np.savez_compressed(
        DATA / "train_matrix.npz",
        Z=Xa, mask=Ma, Y=Ya, Y_usage=Ua,
        season=np.array([m["season"] for m in META], dtype=np.int32),
        week=np.array([m["week"] for m in META], dtype=np.int32),
        gsis=np.array([m["gsis"] for m in META]),
        name=np.array([m["name"] for m in META]),
        pos=np.array([m["pos"] for m in META]),
        team=np.array([m["team"] for m in META]),
    )
    (DATA / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote train_matrix.npz X={Xa.shape} mask_mean={Ma.mean():.3f}")
    print("  coverage:", {k: round(v, 3) for k, v in manifest["coverage"].items()})

    return {
        "X": Xa.astype(np.float64), "M": Ma.astype(np.float64),
        "Y": Ya.astype(np.float64), "Y_usage": Ua.astype(np.float64),
        "meta": META, "feature_names": FEATURE_NAMES, "target_names": TARGET_NAMES,
        "families": FAMILIES, "family_of": FAMILY_OF, "manifest": manifest,
        "seasons": seasons, "upcoming": upcoming, "up_season": up_season,
        "byes": byes, "usage_recon_names": USAGE_RECON_NAMES,
    }


def build_upcoming(up_season, last_bundle, ctx, meta, league_avg_pos,
                   league_avg_pass, league_avg_rush, last_season,
                   byes, ep_idx, rz_idx, ped_by_gsis):
    bp = last_bundle["bp"]
    weeks = [w for (ssn, w, _t) in ctx if ssn == up_season]
    if not weeks:
        return {"season": up_season, "week": None, "rows": np.zeros((0, len(FEATURE_NAMES))),
                "masks": np.zeros((0, len(FEATURE_NAMES))), "meta": []}
    week = min(weeks)
    rows, masks, rmeta = [], [], []
    for gsis, games in bp.items():
        if len(games) < 4:
            continue
        m = meta.get(gsis, {})
        team = m.get("latest_team") or games[-1]["team"]
        c = ctx.get((up_season, week, team))
        if c is None:
            team = games[-1]["team"]
            c = ctx.get((up_season, week, team))
        if c is None:
            continue
        pos = games[-1]["pos"]
        dvp = league_avg_pos.get((last_season, pos), 12.0)
        dvp_p = league_avg_pass.get((last_season, pos), 8.0)
        dvp_r = league_avg_rush.get((last_season, pos), 4.0)
        depth_row = last_bundle["depth"].get((games[-1]["week"], gsis))
        ngs_hist = [last_bundle["ngs"][k] for k in last_bundle["ngs"]
                    if k[1] == gsis and last_bundle["ngs"][k].get("_mask")]
        pfr_id = m.get("pfr", "")
        pfr_hist = [last_bundle["pfr"][k] for k in last_bundle["pfr"]
                    if k[1] == pfr_id and last_bundle["pfr"][k].get("_mask")]
        ep_hist = [ep_idx[(gsis, last_season, g["week"])]
                   for g in games if (gsis, last_season, g["week"]) in ep_idx]
        rz_hist = [rz_idx[(gsis, last_season, g["week"])]
                   for g in games if (gsis, last_season, g["week"]) in rz_idx]
        inj_hist = [last_bundle["inj"].get((g["week"], gsis), "") for g in games]
        # WR1 share: max WR target_share on same team in last week of hist
        wr1 = 0.0
        last_wk, last_tm = games[-1]["week"], games[-1]["team"]
        for _gid, gs in bp.items():
            for g0 in gs:
                if g0["week"] == last_wk and g0["team"] == last_tm and g0["pos"] == "WR":
                    wr1 = max(wr1, g0.get("target_share", 0.0))
        row_v, row_m = assemble_row(
            games, games[-1], c, m, pos, dvp, dvp, dvp_p, dvp_r, dvp_p,
            depth_row, inj_hist, ngs_hist, pfr_hist, ep_hist, rz_hist, 0.0, 0.55, 0.0, wr1,
            ped_by_gsis.get(gsis), last_bundle["ppg"].get(gsis, 0.0), last_season,
        )
        rows.append(row_v)
        masks.append(row_m)
        moved = bool(m.get("latest_team") and m["latest_team"] != games[-1]["team"])
        rmeta.append({
            "season": up_season, "week": week, "gsis": gsis,
            "name": games[-1]["name"], "pos": pos, "team": team,
            "prev_team": games[-1]["team"], "moved": moved,
            "bye": byes.get(team), "status": m.get("status", ""),
            "opp": c["opp"], "headshot": games[-1]["headshot"],
            "conditions": {
                "roof": c["roof"], "temp": c["temp"], "wind": c["wind"],
                "is_home": c["is_home"], "team_implied": round(c["team_implied"], 1),
                "total": c["total_line"], "spread_team": c["spread_team"],
                "primetime": c["is_primetime"], "gameday": c["gday"],
            },
        })
    return {
        "season": up_season, "week": week,
        "rows": np.array(rows, dtype=np.float64),
        "masks": np.array(masks, dtype=np.float64),
        "meta": rmeta,
    }


def main() -> int:
    offline = "--offline" in sys.argv
    d = build(offline=offline)
    assert d["X"].shape[0] == len(d["meta"]) == d["Y"].shape[0]
    assert d["X"].shape[1] == len(d["feature_names"])
    assert d["M"].shape == d["X"].shape
    assert not np.isnan(d["X"]).any(), "NaNs in feature matrix"
    up = d["upcoming"]
    print(f"upcoming: {up['season']} week {up['week']}: {len(up['meta'])} players")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
