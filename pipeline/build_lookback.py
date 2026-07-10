"""Lookback league engine — grade drafts + write blog-style narratives for every
prior season, and seed them as a database the realtime version can extend.

For each season 2016-2025 it builds a full 12-team best-ball fantasy season from
REAL NFL results and tells its story:

  DRAFT   VOR-ordered snake draft (order = each player's value-over-replacement
          from the PRIOR season, position-capped) -> 12 rosters. Each team's
          draft is GRADED A+..F on the actual season value they captured, with
          their best steal and biggest bust called out.
  SEASON  best-ball weekly scores (objective: no start/sit luck) -> round-robin
          schedule -> standings -> a 6-team playoff -> a champion.
  STORY   templated blog prose: a draft recap, a one-line recap for every week,
          and a season recap (champion, MVP, bust of the year, draft-grade vs
          finish). Source-agnostic: `grade_drafts` / narrative fns take picks +
          results, so a REAL league's draft (from ESPN/Sleeper history) grades
          through the identical code.

Output: assets/lookback_seasons.json — the seeded DB.
Run:  python pipeline/build_lookback.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import nfl_data as nfl
from nfl_data import num

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "lookback_seasons.json"

SKILL = ("QB", "RB", "WR", "TE")
KDST = ("K", "DST")
ALL_POS = SKILL + KDST
N_TEAMS = 12
ROSTER = 15          # 14 skill + room for K/DST
LINEUP = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DST": 1}
FLEX_ELIG = ("RB", "WR", "TE")
POS_CAP = {"QB": 2, "RB": 6, "WR": 7, "TE": 3, "K": 1, "DST": 1}
# replacement rank per position in a 12-team league (starter demand + a bit)
REPL_RANK = {"QB": 12, "RB": 30, "WR": 36, "TE": 12, "K": 12, "DST": 12}
REG_WEEKS = 14
PLAYOFF_WEEKS = (15, 16, 17)

MANAGERS = ["The Analysts", "Gridiron Gang", "Cache Money", "Regression Kings",
            "Zero RB Zealots", "Waiver Warlocks", "Ceiling Chasers", "Floor Generals",
            "Model Behavior", "Vector Victors", "Dynasty Warriors", "Bye Week Bandits"]

GRADE_TIERS = [(.92, "A+"), (.84, "A"), (.75, "A-"), (.66, "B+"), (.55, "B"),
               (.45, "B-"), (.36, "C+"), (.27, "C"), (.18, "C-"), (.09, "D"), (0, "F")]


def letter(pct):
    for t, g in GRADE_TIERS:
        if pct >= t:
            return g
    return "F"


# ---------------------------------------------------------------------------
# load real weekly production per season
# ---------------------------------------------------------------------------

def load_season(season):
    players = {}
    for r in nfl.weekly_stats(season):
        if r.get("season_type") != "REG":
            continue
        pos = (r.get("position") or "").strip()
        if pos not in SKILL:
            continue
        g = r.get("player_id")
        p = players.setdefault(g, {"name": r.get("player_display_name") or r.get("player_name"),
                                   "pos": pos, "team": r.get("team", ""),
                                   "headshot": r.get("headshot_url", ""), "weeks": {}})
        p["weeks"][int(num(r, "week"))] = num(r, "fantasy_points_ppr")
        p["team"] = r.get("team", p["team"])
    for p in players.values():
        wk = p["weeks"]
        p["games"] = len(wk)
        p["total"] = round(sum(wk.values()), 1)
        p["ppg"] = p["total"] / max(1, p["games"])
    # Merge K/DST season rates from kdst.json (no weekly series — flat ppg every week).
    kdst_path = ROOT / "assets" / "kdst.json"
    if kdst_path.exists():
        kdst = json.loads(kdst_path.read_text(encoding="utf-8"))
        for arr, pos in ((kdst.get("kickers") or [], "K"), (kdst.get("dst") or [], "DST")):
            for row in arr:
                hist = row.get("history") or {}
                ppg = hist.get(season, hist.get(str(season)))
                if ppg is None:
                    continue
                key = f"{pos}:{row.get('key') or row.get('name')}"
                weeks = {w: float(ppg) for w in range(1, 19)}
                players[key] = {
                    "name": row["name"], "pos": pos, "team": row.get("team", ""),
                    "headshot": "", "weeks": weeks, "games": 16,
                    "total": round(float(ppg) * 16, 1), "ppg": float(ppg),
                }
    return players


def replacement_ppg(players, season_min_games=4):
    out = {}
    for pos in ALL_POS:
        ppgs = sorted((p["ppg"] for p in players.values()
                       if p["pos"] == pos and p["games"] >= season_min_games), reverse=True)
        r = REPL_RANK[pos]
        out[pos] = ppgs[r] if len(ppgs) > r else (ppgs[-1] if ppgs else 0.0)
    return out


# ---------------------------------------------------------------------------
# draft (order by prior-season VOR, position-capped snake)
# ---------------------------------------------------------------------------

def mock_draft(season, players, prior_vor, seed):
    # draftable = players with a positive prior-season expectation, best first
    board = sorted([g for g in players if g in prior_vor],
                   key=lambda g: prior_vor[g], reverse=True)
    order = list(range(N_TEAMS))
    rng = _rng(seed)
    rng.shuffle(order)                       # this season's draft-slot assignment
    rosters = {i: [] for i in range(N_TEAMS)}
    counts = {i: {p: 0 for p in ALL_POS} for i in range(N_TEAMS)}
    pick_no = {i: {} for i in range(N_TEAMS)}
    bi = 0
    for rnd in range(ROSTER):
        seq = order if rnd % 2 == 0 else order[::-1]
        for slot, team in enumerate(seq):
            # best available this team can still roster (position cap + basic needs)
            while bi < len(board):
                g = board[bi]
                pos = players[g]["pos"]
                need_ok = counts[team][pos] < POS_CAP.get(pos, 1)
                if need_ok:
                    rosters[team].append(g)
                    counts[team][pos] += 1
                    pick_no[team][g] = rnd * N_TEAMS + slot + 1
                    bi += 1
                    break
                bi += 1
            # if board exhausted, leave short (rare)
        # reset scan pointer each round so capped-out picks aren't skipped forever
        board = [g for g in board if all(g not in rosters[t] for t in range(N_TEAMS))]
        bi = 0
    return rosters, pick_no, order


def _rng(seed):
    import random
    r = random.Random(seed)
    return r


# ---------------------------------------------------------------------------
# best-ball weekly scoring, schedule, standings, playoffs
# ---------------------------------------------------------------------------

def best_ball(roster, week, players):
    by_pos = {p: [] for p in ALL_POS}
    for g in roster:
        pts = players[g]["weeks"].get(week)
        if pts is not None:
            by_pos[players[g]["pos"]].append((pts, g))
    for pos in by_pos:
        by_pos[pos].sort(reverse=True)
    used, total, starters = set(), 0.0, []
    for pos, n in (("QB", 1), ("RB", 2), ("WR", 2), ("TE", 1), ("K", 1), ("DST", 1)):
        for pts, g in by_pos[pos][:n]:
            used.add(g); total += pts; starters.append((pts, g))
    # FLEX = best remaining RB/WR/TE
    flex = sorted([(pts, g) for pos in FLEX_ELIG for pts, g in by_pos[pos] if g not in used],
                  reverse=True)
    if flex:
        pts, g = flex[0]; used.add(g); total += pts; starters.append((pts, g))
    return round(total, 1), starters


def round_robin(n, weeks):
    """Deterministic rotating pairings for `weeks` weeks over n teams."""
    teams = list(range(n))
    sched = []
    arr = teams[:]
    for w in range(weeks):
        pairs = [(arr[i], arr[n - 1 - i]) for i in range(n // 2)]
        sched.append(pairs)
        arr = [arr[0]] + [arr[-1]] + arr[1:-1]   # rotate
    return sched


def run_season(rosters, players):
    weekly_scores = {i: {} for i in range(N_TEAMS)}
    player_of_week = {}
    for w in range(1, max(REG_WEEKS, PLAYOFF_WEEKS[-1]) + 1):
        best = (-1, None, None)  # (pts, team, gsis)
        for i in range(N_TEAMS):
            total, starters = best_ball(rosters[i], w, players)
            weekly_scores[i][w] = total
            for pts, g in starters:
                if pts > best[0]:
                    best = (pts, i, g)
        if best[1] is not None:
            player_of_week[w] = {"team": best[1], "gsis": best[2], "pts": round(best[0], 1)}
    # regular-season schedule + standings
    sched = round_robin(N_TEAMS, REG_WEEKS)
    wins = {i: 0 for i in range(N_TEAMS)}
    pf = {i: 0.0 for i in range(N_TEAMS)}
    weekly_results = []
    for w, pairs in enumerate(sched, start=1):
        wk = []
        for a, b in pairs:
            sa, sb = weekly_scores[a][w], weekly_scores[b][w]
            if sa >= sb:
                wins[a] += 1
            else:
                wins[b] += 1
            pf[a] += sa; pf[b] += sb
            wk.append((a, b, sa, sb))
        weekly_results.append(wk)
    standings = sorted(range(N_TEAMS), key=lambda i: (wins[i], pf[i]), reverse=True)
    # 6-team playoff: seeds 1-2 bye wk15, then 15=(3v6,4v5),16=semis,17=final
    champion = playoffs(standings, weekly_scores)
    return {"weekly_scores": weekly_scores, "wins": wins, "pf": pf,
            "standings": standings, "weekly_results": weekly_results,
            "player_of_week": player_of_week, "champion": champion}


def playoffs(standings, weekly_scores):
    seeds = standings[:6]
    w15 = [(seeds[2], seeds[5]), (seeds[3], seeds[4])]
    winners = [a if weekly_scores[a][15] >= weekly_scores[b][15] else b for a, b in w15]
    semis = [(seeds[0], winners[1]), (seeds[1], winners[0])]
    finalists = [a if weekly_scores[a][16] >= weekly_scores[b][16] else b for a, b in semis]
    a, b = finalists
    return a if weekly_scores[a][17] >= weekly_scores[b][17] else b


# ---------------------------------------------------------------------------
# grading + narratives
# ---------------------------------------------------------------------------

def grade_drafts(rosters, pick_no, players, actual_vor, prior_vor):
    """Value captured = sum of positive actual VOR across a team's draft. Grade
    on percentile within the league. Also each team's best steal + biggest bust."""
    value = {}
    steal = {}
    bust = {}
    for i, roster in rosters.items():
        value[i] = round(sum(max(0.0, actual_vor.get(g, 0.0)) for g in roster), 1)
        # steal: latest pick with the most actual value; bust: earliest pick with worst
        best_s = max(roster, key=lambda g: actual_vor.get(g, -9) - prior_vor.get(g, 0) * 0.2, default=None)
        worst_b = min(roster, key=lambda g: actual_vor.get(g, 0) - _draft_cost(pick_no[i].get(g, 200)), default=None)
        steal[i] = best_s
        bust[i] = worst_b
    ranked = sorted(rosters, key=lambda i: value[i], reverse=True)
    grade = {}
    for rank, i in enumerate(ranked):
        grade[i] = letter(1 - rank / (N_TEAMS - 1))
    return value, grade, steal, bust, ranked


def _draft_cost(pick):
    """Rough VOR a pick is 'expected' to return, by draft slot (early = more)."""
    return max(0.0, 14.0 - 0.09 * pick)


def fmt_pts(x):
    return f"{x:.1f}"


def draft_narrative(season, rosters, pick_no, players, grade, value, steal, bust, ranked, actual_vor):
    best, worst = ranked[0], ranked[-1]
    # league-wide steal of the draft: highest actual VOR taken latest
    all_picks = [(i, g) for i in rosters for g in rosters[i]]
    steal_league = max(all_picks, key=lambda ig: actual_vor.get(ig[1], -9) if pick_no[ig[0]].get(ig[1], 0) > N_TEAMS * 4 else -9, default=None)
    bust_league = min(all_picks, key=lambda ig: actual_vor.get(ig[1], 0) if pick_no[ig[0]].get(ig[1], 200) <= N_TEAMS * 2 else 99, default=None)
    lines = []
    lines.append(f"**The {season} Draft, graded.** {MANAGERS[best]} ran away with the class — "
                 f"an {grade[best]} draft that banked {fmt_pts(value[best])} points of value over replacement, "
                 f"anchored by {players[steal[best]]['name']}. At the other end, {MANAGERS[worst]} "
                 f"earned a {grade[worst]}: {players[bust[worst]]['name']} never returned the draft capital.")
    if steal_league:
        i, g = steal_league
        lines.append(f"Steal of the draft: **{players[g]['name']}** ({players[g]['pos']}), "
                     f"scooped at pick {pick_no[i].get(g)} by {MANAGERS[i]} and worth "
                     f"{fmt_pts(actual_vor.get(g, 0))} VOR.")
    if bust_league:
        i, g = bust_league
        lines.append(f"Biggest reach: **{players[g]['name']}** went at pick {pick_no[i].get(g)} to "
                     f"{MANAGERS[i]} and returned just {fmt_pts(actual_vor.get(g, 0))} VOR.")
    return " ".join(lines)


def week_narrative(week, res, players, pow_):
    wk = res["weekly_results"][week - 1] if week <= REG_WEEKS else None
    parts = []
    if wk:
        hi = max(wk, key=lambda m: max(m[2], m[3]))
        hi_team, hi_pts = (hi[0], hi[2]) if hi[2] >= hi[3] else (hi[1], hi[3])
        blow = max(wk, key=lambda m: abs(m[2] - m[3]))
        parts.append(f"{MANAGERS[hi_team]} hung a league-high {fmt_pts(hi_pts)}")
        margin = abs(blow[2] - blow[3])
        if margin > 35:
            wnr, lsr = (blow[0], blow[1]) if blow[2] >= blow[3] else (blow[1], blow[0])
            parts.append(f"{MANAGERS[wnr]} bludgeoned {MANAGERS[lsr]} by {fmt_pts(margin)}")
    p = pow_.get(week)
    if p:
        parts.append(f"player of the week: {players[p['gsis']]['name']} ({fmt_pts(p['pts'])})")
    return f"**Week {week}.** " + "; ".join(parts) + "." if parts else f"**Week {week}.**"


def season_narrative(season, res, rosters, players, grade, value, ranked):
    champ = res["champion"]
    standings = res["standings"]
    # MVP = most total best-ball starter points across the season among rostered
    totals = {}
    for i, roster in rosters.items():
        for g in roster:
            totals[g] = players[g]["total"]
    mvp = max(totals, key=totals.get)
    runner = standings[1]
    champ_grade = grade[champ]
    lines = []
    lines.append(f"**{season}: {MANAGERS[champ]} are your champions.** They finished the regular "
                 f"season {res['wins'][champ]}-{REG_WEEKS - res['wins'][champ]} "
                 f"({fmt_pts(res['pf'][champ])} PF) and closed it out over {MANAGERS[runner]}.")
    lines.append(f"League MVP: **{players[mvp]['name']}** ({players[mvp]['pos']}), "
                 f"{fmt_pts(players[mvp]['total'])} total points.")
    draft_rank = ranked.index(champ) + 1
    if draft_rank <= 3:
        lines.append(f"The title traced straight to the draft — {MANAGERS[champ]} had a top-{draft_rank} "
                     f"class ({champ_grade}).")
    else:
        lines.append(f"Proof the draft isn't destiny: {MANAGERS[champ]}'s class only graded {champ_grade} "
                     f"(#{draft_rank}), but the trophy is the trophy.")
    return " ".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_season(season, players, prior_players):
    if not players or not prior_players:
        return None
    repl = replacement_ppg(players)
    actual_vor = {g: round(p["ppg"] - repl[p["pos"]], 2) for g, p in players.items() if p["games"] >= 3}
    # draft order uses each player's prior-season value over replacement
    prior_vor = {}
    if prior_players:
        prepl = replacement_ppg(prior_players)
        for g, p in prior_players.items():
            if p["games"] >= 6:
                prior_vor[g] = p["ppg"] - prepl[p["pos"]]
    if not prior_vor:
        return None
    rosters, pick_no, order = mock_draft(season, players, prior_vor, seed=f"vg-draft-{season}")
    res = run_season(rosters, players)
    value, grade, steal, bust, ranked = grade_drafts(rosters, pick_no, players, actual_vor, prior_vor)
    pick_of = {g: pick_no[i][g] for i in range(N_TEAMS) for g in rosters[i]}   # global overall pick

    teams = []
    for i in range(N_TEAMS):
        picks = sorted(rosters[i], key=lambda g: pick_no[i].get(g, 999))
        teams.append({
            "name": MANAGERS[i],
            "grade": grade[i], "draft_value": value[i],
            "record": f"{res['wins'][i]}-{REG_WEEKS - res['wins'][i]}",
            "wins": res["wins"][i], "points_for": round(res["pf"][i], 1),
            "seed": res["standings"].index(i) + 1,
            "champion": i == res["champion"],
            "best_pick": _player_brief(players, steal[i], pick_no[i], actual_vor),
            "worst_pick": _player_brief(players, bust[i], pick_no[i], actual_vor),
            "roast": team_roast(i, MANAGERS[i], rosters, pick_of, players, actual_vor, season),
            "roster": [_player_brief(players, g, pick_no[i], actual_vor) for g in picks[:8]],
        })
    teams.sort(key=lambda t: (-t["wins"], -t["points_for"]))

    weeks = [{"week": w, "text": week_narrative(w, res, players, res["player_of_week"])}
             for w in range(1, REG_WEEKS + 1)]
    narr = {
        "draft": draft_narrative(season, rosters, pick_no, players, grade, value, steal, bust, ranked, actual_vor),
        "season": season_narrative(season, res, rosters, players, grade, value, ranked),
        "weeks": weeks,
    }
    return {"season": season, "teams": teams, "champion": MANAGERS[res["champion"]],
            "narratives": narr}


# ---------------------------------------------------------------------------
# redraft + trash talk (call out the league's big hits and big misses)
# ---------------------------------------------------------------------------

MISS_OPENERS = [
    "{tm} used pick {pk} on {name} and got a cool {vor} VOR for their trouble",
    "Somebody tell {tm} that pick {pk} — {name} — returned {vor} VOR",
    "{tm} slammed the table at pick {pk} for {name}. It returned {vor} VOR",
    "Pick {pk}, {tm} on the clock, hard-forces {name}: {vor} VOR",
]
HIT_TAGS = [
    "credit though — {name} at {pk} was grand larceny ({vor} VOR)",
    "at least {name} at pick {pk} was a heist ({vor} VOR)",
    "saving grace: {name} ({pk}) printed {vor} VOR",
]


def redraft_pick(bust_g, pick_of, players, actual_vor):
    """The best player who was still on the board when `bust_g` was taken."""
    P = pick_of.get(bust_g, 999)
    cands = [(actual_vor.get(g, 0.0), g) for g, pk in pick_of.items()
             if pk > P and actual_vor.get(g, 0.0) > 0]
    return max(cands)[1] if cands else None


def team_roast(i, name, rosters, pick_of, players, actual_vor, seed):
    import random
    rng = random.Random(f"{seed}-{i}")
    roster = rosters[i]
    # miss = drafted earliest-and-worst; hit = latest-and-best
    miss = min(roster, key=lambda g: actual_vor.get(g, 0) - _draft_cost(pick_of.get(g, 200)), default=None)
    hit = max(roster, key=lambda g: actual_vor.get(g, -9) - pick_of.get(g, 0) * -0.02, default=None)
    rd = redraft_pick(miss, pick_of, players, actual_vor) if miss else None
    parts = []
    if miss:
        parts.append(rng.choice(MISS_OPENERS).format(
            tm=name, pk=pick_of.get(miss), name=players[miss]["name"], vor=f"{actual_vor.get(miss, 0):+.1f}"))
        if rd:
            parts.append(f"{players[rd]['name']} (went {pick_of.get(rd)}, {actual_vor.get(rd, 0):+.1f} VOR) was right there")
    if hit and hit != miss:
        parts.append(rng.choice(HIT_TAGS).format(
            name=players[hit]["name"], pk=pick_of.get(hit), vor=f"{actual_vor.get(hit, 0):+.1f}"))
    return {
        "line": " — ".join(parts) + ".",
        "miss": _player_brief(players, miss, {miss: pick_of.get(miss)} if miss else {}, actual_vor),
        "hit": _player_brief(players, hit, {hit: pick_of.get(hit)} if hit else {}, actual_vor),
        "redraft": _player_brief(players, rd, {rd: pick_of.get(rd)} if rd else {}, actual_vor),
    }


def _player_brief(players, g, pick_no, actual_vor):
    if not g or g not in players:
        return None
    p = players[g]
    return {"name": p["name"], "pos": p["pos"], "team": p["team"], "headshot": p["headshot"],
            "pick": pick_no.get(g), "ppg": round(p["ppg"], 1), "total": p["total"],
            "vor": actual_vor.get(g, 0.0)}


HIST_START = 1999   # every prior season nflverse publishes

def main():
    t0 = time.time()
    latest = nfl.latest_stats_season()
    seasons = [y for y in range(HIST_START, latest + 1) if nfl.weekly_stats(y)]
    print(f"seeding lookback league for {seasons[0]}-{seasons[-1]} ...")
    out = []
    prior_players = load_season(seasons[0] - 1)   # 2015 seeds the 2016 draft order
    for s in seasons:
        cur_players = load_season(s)
        rec = build_season(s, cur_players, prior_players)
        if rec:
            out.append(rec)
            best = max(rec["teams"], key=lambda t: t["draft_value"])
            print(f"  {s}: champ {rec['champion']}; best draft {best['name']} ({best['grade']})")
        prior_players = cur_players
    OUT.write_text(json.dumps({
        "built": time.strftime("%Y-%m-%d"), "n_teams": N_TEAMS, "format": "best-ball PPR",
        "note": "Seeded from real NFL results via VOR-ordered mock drafts; the grading + "
                "narrative engine is source-agnostic and accepts a real league's draft history.",
        "seasons": out,
    }, separators=(",", ":")), encoding="utf-8")
    print(f"\nwrote {OUT.name}: {len(out)} seasons ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
