"""Depth-chart schema gate for build_features.index_depth.

nflverse re-sourced depth_charts_2025.csv from ESPN: timestamped dumps
(dt, team, gsis_id, pos_abb, pos_slot, pos_rank), no week column. The
week-keyed files through 2024 (season, club_code, week, depth_team,
formation, gsis_id, depth_position) are unchanged. These tests pin, on
schema-only fixtures (fake ids, fake teams, fake season):

  * header detection: weekly / asof / unknown, incl. the two real headers;
  * the weekly path is the original parser byte-for-byte -- a frozen copy of
    the pre-dispatch function is the oracle;
  * the as-of path joins on the exact gsis_id key, picks the latest dump
    strictly before the gameday inside the staleness bound, maps slot order
    to depth level, and DROPS (never guesses) ambiguous keys, counting them.

Run:  pytest -p no:cacheprovider pipeline/test_depth_chart_schema.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_features as bf  # noqa: E402
from nfl_data import num  # noqa: E402

SEASON = 2099
W1, W2, W3, W4, W5 = "00-0000001", "00-0000002", "00-0000003", "00-0000004", "00-0000005"
R1, R2 = "00-0000011", "00-0000012"
T1, T2 = "00-0000021", "00-0000022"
X1 = "00-0000031"      # listed by two teams in the same week
Q1 = "00-0000041"
D1 = "00-0000051"
C1 = "00-0000061"
NONSTD = "ABC123456"   # not a gsis id (the 2025 file carries a few elias ids here)

REAL_HEADER_2025 = ["dt", "team", "player_name", "espn_id", "gsis_id", "pos_grp_id",
                    "pos_grp", "pos_id", "pos_name", "pos_abb", "pos_slot", "pos_rank"]
REAL_HEADER_2024 = ["season", "club_code", "week", "game_type", "depth_team", "last_name",
                    "first_name", "football_name", "formation", "gsis_id", "jersey_number",
                    "position", "elias_id", "depth_position", "full_name"]


# ---------------------------------------------------------------------------
# Frozen copy of index_depth as it was before the dispatcher (commit a6fc91e).
# The weekly path must reproduce it exactly.
# ---------------------------------------------------------------------------

def _reference_weekly(rows) -> dict:
    by_team_week_pos: dict[tuple, list] = {}
    out = {}
    for r in rows:
        gsis = (r.get("gsis_id") or "").strip()
        if not gsis:
            continue
        form = (r.get("formation") or "").lower()
        if form and form not in ("offense", "o", ""):
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _games():
    return [
        {"season": "2099", "week": "1", "gameday": "2099-09-07", "home_team": "AAA", "away_team": "BBB"},
        {"season": "2099", "week": "2", "gameday": "2099-09-14", "home_team": "BBB", "away_team": "AAA"},
        {"season": "2099", "week": "2", "gameday": "2099-09-14", "home_team": "CCC", "away_team": "DDD"},
        # another season: must be ignored entirely
        {"season": "2098", "week": "1", "gameday": "2098-09-08", "home_team": "AAA", "away_team": "BBB"},
    ]


def _asof(dt, team, gsis, pos, slot, rank):
    return {"dt": dt, "team": team, "player_name": "fixture", "espn_id": "0",
            "gsis_id": gsis, "pos_grp_id": "21", "pos_grp": "3WR 1TE", "pos_id": "0",
            "pos_name": pos, "pos_abb": pos, "pos_slot": str(slot), "pos_rank": str(rank)}


def _asof_rows():
    d05 = "2099-09-05T07:00:00Z"   # AAA: older than d06 -> not chosen for week 1
    d06 = "2099-09-06T07:00:00Z"   # AAA/BBB/CCC: chosen for week 1 (CCC: 8 days before week 2 -> stale)
    d07 = "2099-09-07T07:00:00Z"   # AAA: ON gameday -> excluded for week 1; 7 days before week 2 -> chosen
    d13 = "2099-09-13T07:00:00Z"   # BBB: chosen for week 2
    return [
        # outside the season window entirely: ignored
        _asof("2099-08-01T07:00:00Z", "AAA", W1, "WR", 1, 1),
        # AAA older dump -- if this were chosen, W2 would be WR1 and W1 absent
        _asof(d05, "AAA", W2, "WR", 2, 1),
        _asof(d05, "AAA", R1, "RB", 11, 1),
        # AAA chosen dump for week 1 -- WR has three slots, rank is global
        _asof(d06, "AAA", W1, "WR", 1, 1),
        _asof(d06, "AAA", W2, "WR", 2, 2),
        _asof(d06, "AAA", W3, "WR", 8, 3),
        _asof(d06, "AAA", W4, "WR", 1, 4),
        _asof(d06, "AAA", W5, "WR", 2, 5),
        _asof(d06, "AAA", NONSTD, "WR", 8, 6),   # in the group, never joins
        _asof(d06, "AAA", "", "WR", 1, 7),        # empty id: skipped, counted
        _asof(d06, "AAA", R1, "RB", 11, 1),
        _asof(d06, "AAA", R2, "RB", 11, 2),
        _asof(d06, "AAA", X1, "RB", 11, 3),
        _asof(d06, "AAA", T1, "TE", 10, 1),
        _asof(d06, "AAA", T2, "TE", 10, 2),
        _asof(d06, "AAA", T1, "FB", 12, 1),       # T1 at two slots -> ambiguous
        _asof(d06, "AAA", W1, "PR", 1, 1),        # special-teams listing: not offense, no ambiguity
        _asof(d06, "AAA", W1, "LCB", 1, 1),       # defense listing: not offense, no ambiguity
        _asof("2099-09-06", "AAA", W1, "WR", 1, 1),          # malformed dt: counted, skipped
        _asof(d06, "AAA", "00-0000099", "WR", 1, "x"),       # malformed rank: counted, skipped
        # AAA dump on gameday: must not feed week 1; feeds week 2 (exactly 7 days before)
        _asof(d07, "AAA", W5, "WR", 1, 1),
        _asof(d07, "AAA", R2, "RB", 11, 1),
        # BBB
        _asof(d06, "BBB", Q1, "QB", 9, 1),
        _asof(d06, "BBB", X1, "RB", 11, 1),       # X1 also on AAA's snapshot -> ambiguous
        _asof(d13, "BBB", Q1, "QB", 9, 1),
        # CCC: only dump is 8 days before its week-2 game -> no snapshot
        _asof(d06, "CCC", C1, "QB", 9, 1),
        # DDD: dump exactly 7 days before -> inside the bound
        _asof(d07, "DDD", D1, "QB", 9, 1),
    ]


def _weekly(week, team, depth_team, formation, gsis, depth_position):
    return {"season": str(SEASON), "club_code": team, "week": str(week), "game_type": "REG",
            "depth_team": str(depth_team), "last_name": "x", "first_name": "y",
            "football_name": "y", "formation": formation, "gsis_id": gsis,
            "jersey_number": "0", "position": depth_position, "elias_id": "",
            "depth_position": depth_position, "full_name": "y x"}


def _weekly_rows():
    return [
        _weekly(1, "AAA", 1, "Offense", W1, "WR"),
        _weekly(1, "AAA", 1, "Offense", W2, "WR"),
        _weekly(1, "AAA", 2, "Offense", W3, "WR"),
        _weekly(1, "AAA", 1, "Offense", R1, "RB"),
        _weekly(1, "AAA", 1, "Defense", D1, "LCB"),        # skipped
        _weekly(1, "AAA", 1, "Special Teams", W1, "PR"),   # skipped
        _weekly(0, "AAA", 1, "Offense", W4, "WR"),         # week 0: skipped
        _weekly(1, "AAA", 1, "Offense", "", "WR"),         # empty id: skipped
        _weekly(1, "AAA", 2, "", T1, "TE"),                # empty formation: kept
        _weekly(1, "AAA", 1, "Offense", T1, "FB"),         # duplicate id: old path = last wins
        _weekly(2, "AAA", 3, "Offense", W1, "WR"),
    ]


def _patch(monkeypatch, rows, games=None):
    monkeypatch.setattr(bf.nfl, "depth_charts_iter", lambda year, offline=False: iter(rows))
    monkeypatch.setattr(bf.nfl, "games", lambda offline=False: games if games is not None else _games())
    bf.DEPTH_ASOF_STATS.clear()


# ---------------------------------------------------------------------------
# Header detection
# ---------------------------------------------------------------------------

def test_schema_detection():
    assert bf.depth_schema(REAL_HEADER_2024) == "weekly"
    assert bf.depth_schema(REAL_HEADER_2025) == "asof"
    assert bf.depth_schema(["dt", "team", "gsis_id"]) == "unknown"
    assert bf.depth_schema(["foo"]) == "unknown"
    # a file carrying both week and dt is not something we have seen: weekly wins
    assert bf.depth_schema(REAL_HEADER_2024 + ["dt"]) == "weekly"


def test_unknown_schema_masks(monkeypatch, capsys):
    _patch(monkeypatch, [{"foo": "1", "gsis_id": W1}])
    assert bf.index_depth(SEASON, True) == {}
    assert "unrecognised schema" in capsys.readouterr().out


def test_empty_file_masks(monkeypatch):
    _patch(monkeypatch, [])
    assert bf.index_depth(SEASON, True) == {}


# ---------------------------------------------------------------------------
# Weekly path: unchanged
# ---------------------------------------------------------------------------

def test_weekly_path_matches_frozen_reference(monkeypatch):
    rows = _weekly_rows()
    _patch(monkeypatch, rows)
    got = bf.index_depth(SEASON, True)
    assert got == _reference_weekly(_weekly_rows())
    assert got == bf._index_depth_weekly(_weekly_rows())
    # hand-checked expectations of the original behaviour
    assert got[(1, W1)] == {"depth_rank": 1.0, "is_starter": 1.0, "depth_ahead": 0.0}
    assert got[(1, W2)] == {"depth_rank": 1.0, "is_starter": 1.0, "depth_ahead": 1.0}
    assert got[(1, W3)] == {"depth_rank": 2.0, "is_starter": 0.0, "depth_ahead": 2.0}
    assert got[(1, T1)] == {"depth_rank": 1.0, "is_starter": 1.0, "depth_ahead": 0.0}  # last wins
    assert got[(2, W1)] == {"depth_rank": 3.0, "is_starter": 0.0, "depth_ahead": 0.0}
    assert (1, D1) not in got and (0, W4) not in got and (1, W4) not in got
    assert not bf.DEPTH_ASOF_STATS


# ---------------------------------------------------------------------------
# As-of path
# ---------------------------------------------------------------------------

def test_asof_exact_join_and_ambiguity_drop(monkeypatch):
    _patch(monkeypatch, _asof_rows())
    got = bf.index_depth(SEASON, True)
    st = bf.DEPTH_ASOF_STATS[SEASON]

    # latest dump strictly before the gameday: W1 exists only in the 09-06 dump
    assert got[(1, W1)] == {"depth_rank": 1.0, "is_starter": 1.0, "depth_ahead": 0.0}
    # three WR slots -> three level-1 WRs, ordered by (level, gsis) like the weekly path
    assert got[(1, W2)] == {"depth_rank": 1.0, "is_starter": 1.0, "depth_ahead": 1.0}
    assert got[(1, W3)] == {"depth_rank": 1.0, "is_starter": 1.0, "depth_ahead": 2.0}
    assert got[(1, W4)] == {"depth_rank": 2.0, "is_starter": 0.0, "depth_ahead": 3.0}
    assert got[(1, W5)] == {"depth_rank": 2.0, "is_starter": 0.0, "depth_ahead": 4.0}
    # single-slot position: rank is the level
    assert got[(1, R1)] == {"depth_rank": 1.0, "is_starter": 1.0, "depth_ahead": 0.0}
    assert got[(1, R2)] == {"depth_rank": 2.0, "is_starter": 0.0, "depth_ahead": 1.0}
    # ambiguous keys are absent (mask 0 downstream), but still count for others' depth_ahead
    assert (1, T1) not in got                     # TE and FB on the same snapshot
    assert (1, X1) not in got                     # on AAA's and BBB's snapshots
    assert got[(1, T2)] == {"depth_rank": 2.0, "is_starter": 0.0, "depth_ahead": 1.0}
    # ids that are not gsis ids never join; empty ids never enter
    assert (1, NONSTD) not in got and (1, "") not in got and (1, "00-0000099") not in got
    # special-teams / defense listings of an offense player do not make him ambiguous
    assert (1, W1) in got
    # BBB week 1
    assert got[(1, Q1)] == {"depth_rank": 1.0, "is_starter": 1.0, "depth_ahead": 0.0}

    # week 2: AAA's 09-07 dump is exactly 7 days before -> used; W1 is not on it
    assert got[(2, W5)] == {"depth_rank": 1.0, "is_starter": 1.0, "depth_ahead": 0.0}
    assert got[(2, R2)] == {"depth_rank": 1.0, "is_starter": 1.0, "depth_ahead": 0.0}
    assert (2, W1) not in got
    assert got[(2, Q1)] == {"depth_rank": 1.0, "is_starter": 1.0, "depth_ahead": 0.0}
    assert got[(2, D1)] == {"depth_rank": 1.0, "is_starter": 1.0, "depth_ahead": 0.0}
    # CCC's only dump is 8 days old -> no snapshot, nothing invented
    assert (2, C1) not in got

    # every emitted key has all three role fields
    assert all(set(v) == {"depth_rank", "is_starter", "depth_ahead"} for v in got.values())
    assert len(got) == 13

    # counters
    assert st["team_weeks_scheduled"] == 6
    assert st["team_weeks_joined"] == 5
    assert st["team_weeks_no_snapshot"] == 1
    assert st["dumps_used"] == 3   # distinct dump timestamps: 09-06, 09-07, 09-13
    assert st["keys_joined"] == 13
    assert st["keys_ambiguous_multi_slot"] == 1
    assert st["keys_ambiguous_multi_team"] == 1
    assert st["keys_ambiguous_rank_tie"] == 0
    assert st["keys_unjoinable_gsis"] == 1
    assert st["rows_empty_gsis"] == 1
    assert st["rows_bad_dt"] == 1
    assert st["rows_bad_rank"] == 1
    assert st["dropped_keys"] == [[1, T1], [1, X1]]


def test_asof_rank_tie_is_ambiguous(monkeypatch):
    d06 = "2099-09-06T07:00:00Z"
    rows = [
        _asof(d06, "AAA", W1, "WR", 1, 1),
        _asof(d06, "AAA", W2, "WR", 1, 1),    # same slot, same rank: which is first?
        _asof(d06, "AAA", W3, "WR", 2, 2),
    ]
    _patch(monkeypatch, rows)
    got = bf.index_depth(SEASON, True)
    st = bf.DEPTH_ASOF_STATS[SEASON]
    assert (1, W1) not in got and (1, W2) not in got
    assert got[(1, W3)]["depth_rank"] == 1.0
    assert st["keys_ambiguous_rank_tie"] == 2
    assert st["dropped_keys"] == [[1, W1], [1, W2]]


def test_asof_without_schedule_masks(monkeypatch, capsys):
    _patch(monkeypatch, _asof_rows(), games=[])
    assert bf.index_depth(SEASON, True) == {}
    assert "games.csv has no" in capsys.readouterr().out
    assert bf.DEPTH_ASOF_STATS[SEASON]["team_weeks_scheduled"] == 0


def test_asof_never_defaults_a_missing_player(monkeypatch):
    """A player with a stats row but no depth listing must simply be absent."""
    _patch(monkeypatch, _asof_rows())
    got = bf.index_depth(SEASON, True)
    assert (1, "00-0000777") not in got
    assert bf.DEPTH_ASOF_STATS[SEASON]["rows_total"] == len(_asof_rows())


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main(["-p", "no:cacheprovider", "-q", __file__]))
