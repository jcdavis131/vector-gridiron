"""Shared nflverse (+ ffopportunity) data access for Vector Gridiron.

Cached, resumable, stdlib. All feeds land under pipeline/cache/.

Feeds:
  weekly_stats / snaps / games / players / injuries  — baseline (v1)
  depth_charts(year)   — role / starter depth (week-keyed through 2024)
  ngs(stat, year)      — Next Gen Stats weekly (passing|receiving|rushing), 2016+
  pfr_adv(stat, year)  — PFR advanced weekly (pass|rec|rush), ~2018+
  ep_weekly(year)      — ffopportunity expected fantasy points (CC-BY-SA)
  draft_picks() / combine() — pedigree for rookies / aux head
"""

from __future__ import annotations

import csv
import gzip
import io
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "pipeline" / "cache"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) vector-gridiron-pipeline/3.0"

REL = "https://github.com/nflverse/nflverse-data/releases/download"
STATS_URL = REL + "/stats_player/stats_player_week_{year}.csv"
SNAPS_URL = REL + "/snap_counts/snap_counts_{year}.csv"
PLAYERS_URL = REL + "/players/players.csv"
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
INJURIES_URL = REL + "/injuries/injuries_{year}.csv"
DEPTH_URL = REL + "/depth_charts/depth_charts_{year}.csv"
NGS_URL = REL + "/nextgen_stats/ngs_{year}_{stat}.csv.gz"
PFR_ADV_URL = REL + "/pfr_advstats/advstats_week_{stat}_{year}.csv"
DRAFT_URL = REL + "/draft_picks/draft_picks.csv"
COMBINE_URL = REL + "/combine/combine.csv"
# ffopportunity weekly EP — cite "ffopportunity / ffverse (CC-BY-SA 4.0)"
EP_WEEKLY_URL = (
    "https://github.com/ffverse/ffopportunity/releases/download/"
    "latest-data/ep_weekly_{year}.csv"
)

FIRST_SEASON = 2016
# nflverse injury source died after 2024 — family must be masked for 2025+.
INJURIES_LAST_SEASON = 2024


# ---------------------------------------------------------------------------
# Cached fetch
# ---------------------------------------------------------------------------

def fetch_bytes(url: str, cache_name: str, offline: bool = False,
                max_age_days: float | None = None) -> bytes | None:
    p = CACHE / cache_name
    if p.exists():
        if max_age_days is not None and not offline:
            age = (time.time() - p.stat().st_mtime) / 86400
            if age > max_age_days:
                try:
                    p.unlink()
                except OSError:
                    pass
            else:
                return p.read_bytes()
        else:
            return p.read_bytes()
    if offline:
        print(f"  {cache_name}: not cached and --offline -- skipping")
        return None
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(4):
        try:
            raw = urllib.request.urlopen(req, timeout=180).read()
            CACHE.mkdir(parents=True, exist_ok=True)
            p.write_bytes(raw)
            time.sleep(0.2)
            return raw
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            code = getattr(e, "code", None)
            if code == 404:
                print(f"  {cache_name}: 404 (not published) -- skipping")
                return None
            wait = 2 ** attempt
            print(f"  {cache_name}: attempt {attempt + 1}/4 failed "
                  f"({type(e).__name__}: {e}); sleeping {wait}s")
            time.sleep(wait)
    print(f"  {cache_name}: EXHAUSTED retries -- skipping")
    return None


def fetch_text(url: str, cache_name: str, offline: bool = False,
               max_age_days: float | None = None) -> str | None:
    raw = fetch_bytes(url, cache_name, offline, max_age_days)
    if raw is None:
        return None
    if cache_name.endswith(".gz"):
        return gzip.decompress(raw).decode("utf-8", "replace")
    return raw.decode("utf-8", "replace")


def _rows(text: str | None) -> list[dict]:
    if not text:
        return []
    return list(csv.DictReader(io.StringIO(text)))


# ---------------------------------------------------------------------------
# Baseline feeds
# ---------------------------------------------------------------------------

def weekly_stats(year: int, offline: bool = False) -> list[dict]:
    return _rows(fetch_text(STATS_URL.format(year=year),
                            f"stats_player_week_{year}.csv", offline))


def snaps(year: int, offline: bool = False) -> list[dict]:
    return _rows(fetch_text(SNAPS_URL.format(year=year),
                            f"snap_counts_{year}.csv", offline))


def games(offline: bool = False) -> list[dict]:
    return _rows(fetch_text(GAMES_URL, "games.csv", offline, max_age_days=6))


def players(offline: bool = False) -> list[dict]:
    return _rows(fetch_text(PLAYERS_URL, "players.csv", offline))


def injuries(year: int, offline: bool = False) -> list[dict]:
    """Weekly injury report. Empty for season > INJURIES_LAST_SEASON (source dead)."""
    if year > INJURIES_LAST_SEASON:
        return []
    return _rows(fetch_text(INJURIES_URL.format(year=year),
                            f"injuries_{year}.csv", offline, max_age_days=3))


# ---------------------------------------------------------------------------
# Deep feeds (MTNN v2)
# ---------------------------------------------------------------------------

def depth_charts(year: int, offline: bool = False) -> list[dict]:
    """Offense depth chart rows. Through 2024: week-keyed. 2025+: may be
    timestamped ESPN dumps — caller should as-of join when week is absent."""
    return _rows(fetch_text(DEPTH_URL.format(year=year),
                            f"depth_charts_{year}.csv", offline, max_age_days=2))


def ngs(stat: str, year: int, offline: bool = False) -> list[dict]:
    """Next Gen Stats weekly. stat in {passing, receiving, rushing}. 2016+."""
    if year < 2016:
        return []
    if stat not in ("passing", "receiving", "rushing"):
        raise ValueError(f"ngs stat must be passing|receiving|rushing, got {stat}")
    return _rows(fetch_text(NGS_URL.format(year=year, stat=stat),
                            f"ngs_{year}_{stat}.csv.gz", offline))


def pfr_adv(stat: str, year: int, offline: bool = False) -> list[dict]:
    """PFR advanced weekly. stat in {pass, rec, rush}."""
    if stat not in ("pass", "rec", "rush"):
        raise ValueError(f"pfr_adv stat must be pass|rec|rush, got {stat}")
    return _rows(fetch_text(PFR_ADV_URL.format(year=year, stat=stat),
                            f"advstats_week_{stat}_{year}.csv", offline))


def ep_weekly(year: int, offline: bool = False) -> list[dict]:
    """ffopportunity expected fantasy points by player-week (CC-BY-SA)."""
    return _rows(fetch_text(EP_WEEKLY_URL.format(year=year),
                            f"ep_weekly_{year}.csv", offline))


def draft_picks(offline: bool = False) -> list[dict]:
    return _rows(fetch_text(DRAFT_URL, "draft_picks.csv", offline))


def combine(offline: bool = False) -> list[dict]:
    return _rows(fetch_text(COMBINE_URL, "combine.csv", offline))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def num(row: dict, key: str, default: float = 0.0) -> float:
    v = row.get(key, "")
    if v in ("", "NA", "None", None):
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def available_seasons(last: int, offline: bool = False) -> list[int]:
    out = []
    for y in range(FIRST_SEASON, last + 1):
        if (CACHE / f"stats_player_week_{y}.csv").exists() or not offline:
            if weekly_stats(y, offline):
                out.append(y)
    return out


def latest_stats_season(offline: bool = False) -> int:
    for y in range(date.today().year + 1, FIRST_SEASON - 1, -1):
        if weekly_stats(y, offline):
            return y
    return FIRST_SEASON


def smoke(offline: bool = False) -> int:
    """Fetch one year of each deep feed and print row counts."""
    y = 2024
    checks = [
        ("weekly_stats", lambda: weekly_stats(y, offline)),
        ("snaps", lambda: snaps(y, offline)),
        ("depth_charts", lambda: depth_charts(y, offline)),
        ("ngs_receiving", lambda: ngs("receiving", y, offline)),
        ("ngs_rushing", lambda: ngs("rushing", y, offline)),
        ("ngs_passing", lambda: ngs("passing", y, offline)),
        ("pfr_rec", lambda: pfr_adv("rec", y, offline)),
        ("pfr_rush", lambda: pfr_adv("rush", y, offline)),
        ("pfr_pass", lambda: pfr_adv("pass", y, offline)),
        ("ep_weekly", lambda: ep_weekly(y, offline)),
        ("draft_picks", lambda: draft_picks(offline)),
        ("combine", lambda: combine(offline)),
    ]
    ok = 0
    for name, fn in checks:
        rows = fn()
        n = len(rows)
        print(f"  {name}: {n} rows")
        if n > 0:
            ok += 1
    print(f"smoke: {ok}/{len(checks)} feeds non-empty")
    return 0 if ok >= 8 else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(smoke("--offline" in sys.argv))
