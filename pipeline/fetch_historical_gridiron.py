"""
Vector Gridiron historical backfill — zero-deps, offline-first, resumable.
Analogous to vector-hoops fetch_odds_sportsoddshistory + fetch_draft_history.

Covers NFL win totals pattern:
  https://www.covers.com/sportsoddshistory/nfl-team/?sa=nfl&Team=TEAM_ABBR
  Team pages list year-by-year O/U (like Covers NBA pattern).

Draft pedigree + combine (40-yard dash) placeholder:
  - nflverse draft picks via nflreadpy wrapper (future)
  - combine via nfldata.com combine JSON (placeholder offline)

Writes:
  assets/data/nfl_win_totals.json
    {
      built: ISO8601,
      source: str,
      coverage: "NxM >=20 teams",
      seasons: { "2024": {"ARI":6.5, ...}, ... }
    }

Zero-deps: urllib + re + json + pathlib + time + random + subprocess(curl)
Resumable: merges into existing file, skips seasons already >=20 teams unless --force.
Offline-first: if --offline or network blocked, returns existing file and exits 0.
No fake metrics: empty seasons stay empty, never synthesized.

Usage:
  python pipeline/fetch_historical_gridiron.py
  python pipeline/fetch_historical_gridiron.py --offline
  python pipeline/fetch_historical_gridiron.py --team ARI --force
  python pipeline/fetch_historical_gridiron.py --allow-gambling  # enables Covers fetch (gambling domain gate)
"""

from __future__ import annotations
import json
import re
import sys
import time
import random
import pathlib
import urllib.request
import urllib.error
import subprocess
import datetime
import argparse
from urllib.parse import quote_plus

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEST = ROOT / "assets" / "data" / "nfl_win_totals.json"
CACHE_DIR = ROOT / "pipeline" / "cache"
CACHE_RAW = CACHE_DIR / "nfl_win_totals_soh_raw.json"
CACHE_COVERS = CACHE_DIR / "covers_nfl_raw.json"

# NFL 32 teams — Covers Team param uses city abbreviation sometimes.
NFL_TEAMS = [
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB",
    "HOU","IND","JAX","KC","LV","LAC","LAR","MIA","MIN","NE","NO","NYG",
    "NYJ","PHI","PIT","SF","SEA","TB","TEN","WAS"
]
TEAMS_FULL = [
    "Arizona Cardinals","Atlanta Falcons","Baltimore Ravens","Buffalo Bills",
    "Carolina Panthers","Chicago Bears","Cincinnati Bengals","Cleveland Browns",
    "Dallas Cowboys","Denver Broncos","Detroit Lions","Green Bay Packers",
    "Houston Texans","Indianapolis Colts","Jacksonville Jaguars","Kansas City Chiefs",
    "Las Vegas Raiders","Los Angeles Chargers","Los Angeles Rams","Miami Dolphins",
    "Minnesota Vikings","New England Patriots","New Orleans Saints","New York Giants",
    "New York Jets","Philadelphia Eagles","Pittsburgh Steelers","San Francisco 49ers",
    "Seattle Seahawks","Tampa Bay Buccaneers","Tennessee Titans","Washington Commanders"
]
ABBR_MAP = {
    "Arizona Cardinals":"ARI","Atlanta Falcons":"ATL","Baltimore Ravens":"BAL","Buffalo Bills":"BUF",
    "Carolina Panthers":"CAR","Chicago Bears":"CHI","Cincinnati Bengals":"CIN","Cleveland Browns":"CLE",
    "Dallas Cowboys":"DAL","Denver Broncos":"DEN","Detroit Lions":"DET","Green Bay Packers":"GB",
    "Houston Texans":"HOU","Indianapolis Colts":"IND","Jacksonville Jaguars":"JAX","Kansas City Chiefs":"KC",
    "Las Vegas Raiders":"LV","Los Angeles Chargers":"LAC","Los Angeles Rams":"LAR","Miami Dolphins":"MIA",
    "Minnesota Vikings":"MIN","New England Patriots":"NE","New Orleans Saints":"NO","New York Giants":"NYG",
    "New York Jets":"NYJ","Philadelphia Eagles":"PHI","Pittsburgh Steelers":"PIT","San Francisco 49ers":"SF",
    "Seattle Seahawks":"SEA","Tampa Bay Buccaneers":"TB","Tennessee Titans":"TEN","Washington Commanders":"WAS"
}
TEAM_FULL = {
    "ARI":"Arizona Cardinals","ATL":"Atlanta Falcons","BAL":"Baltimore Ravens",
    "BUF":"Buffalo Bills","CAR":"Carolina Panthers","CHI":"Chicago Bears",
    "CIN":"Cincinnati Bengals","CLE":"Cleveland Browns","DAL":"Dallas Cowboys",
    "DEN":"Denver Broncos","DET":"Detroit Lions","GB":"Green Bay Packers",
    "HOU":"Houston Texans","IND":"Indianapolis Colts","JAX":"Jacksonville Jaguars",
    "KC":"Kansas City Chiefs","LV":"Las Vegas Raiders","LAC":"Los Angeles Chargers",
    "LAR":"Los Angeles Rams","MIA":"Miami Dolphins","MIN":"Minnesota Vikings",
    "NE":"New England Patriots","NO":"New Orleans Saints","NYG":"New York Giants",
    "NYJ":"New York Jets","PHI":"Philadelphia Eagles","PIT":"Pittsburgh Steelers",
    "SF":"San Francisco 49ers","SEA":"Seattle Seahawks","TB":"Tampa Bay Buccaneers",
    "TEN":"Tennessee Titans","WAS":"Washington Commanders",
}

UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML,like Gecko) Chrome/122.0",
    "Scout/1.1 (+vector-gridiron backfill; respectful 4s delay)",
]

def fetch_url(url: str, via="direct") -> str:
    """Fetch with UA rotation, curl fallback. Returns '' on fail."""
    for ua in UA_POOL:
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": ua,
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "identity",
                "Referer": "https://www.covers.com/",
            })
            with urllib.request.urlopen(req, timeout=20) as r:
                data = r.read().decode("utf-8", errors="ignore")
                if len(data) > 800:
                    low = data.lower()
                    if "you have exceeded" in low and len(data) < 7000:
                        print(f"    {via} {url[:80]} BLOCK flagged ({len(data)} chars)", flush=True)
                        continue
                    if "rate limit" in low and len(data) < 7000:
                        print(f"    {via} {url[:80]} rate-limited", flush=True)
                        continue
                    return data
        except urllib.error.HTTPError as e:
            if e.code in (404, 410):
                return ""
            time.sleep(1.0)
        except Exception:
            time.sleep(0.6)
            continue
    try:
        html = subprocess.check_output(
            ["curl","-sL","-A", random.choice(UA_POOL), "--compressed","-m","20", url],
            text=True, stderr=subprocess.DEVNULL
        )
        if len(html) > 800:
            if "rate limit" in html.lower() and len(html) < 7000:
                print(f"    curl {url[:80]} rate-limited", flush=True)
                return ""
            return html
    except Exception:
        pass
    return ""

def parse_covers_team_page(html: str) -> dict[int, float]:
    """
    Covers NFL team history page contains table with year + win totals O/U.
    Returns {year: total}
    """
    if not html:
        return {}
    out: dict[int, float] = {}
    # Primary: Covers Regular Season Win Totals table
    idx = html.find("Regular Season Win Totals")
    search_area = html[idx:idx+50000] if idx != -1 else html
    # pattern <td>2024</td> <td>6.5</td>
    for year_str, ou_str in re.findall(r'<td[^>]*>\s*(20[0-2]\d|19\d{2})\s*</td>\s*.*?<\s*td[^>]*>\s*(\d{1,2}\.[05]|N/A)\s*</td>', search_area, re.S | re.I):
        if ou_str == "N/A":
            continue
        try:
            y = int(year_str)
            v = float(ou_str)
            if 2.5 <= v <= 14.5 and y not in out:
                out[y] = v
        except:
            pass
    if len(out) >= 2:
        return out
    # fallback looser
    for m in re.finditer(r'(20[0-2]\d)[^0-9]{1,80}(\d{1,2}\.[05])', html):
        try:
            y = int(m.group(1))
            v = float(m.group(2))
            if 2.5 <= v <= 14.5 and y not in out:
                out[y] = v
        except:
            continue
    return out

def fetch_covers_nfl_win_totals(allow_gambling=False, single_team=None, force=False, offline=False, since=2000):
    """
    Covers NFL fetch analogous to hoops Covers pattern.
    If offline or not allow_gambling, returns existing DEST unchanged (honest).
    Resumable merge into DEST.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        try:
            doc = json.loads(DEST.read_text(encoding="utf-8"))
        except:
            doc = {"built":"","source":"","coverage":"","seasons":{}}
    else:
        doc = {"built":"","source":"","coverage":"","seasons":{}}

    seasons: dict[str, dict[str, float]] = doc.get("seasons", {})
    # Handle legacy flat doc where seasons dict is root
    if not seasons and doc and all(isinstance(v, dict) for k,v in doc.items() if k not in ("built","source","coverage","seasons")):
        seasons = {k:v for k,v in doc.items() if isinstance(v, dict) and k not in ("built","source","coverage")}

    seasons_norm: dict[str, dict[str, float]] = {}
    for k, v in seasons.items():
        if k == "seasons":
            continue
        if isinstance(v, dict):
            seasons_norm[str(k)] = {str(team).upper(): float(val) for team, val in v.items() if isinstance(val, (int,float))}
        else:
            seasons_norm[str(k)] = {}
    for y in range(2003, 2026):
        seasons_norm.setdefault(str(y), {})

    if offline:
        print("[gridiron] offline flag — returning existing without fetch", flush=True)
        # if missing, still create offline stub (zero-deps, resumable)
        if not DEST.exists():
            doc["built"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            doc["source"] = "offline stub — Covers gated, BetMGM/manual optional"
            doc["coverage"] = f"{sum(1 for vs in seasons_norm.values() if len(vs)>=20)}/{len(seasons_norm)} >=20"
            doc["seasons"] = seasons_norm
            DEST.parent.mkdir(parents=True, exist_ok=True)
            DEST.write_text(json.dumps(doc, indent=2), encoding="utf-8")
            print(f"[gridiron] wrote offline stub {DEST}")
        return doc

    if not allow_gambling:
        print("[gridiron] Covers is gambling domain — pass --allow-gambling after user confirmation to enable live fetch.", flush=True)
        print("[gridiron] Still ensuring DEST exists/offline stub preserved.", flush=True)
        if not DEST.exists():
            doc["built"] = datetime.datetime.utcnow().isoformat()+"Z"
            doc["source"] = "offline stub — Covers gated, BetMGM/manual optional"
            doc["coverage"] = f"{sum(1 for vs in seasons_norm.values() if len(vs)>=20)}/{len(seasons_norm)} >=20"
            doc["seasons"] = seasons_norm
            DEST.parent.mkdir(parents=True, exist_ok=True)
            DEST.write_text(json.dumps(doc, indent=2), encoding="utf-8")
            print(f"[gridiron] wrote offline stub {DEST}")
        return doc

    merged_any = False
    raw_cache = {}
    if CACHE_COVERS.exists():
        try:
            raw_cache = json.loads(CACHE_COVERS.read_text())
        except:
            raw_cache = {}

    # Resolve team list
    if single_team:
        teams_iter = [single_team.upper()]
        # map full names if user passed full
        if single_team in TEAMS_FULL:
            teams_iter = [ABBR_MAP[single_team]]
    else:
        teams_iter = NFL_TEAMS

    for team in teams_iter:
        full_name = TEAM_FULL.get(team, team)
        team_q = quote_plus(full_name)
        covers_urls = [
            f"https://www.covers.com/sportsoddshistory/nfl-team/?sa=nfl&Team={team_q}",
            f"https://www.covers.com/sportsoddshistory/nfl-team/?sa=nfl&Team={team}",
            f"https://www.covers.com/sports/nfl/{full_name.lower().replace(' ','-')}-vs-spread-history",
        ]
        print(f"\n=== Covers NFL team {team} {full_name} ===", flush=True)
        team_parsed = None
        for url in covers_urls:
            print(f"  fetch {url[:100]}", flush=True)
            html = fetch_url(url, via="covers-nfl")
            if not html:
                print(f"    empty", flush=True)
                continue
            print(f"    {len(html)} chars", flush=True)
            parsed = parse_covers_team_page(html)
            print(f"    parsed {len(parsed)} years {sorted(parsed.items())[:3]}", flush=True)
            if len(parsed) >= 2:
                team_parsed = parsed
                raw_cache[team] = {"url": url, "parsed": parsed, "ts": datetime.datetime.utcnow().isoformat()}
                CACHE_COVERS.write_text(json.dumps(raw_cache, indent=2), encoding="utf-8")
                break
        if not team_parsed:
            print(f"  no data for {team}, skip", flush=True)
            time.sleep(3.5 + random.random()*1.2)
            continue

        for year, total in team_parsed.items():
            if year < since:
                continue
            sy = str(year)
            seasons_norm.setdefault(sy, {})
            seasons_norm[sy][team] = float(total)
            merged_any = True

        if merged_any:
            doc["built"] = datetime.datetime.utcnow().isoformat()+"Z"
            doc["source"] = "Covers SportsOddsHistory NFL team pages Regular Season Win Totals + BetMGM manual (gated)"
            total_y = len(seasons_norm)
            full = sum(1 for vs in seasons_norm.values() if len(vs)>=20)
            total_entries = sum(len(v) for v in seasons_norm.values())
            doc["coverage"] = f"{full}/{total_y} >=20 teams, total entries {total_entries}"
            doc["seasons"] = seasons_norm
            DEST.write_text(json.dumps(doc, indent=2), encoding="utf-8")
            print(f"  merged checkpoint {DEST} coverage {doc['coverage']}", flush=True)

        time.sleep(3.5 + random.uniform(0,1.8))

    if merged_any or not DEST.exists():
        doc["built"] = datetime.datetime.utcnow().isoformat()+"Z"
        doc["source"] = "Covers SportsOddsHistory NFL team pages O/U + BetMGM manual (gated)" if allow_gambling else "offline stub — Covers gated"
        full = sum(1 for vs in seasons_norm.values() if len(vs)>=20)
        doc["coverage"] = f"{full}/{len(seasons_norm)} >=20 teams, total entries {sum(len(v) for v in seasons_norm.values())}"
        doc["seasons"] = seasons_norm
        DEST.parent.mkdir(parents=True, exist_ok=True)
        DEST.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    print("\n=== FINAL SUMMARY NFL win totals ===")
    print(f"seasons {len(seasons_norm)} full>=20 {sum(1 for v in seasons_norm.values() if len(v)>=20)} total entries {sum(len(v) for v in seasons_norm.values())}")
    for y in sorted(seasons_norm.keys()):
        print(f" {y}: {len(seasons_norm[y])}")
    return doc

def fetch_draft_pedigree_stub(out_path=None, offline=False):
    path = pathlib.Path(out_path) if out_path else ROOT / "assets" / "data" / "draft_pedigree.json"
    if path.exists() and offline:
        print(f"[draft] offline — keeping existing {path}")
        try:
            return json.loads(path.read_text())
        except:
            return {"built":"","source":"","players":{}}
    if path.exists():
        try:
            doc = json.loads(path.read_text())
            if doc.get("players") and len(doc["players"]) > 10 and "--force" not in sys.argv:
                print(f"[draft] existing {path} {len(doc['players'])} players, skip (use --force to refetch)")
                return doc
        except:
            pass

    stub = {
        "built": datetime.datetime.utcnow().isoformat()+"Z",
        "source": "stub — nflverse draft + combine planned (nflreadpy load_draft_picks / load_combine)",
        "coverage": "0 players — needs nflverse local GPU lane",
        "players": {},
        "note": "Gridiron analog to hoops draft pedigree: round/pick affects fantasy prior + forty affects speed score. Combine 40-yard dash ~4.22-4.84 WR/RB. Analogous to hoops 1.4MB player props.",
        "todo": [
            "Port hoops pipeline/fetch_draft_history.py -> NFL draft API (nflverse nfl_draft_picks)",
            "Add combine 40-yard dash from https://www.nfl.com/combine/tracker or nflreadpy load_combine()",
            "Wire into pipeline/train_mtnn.py tower age+athleticism family (age 8 already; add forty, vert, bench)",
            "Use in front-office eval similar to hoops Front Office Lab"
        ]
    }
    vectors_path = ROOT / "assets" / "vectors.json"
    if vectors_path.exists():
        try:
            j = json.loads(vectors_path.read_text())
            players = j.get("players", []) if isinstance(j, dict) else []
            if players:
                stub["players"] = {
                    p.get("name","player_"+str(i)): {
                        "draft_year": None,
                        "draft_round": None,
                        "draft_pick": None,
                        "forty": None,
                        "stub": True
                    } for i, p in enumerate(players[:20])
                }
                stub["coverage"] = f"stub {len(stub['players'])} players with null actual — template only"
        except:
            pass

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stub, indent=2), encoding="utf-8")
    print(f"[draft] wrote stub {path} coverage {stub['coverage']}")
    return stub

def main():
    ap = argparse.ArgumentParser(description="Vector Gridiron historical backfill — zero-deps offline-first resumable")
    ap.add_argument("--since", type=int, default=2003, help="earliest year")
    ap.add_argument("--team", type=str, default=None, help="single team abbrev ARI etc or full name")
    ap.add_argument("--offline", action="store_true", help="offline — keep existing, no network")
    ap.add_argument("--allow-gambling", action="store_true", help="enable Covers fetch (gambling domain gate)")
    ap.add_argument("--force", action="store_true", help="force refetch even if season full")
    args = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if args.team:
        # allow full name with spaces
        team_arg = args.team.strip()
    else:
        team_arg = None

    fetch_covers_nfl_win_totals(
        allow_gambling=args.allow_gambling,
        single_team=team_arg,
        force=args.force,
        offline=args.offline,
        since=args.since
    )

    fetch_draft_pedigree_stub(offline=args.offline)

    print("\n[gridiron] Done. Output assets/data/nfl_win_totals.json structure:")
    print("  {built, source, coverage, seasons: {season: {team: total}}}")

    try:
        if DEST.exists():
            doc = json.loads(DEST.read_text())
            assert "built" in doc and "seasons" in doc
            print(f"[gate] {DEST} OK {doc.get('coverage')}")
    except Exception as e:
        print(f"[gate] FAIL {DEST}: {e}")
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
