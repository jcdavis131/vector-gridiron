"""Consensus ADP (average draft position) from a variety of free sources, so
the draft board can flag draft-day value — where the model likes a player more
than the room is paying for him.

Sources (all free JSON, cached under pipeline/cache/):
  Fantasy Football Calculator  PPR + Half-PPR  (name-keyed)
  MyFantasyLeague              PPR             (id-keyed -> name via players feed)

Each source's overall-pick ADP is joined on a normalized name+position key
(same key the projection board uses), then averaged into a consensus ADP with a
spread, per-source breakdown, and overall/positional draft ranks.

  python pipeline/build_adp.py [year]   ->  assets/adp.json
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
CACHE = ROOT / "pipeline" / "cache"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) vector-gridiron-adp/1.0"
SKILL = {"QB", "RB", "WR", "TE"}


def norm_key(name: str, pos: str) -> str:
    s = (name or "").lower()
    s = re.sub(r"[.'`]", "", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s)
    s = re.sub(r"[^a-z ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return f"{s}|{pos}"


def fetch_json(url: str, cache_name: str, ttl_days: float = 1.0):
    """Fetch fresh (ADP moves daily); fall back to cache on any failure."""
    p = CACHE / cache_name
    fresh = p.exists() and (time.time() - p.stat().st_mtime) < ttl_days * 86400
    if fresh:
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    try:
        raw = urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": UA}), timeout=45).read()
        CACHE.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
        time.sleep(0.3)
        return json.loads(raw)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as e:
        print(f"  {cache_name}: fetch failed ({type(e).__name__}); "
              f"{'using cache' if p.exists() else 'skipping'}")
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None


def ffc(fmt: str, year: int, teams: int = 12) -> dict:
    url = f"https://fantasyfootballcalculator.com/api/v1/adp/{fmt}?teams={teams}&year={year}"
    d = fetch_json(url, f"adp_ffc_{fmt}_{year}.json")
    out = {}
    for p in (d or {}).get("players", []) or []:
        pos = p.get("position")
        if pos in SKILL and p.get("adp"):
            out[norm_key(p["name"], pos)] = {"name": p["name"], "pos": pos,
                                             "team": p.get("team", ""), "adp": float(p["adp"])}
    return out


def mfl(year: int) -> dict:
    pl = fetch_json(f"https://api.myfantasyleague.com/{year}/export?TYPE=players&DETAILS=1&JSON=1",
                    f"adp_mfl_players_{year}.json", ttl_days=7)
    idmap = {}
    for p in (((pl or {}).get("players") or {}).get("player") or []):
        nm = p.get("name", "")
        if "," in nm:
            last, first = [x.strip() for x in nm.split(",", 1)]
            nm = f"{first} {last}"
        idmap[p.get("id")] = {"name": nm, "pos": p.get("position", ""), "team": p.get("team", "")}
    ad = fetch_json(f"https://api.myfantasyleague.com/{year}/export?TYPE=adp&PERIOD=RECENT"
                    f"&FCOUNT=12&IS_PPR=1&IS_KEEPER=N&IS_MOCK=-1&CUTOFF=5&JSON=1",
                    f"adp_mfl_ppr_{year}.json")
    out = {}
    for p in (((ad or {}).get("adp") or {}).get("player") or []):
        info = idmap.get(p.get("id"))
        if info and info["pos"] in SKILL and p.get("averagePick"):
            out[norm_key(info["name"], info["pos"])] = {"name": info["name"], "pos": info["pos"],
                                                        "team": info["team"], "adp": float(p["averagePick"])}
    return out


def main() -> int:
    year = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else date.today().year
    print(f"pulling ADP for {year} ...")
    sources = {}
    for label, getter in (("ffc_ppr", lambda: ffc("ppr", year)),
                          ("ffc_half", lambda: ffc("half-ppr", year)),
                          ("mfl", lambda: mfl(year))):
        s = getter()
        if s:
            sources[label] = s
            print(f"  {label}: {len(s)} players")

    if not sources:
        # offseason / all sources down: write an empty (but valid) artifact
        ASSETS.mkdir(exist_ok=True)
        (ASSETS / "adp.json").write_text(json.dumps(
            {"built": time.strftime("%Y-%m-%d"), "season": year, "sources": [],
             "count": 0, "players": []}, separators=(",", ":")), encoding="utf-8")
        print("no ADP sources returned data — wrote empty adp.json")
        return 0

    keys = set().union(*[set(s) for s in sources.values()])
    players = []
    for k in keys:
        per, adps, ident = {}, [], None
        for label, s in sources.items():
            if k in s:
                per[label] = round(s[k]["adp"], 1)
                adps.append(s[k]["adp"])
                ident = s[k]
        consensus = sum(adps) / len(adps)
        spread = (max(adps) - min(adps)) if len(adps) > 1 else 0.0
        players.append({"key": k, "name": ident["name"], "pos": ident["pos"],
                        "team": ident["team"], "adp": round(consensus, 1),
                        "spread": round(spread, 1), "n": len(adps), "per": per})

    players.sort(key=lambda p: p["adp"])
    for i, p in enumerate(players):
        p["adp_rank"] = i + 1
    for pos in ("QB", "RB", "WR", "TE"):
        for j, p in enumerate([x for x in players if x["pos"] == pos]):
            p["pos_rank"] = j + 1

    ASSETS.mkdir(exist_ok=True)
    (ASSETS / "adp.json").write_text(json.dumps({
        "built": time.strftime("%Y-%m-%d"), "season": year,
        "sources": list(sources), "count": len(players), "players": players,
    }, separators=(",", ":")), encoding="utf-8")
    print(f"\nwrote adp.json: {len(players)} players from {len(sources)} sources "
          f"{list(sources)}")
    for p in players[:5]:
        print(f"  ADP {p['adp']:5} {p['name']:22} {p['pos']}  ({p['n']} src, ±{p['spread']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
