"""
fetch_missing_core.py — Vector Gridiron resumable cache backfill (zero-deps)

Pattern from hoops/fetch_preseason_odds.py + merge_salaries.py adapted to NFL.

Domain: NFL / gridiron — next-game fantasy, injury, Vegas lines, weather

Audits:
- pipeline/cache/ (expected: nflverse_pbp_{year}.json, roster_*, weather_*, vegas_*)
- pipeline/data/ (expected: train_matrix.npz, embedding_gridiron.npz)
- assets/data/gridiron.json (1135 bytes skeleton currently)
- assets/vectors.json (398k partially populated)
- coverage years vs hoops 1996-97→2025-26

NFL equivalent of hoops cap_rules / payroll_by_season:
- NFL salary cap by season (hard cap vs NBA soft cap)
- NFL CBA era, TV deal era
- apron analog: none (hard cap single threshold) but note dead cap / franchise tag

Zero-deps resumable: skip if exists && size>0 unless --force
Merge without overwrite: preserves existing correct rows

Usage:
  python pipeline/fetch_missing_core.py --audit-only
  python pipeline/fetch_missing_core.py --year 2024 --dry-run
  python pipeline/fetch_missing_core.py --full --scaffold-write --force
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = ROOT / "pipeline" / "cache"
DATA_DIR = ROOT / "pipeline" / "data"
ASSETS_DATA = ROOT / "assets" / "data"
DEST_GRIDIRON = ASSETS_DATA / "gridiron.json"
DEST_VECTORS = ROOT / "assets" / "vectors.json"

# NFL cap by season — exact analogue of NBA nba_salary_cap.py CAP_BY_SEASON
# Source: Spotrac NFL cap history, OverTheCap
NFL_CAP_BY_SEASON: dict[str, float] = {
    "2011": 120_000_000,  # 2011 CBA start
    "2012": 120_600_000,
    "2013": 123_000_000,
    "2014": 133_000_000,
    "2015": 143_280_000,
    "2016": 155_270_000,
    "2017": 167_000_000,
    "2018": 177_200_000,
    "2019": 188_200_000,
    "2020": 198_200_000,  # COVID — actually flat but spent differently
    "2021": 182_500_000,  # COVID reduction
    "2022": 208_200_000,
    "2023": 224_800_000,
    "2024": 255_400_000,  # +30.6M biggest jump ever (new TV)
    "2025": 279_200_000,  # official 2025
    "2026": 295_000_000,  # est +5.7% smoothing — no repeat of 2024 spike
}

NFL_CBA_BY_SEASON: dict[str, str] = {
    "2011": "2011 CBA — 10yr, rookie wage scale, 47-48.5% revenue split",
    "2012": "2011 CBA",
    "2013": "2011 CBA",
    "2014": "2011 CBA",
    "2015": "2011 CBA",
    "2016": "2011 CBA",
    "2017": "2011 CBA",
    "2018": "2011 CBA",
    "2019": "2011 CBA",
    "2020": "2020 CBA — 17-game season, 48% player share grows to 48.5%, playoff expansion",
    "2021": "2020 CBA — 17 games",
    "2022": "2020 CBA",
    "2023": "2020 CBA",
    "2024": "2020 CBA — $110B TV deal starts 2023, cap spike 2024",
    "2025": "2020 CBA — year 2 of $110B",
    "2026": "2020 CBA",
}

NFL_TV_BY_SEASON: dict[str, str] = {
    "2011": "2006-13 FOX/CBS/NBC/ESPN $3.7B/yr",
    "2012": "2006-13",
    "2013": "2014-22 FOX/CBS/NBC/ESPN $5.9B/yr",
    "2014": "2014-22",
    "2015": "2014-22",
    "2016": "2014-22",
    "2017": "2014-22",
    "2018": "2014-22",
    "2019": "2014-22",
    "2020": "2014-22 — COVID",
    "2021": "2014-22",
    "2022": "2014-22 last year",
    "2023": "2023-33 FOX/CBS/NBC/ESPN/Amazon $10B/yr (~$110B 11yr) + YouTube $2B Sunday Ticket",
    "2024": "2023-33 — spike driver +30M cap",
    "2025": "2023-33 year2",
    "2026": "2023-33 year3 smoothing mandatory",
}

EXPECTED_YEARS = list(range(2020, 2026))  # 2020→2025 for next-game model (nflverse availability)
EXPECTED_CACHE_FILES = len(EXPECTED_YEARS) * 5  # pbp + roster + weather + vegas + participation
N_EXAMPLE = 2000  # synthetic 2000 rows mentioned in eval_scoreboard.json


def audit_cache() -> dict:
    cache_files = list(CACHE.glob("*.json")) + list(CACHE.glob("*.npz")) if CACHE.exists() else []
    data_files = list(DATA_DIR.glob("*")) if DATA_DIR.exists() else []
    cache_pop = [f for f in cache_files if f.is_file() and f.stat().st_size > 0]
    data_pop = [f for f in data_files if f.is_file() and f.stat().st_size > 0]
    empty = [f for f in cache_files if f.is_file() and f.stat().st_size == 0]

    years_found = set()
    for f in cache_files:
        m = re.search(r"(20\d{2})", f.name)
        if m:
            years_found.add(int(m.group(1)))
    missing_years = [y for y in EXPECTED_YEARS if y not in years_found]

    skeleton = True
    gridiron_count = 0
    vectors_count = 0
    gridiron_bytes = 0
    vectors_bytes = 0
    if DEST_GRIDIRON.exists():
        gridiron_bytes = DEST_GRIDIRON.stat().st_size
        try:
            g = json.loads(DEST_GRIDIRON.read_text()[:2_000_000])
            if isinstance(g, dict) and "players" in g:
                gridiron_count = len(g["players"])
            elif isinstance(g, list):
                gridiron_count = len(g)
            else:
                gridiron_count = 1
            skeleton = gridiron_bytes < 5000 or gridiron_count < 100
        except Exception:
            skeleton = gridiron_bytes < 5000
    if DEST_VECTORS.exists():
        vectors_bytes = DEST_VECTORS.stat().st_size
        try:
            v = json.loads(DEST_VECTORS.read_text()[:1_000_000])
            vectors_count = len(v) if isinstance(v, list | dict) else 0
            if isinstance(v, dict):
                vectors_count = len(v.get("players", v))
        except Exception:
            pass

    expected_data = 2  # train_matrix.npz + embedding_gridiron.npz
    data_missing = max(0, expected_data - len(data_pop))
    missing_cache = max(0, EXPECTED_CACHE_FILES - len(cache_pop))
    total_expected = EXPECTED_CACHE_FILES + expected_data
    total_pop = len(cache_pop) + len(data_pop)
    missing_pct = 0 if total_expected == 0 else (total_expected - total_pop) / total_expected * 100

    return {
        "domain": "gridiron",
        "cache_dir": str(CACHE),
        "cache_files": len(cache_files),
        "cache_populated": len(cache_pop),
        "cache_empty": len(empty),
        "cache_years_found": sorted(years_found),
        "missing_years": missing_years,
        "missing_years_count": len(missing_years),
        "expected_cache": EXPECTED_CACHE_FILES,
        "data_dir": str(DATA_DIR),
        "data_files": len(data_files),
        "data_populated": len(data_pop),
        "data_missing": data_missing,
        "expected_data": expected_data,
        "total_expected": total_expected,
        "populated_total": total_pop,
        "missing_cache": missing_cache,
        "missing_pct": round(missing_pct, 1),
        "assets_gridiron_exists": DEST_GRIDIRON.exists(),
        "assets_gridiron_bytes": gridiron_bytes,
        "assets_gridiron_count": gridiron_count,
        "assets_vectors_bytes": vectors_bytes,
        "assets_vectors_count": vectors_count,
        "assets_skeleton": skeleton,
        "coverage_years": f"{EXPECTED_YEARS[0]}-{EXPECTED_YEARS[-1]}",
        "nfl_cap_reference": ("pipeline/cache/nfl_cap_rules.json equivalent to NBA nba_salary_cap.py + cap_rules.json"),
        "expected_vs_hoops": (
            "hoops 686 files 51M 30 seasons fully populated vs gridiron 0 cache files, "
            "data/ missing npz, assets/gridiron.json 1135B stub, vectors 398k partial — 99% cache miss"
        ),
    }


def write_nfl_cap_rules():
    """NFL analog of hoops nba_salary_cap.py + cap_rules.json + payroll_by_season.json"""
    out = CACHE / "nfl_cap_rules.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cap_by_season": NFL_CAP_BY_SEASON,
        "cba_by_season": NFL_CBA_BY_SEASON,
        "tv_by_season": NFL_TV_BY_SEASON,
        "notes": {
            "hard_cap_vs_soft": (
                "NFL is hard cap (no luxury tax/aprons unlike NBA 2023 CBA). "
                "Single threshold — dead cap hits matter more than cap_pct."
            ),
            "franchise_tag": (
                "Tag value = top5 avg at position — restricts mobility, analog to NBA max but team-initiated."
            ),
            "cap_floor": ("89% cash spend floor over 4yr period (2021-24 89%) — no per-season floor like NBA 90%."),
            "spike_year": (
                "2024 +$30.6M (13.6%) from $110B TV — NFL equivalent of NBA 2016 +34% spike, "
                "but mandatory smoothing 2025+ via 2020 CBA growth formula."
            ),
            "source": "Spotrac, OverTheCap, ESPN, NFL Communications 2025 official $279.2M team cap",
        },
    }
    if out.exists() and "--force" not in sys.argv:
        try:
            existing = json.loads(out.read_text())
            # merge: keep existing richer keys, fill missing seasons like hoops merge pattern
            for k in ["cap_by_season", "cba_by_season", "tv_by_season"]:
                if k in payload and k in existing:
                    for season, val in payload[k].items():
                        if season not in existing[k]:
                            existing[k][season] = val
            out.write_text(json.dumps(existing, indent=2))
            print(f"merged {out}")
            return
        except Exception:
            pass
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out} with {len(NFL_CAP_BY_SEASON)} seasons — hard cap analogy to NBA soft cap")


def fetch_year_placeholder(year: int, force=False, offline=False) -> bool:
    CACHE.mkdir(parents=True, exist_ok=True)
    expected = [
        CACHE / f"nflverse_pbp_{year}.json",
        CACHE / f"roster_{year}.json",
        CACHE / f"weather_{year}.json",
        CACHE / f"vegas_{year}.json",
        CACHE / f"participation_{year}.json",
    ]
    if not force and all(p.exists() and p.stat().st_size > 0 for p in expected):
        return True
    if offline:
        return False
    if "--scaffold-write" in sys.argv:
        for p in expected:
            if p.exists() and not force:
                continue
            stub = {
                "year": year,
                "type": p.stem.split("_")[0],
                "rows": 0,
                "stub": True,
                "_scaffold": "fetch_missing_core.py placeholder",
            }
            p.write_text(json.dumps(stub))
        return True
    return False


def main():
    args = sys.argv[1:]
    audit_only = "--audit-only" in args or ("--offline" in args and "--full" not in args)
    dry_run = "--dry-run" in args
    force = "--force" in args
    offline = "--offline" in args
    year_filter = None
    if "--year" in args:
        idx = args.index("--year")
        if idx + 1 < len(args):
            try:
                year_filter = int(args[idx + 1])
            except Exception:
                pass

    audit = audit_cache()
    print(json.dumps(audit, indent=2))

    if dry_run or (audit_only and "--full" not in args and "--scaffold-write" not in args):
        print(
            f"\nGridiron missing {audit['missing_pct']}% cache — "
            f"{audit['populated_total']}/{audit['total_expected']} files"
        )
        print(f"Years missing: {audit['missing_years_count']} — {audit['missing_years']}")
        print(
            f"Skeleton? gridiron.json {audit['assets_gridiron_bytes']}B count={audit['assets_gridiron_count']} "
            f"vectors {audit['assets_vectors_bytes']}B skeleton={audit['assets_skeleton']}"
        )
        print(
            "vs hoops: 51M 686 files — gridiron 0 files, synthetic fallback train_mtnn.py "
            "reports MAE 8.47 vs claimed 4.268 not reproducible offline"
        )
        if not dry_run and audit_only:
            return

    write_nfl_cap_rules()

    if year_filter:
        fetch_year_placeholder(year_filter, force=force, offline=offline)

    if "--full" in args or "--scaffold-write" in args:
        years = [year_filter] if year_filter else EXPECTED_YEARS
        for y in years:
            fetch_year_placeholder(y, force=force, offline=offline)
            time.sleep(0.05)

    print("\nDone gridiron fetch_missing_core.")
    print("Wire real fetch: nflreadpy / nfl_data_py load_pbp seasons, roster, snap counts,")
    print("Open-Meteo weather join, Vegas betting join (nflverse or sportsoddshistory),")
    print(
        "then build pipeline/data/train_matrix.npz X[M,160] M mask Y next-game FPTS "
        "like eval_scoreboard.json plan_for_nflverse_fetch."
    )


if __name__ == "__main__":
    main()
