"""Derive the page's coordinate file from this repo's real vectors export.

`assets/player_seasons.json` is written by pipeline/build_player_seasons.py from nflverse
player_stats:
10,700 player-seasons, 2015-2025, four positions, with the player's real name, team, games
and points per game. Its x/y/z are a PCA-3 of the 18-d per-game rate profile `v`, z-scored
within season across QB/RB/WR/TE. That claim was checked, not taken on trust: recomputing
the PCA here reproduces the shipped x, y and z at |r| = 1.0000 on all three axes.

WHAT THESE COORDINATES ARE, PRECISELY. They are a projection of the FEATURE profile, not of
a learned embedding. The trained model's own output lives in assets/embedding.json - MTNN v2
gated-fusion, 32-d, PCA-3 - and covers 503 players for the 2025 season only. Calling this
map an "embedding map" without that distinction is the kind of overclaim this rebuild
exists to remove, so the page says which one it is drawing.

WHY NOT assets/vectors.json. That filename is taken on this branch by a different artifact
entirely: pipeline/train_mtnn.py writes the model export there and pipeline/eval_next_game.py
reads its fpts_next_pred/fpts_next_true fields back to compute MAE and R2. Its 1,000 rows are
named "player_0_16" through "player_999_x" with seasons numbered 0 to 5 - fine as a training
scratch file, useless as a map. An earlier version of this work overwrote it, which would have
silently broken the eval path. The real player-seasons live under their own name instead.

WHAT IS DELIBERATELY NOT USED. vector-hub/assets/data/gridiron.json holds 646 rows, all
season 2026, and every single name in it is a placeholder of the form "Player 70 QB" - a
roster view whose name join failed. None of it reaches this page.

Re-run after any re-export:  python scripts/build_gridiron_map.py
"""
import datetime
import json
import math
import pathlib
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "assets" / "player_seasons.json"
OUT = ROOT / "public" / "assets" / "gridiron_map.json"
ROSTER_OUT = ROOT / "public" / "assets" / "gridiron_roster.json"
PER_POS = 3

if not SRC.exists():
    raise SystemExit(f"{SRC} is missing; this script has nothing real to read.")

blob = json.loads(SRC.read_bytes().decode("utf-8"))
rows = blob["players"] if isinstance(blob, dict) and "players" in blob else (
    blob if isinstance(blob, list) else next(v for v in blob.values() if isinstance(v, list)))

fake = [r["name"] for r in rows if re.match(r"^(Player|player)[ _]?\d", str(r.get("name", "")))]
if fake:
    raise SystemExit(f"{len(fake)} rows carry placeholder names (e.g. {fake[0]!r}); refusing")
bad = [i for i, r in enumerate(rows)
       if not r.get("name") or r.get("x") is None or r.get("y") is None or not r.get("v")]
if bad:
    raise SystemExit(f"{len(bad)} rows lack a name, coordinates or a vector; refusing a partial map")

# ---- verify the shipped coordinates really are a PCA-3 of v --------------------------------
V = [list(map(float, r["v"])) for r in rows]
n, d = len(V), len(V[0])
mean = [sum(row[k] for row in V) / n for k in range(d)]
C = [[V[i][k] - mean[k] for k in range(d)] for i in range(n)]


def _matvec(vec):
    out = [0.0] * d
    for row in C:
        s = sum(row[k] * vec[k] for k in range(d))
        for k in range(d):
            out[k] += row[k] * s
    return out


comps, evs = [], []
for c in range(3):
    vec = [1.0 if k == c else 0.31 / (k + 1) for k in range(d)]      # fixed, not random
    nv = math.sqrt(sum(x * x for x in vec)) or 1.0
    vec = [x / nv for x in vec]
    for _ in range(400):
        w = _matvec(vec)
        for prev in comps:
            dot = sum(w[k] * prev[k] for k in range(d))
            w = [w[k] - dot * prev[k] for k in range(d)]
        nw = math.sqrt(sum(x * x for x in w))
        if nw < 1e-12:
            break
        nxt = [x / nw for x in w]
        if sum(abs(nxt[k] - vec[k]) for k in range(d)) < 1e-11:
            vec = nxt
            break
        vec = nxt
    comps.append(vec)
    evs.append(sum(sum(row[k] * vec[k] for k in range(d)) ** 2 for row in C) / n)

total_var = sum(sum(row[k] ** 2 for k in range(d)) for row in C) / n
explained = [round(e / total_var, 4) for e in evs] if total_var else [0, 0, 0]
proj = [[sum(C[i][k] * comps[j][k] for k in range(d)) for j in range(3)] for i in range(n)]


def _corr(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return abs(num / (da * db)) if da and db else 0.0


checks = []
for j, ax in enumerate("xyz"):
    r = _corr([p[j] for p in proj], [float(row[ax]) for row in rows])
    checks.append(round(r, 6))
    if r < 0.999:
        raise SystemExit(
            f"REFUSING: the shipped {ax} is not the PCA-3 component it claims to be "
            f"(|r| = {r:.4f}). Something re-wrote the coordinates without re-running the PCA.")
print(f"  coordinate provenance verified: |r| x/y/z = {checks} against a recomputed PCA-3 of v")

out = [{
    "name": r["name"], "pos": r.get("pos"), "team": r.get("team"),
    "season": str(r.get("season")), "games": r.get("games"), "ppg": r.get("ppg"),
    "x": round(float(r["x"]), 4), "y": round(float(r["y"]), 4), "z": round(float(r["z"]), 4),
} for r in rows]

positions = sorted({r["pos"] for r in out if r["pos"]})
seasons = sorted({r["season"] for r in out if r["season"]})
counts = {p: sum(1 for r in out if r["pos"] == p) for p in positions}

# ---- what a similarity means in this space ------------------------------------------------
_stride = max(1, n // 900)
_samp = list(range(0, n, _stride))
_unit = []
for i in _samp:
    m = math.sqrt(sum(x * x for x in V[i])) or 1.0
    _unit.append([x / m for x in V[i]])
_pairs = []
for a in range(len(_unit)):
    ra = _unit[a]
    for b in range(a + 1, len(_unit)):
        rb = _unit[b]
        _pairs.append(sum(ra[k] * rb[k] for k in range(d)))
_pairs.sort()


def _pct(q):
    return round(_pairs[min(len(_pairs) - 1, int(q / 100 * len(_pairs)))], 4)


cosine_dist = {"sample_rows": len(_samp), "n_pairs": len(_pairs), "median": _pct(50),
               "p75": _pct(75), "p95": _pct(95), "p99": _pct(99),
               "note": "cosine between two randomly chosen player-seasons in the 18-d v space, "
                       "over a deterministic stride sample"}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({
    "source": "assets/player_seasons.json",
    "built_utc": datetime.datetime.fromtimestamp(
        SRC.stat().st_mtime, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    "exporter_built_field": blob.get("built") if isinstance(blob, dict) else None,
    "attribution": blob.get("attribution") if isinstance(blob, dict) else None,
    "normalization": blob.get("normalization") if isinstance(blob, dict) else None,
    "n_rows": len(out), "positions": positions, "counts": counts, "seasons": seasons,
    "coords": ("PCA-3 of the 18-d per-game rate profile v, z-scored within season. Copied from "
               "the export and independently reproduced by scripts/build_gridiron_map.py at "
               f"|r| = {checks} on x/y/z."),
    "coords_are_not": ("a projection of the trained model's embedding. That is a separate "
                       "artifact, assets/embedding.json: MTNN v2 gated-fusion 32-d, PCA-3, "
                       "503 players, 2025 season only."),
    "explained_variance": explained,
    "cosine_distribution": cosine_dist,
    "rows": out,
}, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT.relative_to(ROOT)}: {len(out)} real rows, {len(positions)} positions, "
      f"{len(seasons)} seasons, {OUT.stat().st_size / 1024:.0f} KB")
print(f"  explained variance PC1-3: {explained}")
print(f"  random-pair cosine: median {cosine_dist['median']}, p99 {cosine_dist['p99']} "
      f"({cosine_dist['n_pairs']} pairs)")
for p in positions:
    print(f"    {p:<4s} {counts[p]:6d}")


def _dot(a, b):
    return sum(a[k] * b[k] for k in range(d))


unit = []
for i in range(n):
    m = math.sqrt(sum(x * x for x in V[i])) or 1.0
    unit.append([x / m for x in V[i]])

tiles = []
for p in positions:
    idx = [i for i, r in enumerate(out) if r["pos"] == p]
    cen = [sum(unit[i][k] for i in idx) / len(idx) for k in range(d)]
    cn = math.sqrt(sum(x * x for x in cen)) or 1.0
    cen = [x / cn for x in cen]
    chosen = [max(idx, key=lambda i: _dot(unit[i], cen))]
    while len(chosen) < PER_POS and len(chosen) < len(idx):
        chosen.append(max((i for i in idx if i not in chosen),
                          key=lambda i: min(1.0 - _dot(unit[i], unit[c]) for c in chosen)))
    for rank, i in enumerate(chosen):
        # Nearest from a DIFFERENT season. Same-season neighbours are close by construction
        # because v is z-scored within season, so they prove nothing.
        best_j, best = None, -2.0
        for j in range(n):
            if j == i or out[j]["season"] == out[i]["season"]:
                continue
            val = _dot(unit[i], unit[j])
            if val > best:
                best, best_j = val, j
        tiles.append({
            "name": out[i]["name"], "pos": out[i]["pos"], "team": out[i]["team"],
            "season": out[i]["season"], "games": out[i]["games"], "ppg": out[i]["ppg"],
            "x": out[i]["x"], "y": out[i]["y"],
            "pick": "most typical" if rank == 0 else f"spread {rank}",
            "nearest_other_season": {
                "name": out[best_j]["name"], "pos": out[best_j]["pos"],
                "team": out[best_j]["team"], "season": out[best_j]["season"],
                "cosine": round(best, 4),
            },
        })

ROSTER_OUT.write_text(json.dumps({
    "source": "assets/player_seasons.json",
    "selection": (f"per position, {PER_POS} player-seasons: the one nearest that position's mean "
                  "18-d profile, then farthest-point sampling so the tiles span the position "
                  "rather than resampling its centre"),
    "neighbour_rule": ("for each tile, the closest player-season from a DIFFERENT season by "
                       "cosine in the 18-d profile space. Same-season neighbours are close by "
                       "construction because v is z-scored within season."),
    "tiles": tiles,
}, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {ROSTER_OUT.relative_to(ROOT)}: {len(tiles)} tiles")
for t in tiles:
    nb = t["nearest_other_season"]
    print(f"    {t['pos']:<3s} {t['name'][:22]:<22s} {t['season']}  ->  "
          f"{nb['name'][:22]:<22s} {nb['season']}  cos {nb['cosine']:.3f}")
