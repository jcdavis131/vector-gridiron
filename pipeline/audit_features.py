"""Audit the gridiron MTNN input matrix -- same discipline as vector-hoops'
pipeline/audit_features.py, adapted for this repo's schema (weekly rows,
season is already int, targets live in a separate Y array rather than as an
input family).

* dead / near-constant columns -- a feature that never varies is a bias term
  wearing a feature's name,
* redundant pairs (|r| >= 0.98 on commonly-observed rows) -- duplicated input
  signal inflates a family's apparent width and its fusion share,
* leak candidates -- an INPUT column correlating ~1.0 with a real prediction
  TARGET (Y's fpts_ppr/rec_yds/rush_yds/pass_yds/receptions/total_td) is the
  gridiron equivalent of hoops' family-fed-its-own-head leak, just checked
  against Y directly since gridiron's targets aren't mixed into Z,
* coverage cliffs at the newest season, the boundary any live game depends on.

Read-only. Writes pipeline/data/feature_audit.json.

Run:  python pipeline/audit_features.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "pipeline" / "data"
MATRIX = DATA / "train_matrix.npz"
MANIFEST = DATA / "feature_manifest.json"
OUT = DATA / "feature_audit.json"

DUP_R = 0.98
LEAK_R = 0.90  # weekly-row targets are noisier than hoops' season targets; keep the bar honest but not hair-trigger
MIN_OVERLAP = 400
NEAR_CONST_STD = 0.01


def masked_corr(a, b, ma, mb) -> tuple[float, int]:
    both = (ma > 0) & (mb > 0)
    n = int(both.sum())
    if n < MIN_OVERLAP:
        return 0.0, n
    x, y = a[both].astype(np.float64), b[both].astype(np.float64)
    sx, sy = x.std(), y.std()
    if sx < 1e-9 or sy < 1e-9:
        return 0.0, n
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy)), n


def main() -> None:
    m = np.load(MATRIX, allow_pickle=True)
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    Z, M = m["Z"], m["mask"]
    Y = m["Y"] if "Y" in m else None
    Y_usage = m["Y_usage"] if "Y_usage" in m else None
    targets: list[str] = man.get("targets", [])
    feats: list[str] = man["features"]
    fam_of: dict[str, str] = man["families"]
    yr = np.asarray(m["season"]).astype(int)

    fam_cols: dict[str, list[int]] = defaultdict(list)
    for j, f in enumerate(feats):
        fam_cols[fam_of.get(f, "?")].append(j)

    report: dict = {"rows": int(Z.shape[0]), "features": len(feats), "families": len(fam_cols)}

    # 1. dead / near-constant on observed rows
    dead = []
    for j, f in enumerate(feats):
        obs = M[:, j] > 0
        cov = float(obs.mean())
        if obs.sum() < MIN_OVERLAP:
            dead.append({"feature": f, "family": fam_of.get(f), "coverage": round(cov, 4),
                         "why": "coverage below usable threshold"})
            continue
        sd = float(Z[obs, j].std())
        if sd < NEAR_CONST_STD:
            dead.append({"feature": f, "family": fam_of.get(f), "coverage": round(cov, 4),
                         "sd": round(sd, 6), "why": "near-constant where observed"})
    report["dead_or_constant"] = dead

    # 2. redundant pairs
    dups = []
    for j in range(len(feats)):
        for k in range(j + 1, len(feats)):
            r, n = masked_corr(Z[:, j], Z[:, k], M[:, j], M[:, k])
            if abs(r) >= DUP_R:
                dups.append({"a": feats[j], "b": feats[k], "family_a": fam_of.get(feats[j]),
                             "family_b": fam_of.get(feats[k]), "r": round(r, 4), "n": n})
    dups.sort(key=lambda d: -abs(d["r"]))
    report["redundant_pairs"] = dups

    # 3. leak candidates: input column ~ a real Y target. Y has no NaNs (every
    # row is a played week), so every row counts as observed for the target
    # side; Y_usage is a usage-share vector (pass/rush/rec fractions), not a
    # per-target validity mask, so it isn't used here.
    leaks = []
    if Y is not None and targets:
        tmask_all = np.ones(Y.shape[0], dtype=np.float32)
        for ti, tname in enumerate(targets):
            tcol = Y[:, ti]
            for k, f in enumerate(feats):
                r, n = masked_corr(tcol, Z[:, k], tmask_all, M[:, k])
                if abs(r) >= LEAK_R:
                    leaks.append({"target": tname, "input": f, "input_family": fam_of.get(f),
                                 "r": round(r, 4), "n": n})
    leaks.sort(key=lambda d: -abs(d["r"]))
    report["leak_candidates"] = leaks

    # 4. coverage across the newest-season boundary the live game depends on
    newest = int(yr.max())
    eras = {"older": yr < newest - 1, f"{newest-1}_only": yr == newest - 1, f"{newest}_newest": yr == newest}
    fam_cov = {}
    cliffs = []
    for fam, cols in sorted(fam_cols.items()):
        cov = {name: round(float(M[msk][:, cols].mean()), 4) if msk.any() else None
               for name, msk in eras.items()}
        fam_cov[fam] = cov
        older = cov["older"] or 0.0
        newest_cov = cov[f"{newest}_newest"]
        if older > 0.15 and newest_cov is not None and newest_cov < older * 0.5:
            cliffs.append({"family": fam, **cov})
    report["family_coverage_by_era"] = fam_cov
    report["coverage_cliffs_newest"] = cliffs
    report["newest_season"] = newest

    # 5. within-family redundancy
    fam_red = {}
    for fam, cols in sorted(fam_cols.items()):
        if len(cols) < 2:
            continue
        rs = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r, n = masked_corr(Z[:, cols[i]], Z[:, cols[j]], M[:, cols[i]], M[:, cols[j]])
                if n >= MIN_OVERLAP:
                    rs.append(abs(r))
        if rs:
            fam_red[fam] = {"n_features": len(cols), "mean_abs_r": round(float(np.mean(rs)), 4),
                            "max_abs_r": round(float(np.max(rs)), 4)}
    report["within_family_redundancy"] = fam_red

    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"rows={report['rows']} features={report['features']} families={report['families']}")
    print(f"dead/near-constant: {len(dead)}")
    for d in dead[:10]:
        print(f"  {d['feature']:26s} {d.get('why')}")
    print(f"redundant pairs |r|>={DUP_R}: {len(dups)}")
    for d in dups[:15]:
        print(f"  {d['a']:26s} ~ {d['b']:26s} r={d['r']:+.4f} ({d['family_a']}/{d['family_b']})")
    print(f"leak candidates |r|>={LEAK_R} (input vs real target): {len(leaks)}")
    for d in leaks[:10]:
        print(f"  target {d['target']:12s} <- {d['input']:24s} r={d['r']:+.4f} [{d['input_family']}]")
    print(f"coverage cliffs at newest season ({newest}): {len(cliffs)}")
    for c in cliffs:
        print(f"  {c['family']}: {c}")
    print("most redundant families (mean |r|):")
    for fam, v in sorted(fam_red.items(), key=lambda kv: -kv[1]["mean_abs_r"])[:8]:
        print(f"  {fam:14s} n={v['n_features']:2d} mean|r|={v['mean_abs_r']:.3f} max={v['max_abs_r']:.3f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
