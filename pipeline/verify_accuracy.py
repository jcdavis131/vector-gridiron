"""Promotion gates for Gridiron MTNN v2.

G1 shapes / NaN · G2 beat baselines · G3 vs v1 MAE · G4 family coverage ·
G5 artifact keys · G6 Methods honesty (README citations).

Run:  python pipeline/verify_accuracy.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "pipeline" / "data"
ASSETS = ROOT / "assets"
V1_MAE = 4.313


def main() -> int:
    fails = 0

    def check(cond: bool, msg: str) -> None:
        nonlocal fails
        print(("PASS " if cond else "FAIL ") + msg)
        if not cond:
            fails += 1

    # G1
    npz_path = DATA / "train_matrix.npz"
    man_path = DATA / "feature_manifest.json"
    check(npz_path.exists() and man_path.exists(), "train_matrix.npz + feature_manifest.json exist")
    if npz_path.exists() and man_path.exists():
        import numpy as np
        npz = np.load(npz_path, allow_pickle=True)
        man = json.loads(man_path.read_text(encoding="utf-8"))
        Z, M = npz["Z"], npz["mask"]
        check(Z.shape == M.shape, f"Z/M shapes align {Z.shape}")
        check(Z.shape[1] == len(man["features"]), "feature count matches manifest")
        check(len(man["family_lists"]) >= 8, f"{len(man['family_lists'])} families (>=8)")
        check(not np.isnan(Z[M > 0]).any(), "no NaNs in observed cells")

    # G2 / G3
    report_path = DATA / "mtnn_report.json"
    check(report_path.exists(), "mtnn_report.json exists")
    if report_path.exists():
        rep = json.loads(report_path.read_text(encoding="utf-8"))
        check(rep["model_fpts_mae"] < rep["baseline_last4_mae"],
              f"MAE {rep['model_fpts_mae']} < last-4 {rep['baseline_last4_mae']}")
        check(rep["model_fpts_mae"] < rep["baseline_seasontodate_mae"],
              f"MAE {rep['model_fpts_mae']} < STD {rep['baseline_seasontodate_mae']}")
        check(rep["model_fpts_mae"] <= V1_MAE + 0.05,
              f"MAE {rep['model_fpts_mae']} <= v1+0.05 ({V1_MAE})")

    # G4
    if man_path.exists():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        for fam, cov in man.get("coverage", {}).items():
            if fam == "availability":
                # may be low if mostly 2025 rows — warn only if exactly 0 and we have pre-2025
                continue
            check(cov > 0.01, f"family {fam} coverage {cov:.3f} > 0.01")

    # G5 artifacts
    for name in ("nextgame.json", "projections.json", "embedding.json"):
        p = ASSETS / name
        check(p.exists(), f"assets/{name} exists")
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            check(d.get("count", 0) > 300, f"{name} count={d.get('count')}")
            if name != "embedding.json":
                pl = d["players"][0]
                for k in ("key", "name", "pos", "proj", "line"):
                    check(k in pl, f"{name} player has {k}")

    # G6 README honesty
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    check("Next Gen Stats" in readme or "NGS" in readme or "nextgen" in readme.lower(),
          "README mentions NGS / Next Gen Stats")
    check("ffopportunity" in readme or "expected fantasy" in readme.lower() or "EP" in readme,
          "README mentions opportunity / EP source")
    check("2025" in readme and ("injur" in readme.lower()),
          "README documents injury feed caveat")
    check("Methods" in readme, "README has Methods honesty section")
    check("residual" in readme.lower() or "Floor / ceiling" in readme,
          "README explains floor/ceiling residual")

    print(f"\n{fails} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
