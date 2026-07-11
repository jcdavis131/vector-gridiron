"""Promotion gates for Gridiron MTNN v2.

G1 shapes / NaN · G2 beat baselines · G3 vs v1 MAE · G4 family coverage ·
G5 artifact keys · G6 Methods honesty · G7 composite CQS · G8 all 6 positions.

Run:  python pipeline/verify_accuracy.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "pipeline" / "data"
ASSETS = ROOT / "assets"
V1_MAE = 4.313
sys.path.insert(0, str(ROOT / "pipeline"))
import composite_score as cqs  # noqa: E402

ALL_POS = ("QB", "RB", "WR", "TE", "K", "DST")


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
    rep = None
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
    check("Composite" in readme or "CQS" in readme,
          "README documents composite CQS promote gate")

    # G7 composite
    if rep is not None:
        block = rep.get("composite") or cqs.composite_quality(rep)
        check("cqs" in block, f"composite.cqs present ({block.get('cqs')})")
        check(float(block.get("cqs") or 0) >= 50.0,
              f"CQS {block.get('cqs')} >= 50 (sanity floor)")
        check(rep.get("promote_metric") == "cqs",
              f"promote_metric={rep.get('promote_metric')} (want cqs)")
        for pos in ("QB", "RB", "WR", "TE"):
            check(pos in (rep.get("per_pos_mae") or {}),
                  f"per_pos_mae has {pos}")

    # G8 all six fantasy positions on boards + kdst holdout
    kdst_path = ASSETS / "kdst.json"
    check(kdst_path.exists(), "assets/kdst.json exists")
    if kdst_path.exists():
        kdst = json.loads(kdst_path.read_text(encoding="utf-8"))
        check(len(kdst.get("kickers") or []) >= 30,
              f"kickers {len(kdst.get('kickers') or [])} >= 30")
        check(len(kdst.get("dst") or []) == 32,
              f"DST {len(kdst.get('dst') or [])} == 32")
        hold = kdst.get("holdout") or {}
        check(hold.get("kicker_mae") is not None,
              f"kdst holdout kicker_mae={hold.get('kicker_mae')}")
        check(hold.get("dst_mae") is not None,
              f"kdst holdout dst_mae={hold.get('dst_mae')}")
    for name in ("nextgame.json", "projections.json"):
        p = ASSETS / name
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        pos_set = {pl.get("pos") for pl in d.get("players") or []}
        for pos in ALL_POS:
            check(pos in pos_set, f"{name} includes position {pos}")

    print(f"\n{fails} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
