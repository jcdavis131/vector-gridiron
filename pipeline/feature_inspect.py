"""Feature coverage / correlation inspection for MTNN v2 matrix.

Run:  python pipeline/feature_inspect.py
Reads pipeline/data/train_matrix.npz + feature_manifest.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "pipeline" / "data"


def main() -> int:
    npz = np.load(DATA / "train_matrix.npz", allow_pickle=True)
    manifest = json.loads((DATA / "feature_manifest.json").read_text(encoding="utf-8"))
    Z, M = npz["Z"], npz["mask"]
    feats = manifest["features"]
    fams = manifest["family_lists"]
    print(f"matrix {Z.shape}, seasons {manifest['seasons']}")
    print("\nFamily coverage (mean mask):")
    report = {"families": {}, "flags": []}
    for fam, cols in fams.items():
        idx = [feats.index(c) for c in cols]
        cov = float(M[:, idx].mean())
        report["families"][fam] = {
            "coverage": round(cov, 4),
            "n_features": len(cols),
            "mean_abs": round(float(np.abs(Z[:, idx][M[:, idx] > 0]).mean()) if (M[:, idx] > 0).any() else 0, 4),
        }
        print(f"  {fam:14s}  cov={cov:.3f}  n={len(cols)}")
        if cov < 0.05:
            report["flags"].append(f"{fam}: near-empty coverage {cov:.3f}")
        if cov == 0:
            report["flags"].append(f"{fam}: ZERO coverage")

    # pairwise corr within form family (redundancy sniff)
    form_idx = [feats.index(c) for c in fams["form"]]
    sub = Z[:, form_idx]
    msub = M[:, form_idx]
    # only rows with full form mask
    ok = msub.min(axis=1) > 0
    if ok.sum() > 100:
        C = np.corrcoef(sub[ok].T)
        high = []
        for i in range(len(form_idx)):
            for j in range(i + 1, len(form_idx)):
                if abs(C[i, j]) > 0.92:
                    high.append((fams["form"][i], fams["form"][j], round(float(C[i, j]), 3)))
        report["high_corr_form"] = high[:20]
        print(f"\nHigh |r|>0.92 within form: {len(high)} pairs")
        for a, b, r in high[:8]:
            print(f"  {a} ~ {b}: {r}")

    if np.isnan(Z[M > 0]).any():
        report["flags"].append("NaNs in observed cells")
        print("FAIL: NaNs in observed cells")
    else:
        print("\nOK: no NaNs in observed cells")

    (DATA / "feature_inspect.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {DATA / 'feature_inspect.json'}")
    return 1 if any("ZERO" in f for f in report["flags"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
