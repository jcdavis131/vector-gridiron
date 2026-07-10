"""Drop-one family ablation for Gridiron MTNN v2.

Loads train_matrix.npz + mtnn_best.pt, evaluates held-out test PPR MAE with
each family zero-masked, writes ΔMAE into mtnn_report.json["family_ablation"].

Run:  python pipeline/family_ablation.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

import build_features as bf
from train_mtnn import MTNN, family_slices, split_by_family

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "pipeline" / "data"


def main() -> int:
    t0 = time.time()
    npz = np.load(DATA / "train_matrix.npz", allow_pickle=True)
    man = json.loads((DATA / "feature_manifest.json").read_text(encoding="utf-8"))
    ckpt = torch.load(DATA / "mtnn_best.pt", map_location="cpu", weights_only=False)

    Z, M = npz["Z"].astype(np.float32), npz["mask"].astype(np.float32)
    Y = npz["Y"].astype(np.float32)
    seasons = npz["season"].astype(np.int32)
    feats = man["features"]
    families = man["family_lists"]

    te = seasons == int(ckpt.get("test_season", 2025))
    if not te.any():
        te = seasons == int(seasons.max())
    mu = np.asarray(ckpt["mu"], dtype=np.float32)
    sd = np.asarray(ckpt["sd"], dtype=np.float32)
    ymu = np.asarray(ckpt["ymu"], dtype=np.float32)
    ysd = np.asarray(ckpt["ysd"], dtype=np.float32)
    sd = np.where(sd == 0, 1.0, sd)

    Xz = ((Z - mu) / sd) * M
    slices = family_slices(feats, families)
    fam_dims = {fam: len(cols) for fam, cols in slices.items()}
    n_seasons = int(ckpt.get("n_seasons", seasons.max() - seasons.min() + 1))
    season_min = int(ckpt.get("season_min", seasons.min()))

    device = torch.device("cpu")
    model = MTNN(fam_dims, n_seasons=n_seasons, d_emb=int(ckpt.get("d_emb", 32)),
                 n_targets=Y.shape[1], n_usage=3).to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()

    te_idx = np.where(te)[0]
    y_te = Y[te_idx, 0]
    seas_te = torch.tensor(seasons[te_idx] - season_min, dtype=torch.long).clamp(0, n_seasons - 1)

    def mae_with_mask(M_use: np.ndarray) -> float:
        Xrz = ((Z[te_idx] - mu) / sd) * M_use[te_idx]
        xs, ms = split_by_family(Xrz, M_use[te_idx], slices, device)
        with torch.no_grad():
            _, out = model(xs, ms, seas_te)
            pred = out["targets"].numpy() * ysd + ymu
        return float(np.mean(np.abs(pred[:, 0] - y_te)))

    base = mae_with_mask(M)
    print(f"baseline test PPR MAE = {base:.4f}  (n={len(te_idx)})")

    ablation = {"baseline_mae": round(base, 4), "families": {}}
    for fam, cols in families.items():
        idxs = [feats.index(c) for c in cols if c in feats]
        if not idxs:
            continue
        M2 = M.copy()
        M2[:, idxs] = 0.0
        mae = mae_with_mask(M2)
        delta = mae - base
        ablation["families"][fam] = {
            "mae": round(mae, 4),
            "delta_mae": round(delta, 4),
            "n_cols": len(idxs),
        }
        flag = "HURTS" if delta > 0.02 else ("helps?" if delta < -0.01 else "flat")
        print(f"  drop {fam:14s}  MAE {mae:.4f}  dMAE {delta:+.4f}  [{flag}]")

    # rank by how much MAE rises when dropped (importance)
    ranked = sorted(ablation["families"].items(), key=lambda kv: -kv[1]["delta_mae"])
    ablation["importance_rank"] = [f for f, _ in ranked]
    ablation["note"] = (
        "Positive ΔMAE = family helps (MAE rises when dropped). "
        "Negative ΔMAE = family may hurt or be redundant."
    )

    report_path = DATA / "mtnn_report.json"
    rep = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    rep["family_ablation"] = ablation
    report_path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"wrote {report_path}  ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
