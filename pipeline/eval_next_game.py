"""
Eval next-game MAE / R2 for Vector Gridiron.

Computes MAE next-game and R2 from vectors.json or predictions CSV and
documents MAE 4.268 claimed vs reproducible.

Inputs (one of):
  - assets/vectors.json with fpts_next_pred / fpts_next_true or fpts_pred array
  - assets/eval_scoreboard.json (already has claimed)
  - pipeline/data/embedding_gridiron.npz with fpts_pred / fpts_true
  - pipeline/data/train_matrix.npz predictions CSV (pred.csv with columns true,pred)

Usage:
  python pipeline/eval_next_game.py
  python pipeline/eval_next_game.py --vectors assets/vectors.json
  python pipeline/eval_next_game.py --npz pipeline/data/embedding_gridiron.npz
  python pipeline/eval_next_game.py --csv predictions.csv

Outputs:
  - assets/eval_scoreboard.json updated with reproducible MAE + R2
  - prints MAE claimed vs repro

Claimed:
  MAE 4.268 R2 0.39 — from README / dashboard.html offline train noted as
  claimed_not_reproducible when no train pipeline checked in. Now that
  train_mtnn.py exists, reproducible MAE becomes computable once nflverse fetch lands.

Target: MAE 4.268 → 3.8 with Procrustes + RealMLP + MoE + TabPFN distill KL T=2 w=0.15
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
DATA_DIR = ROOT / "pipeline" / "data"


def mae(a, b):
    return float(np.mean(np.abs(np.array(a) - np.array(b))))


def r2_score(y_true, y_pred):
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum() + 1e-9)
    return float(1 - ss_res / ss_tot)


def eval_from_vectors(path: Path):
    data = json.loads(path.read_text())
    preds, trues = [], []
    # support multiple shapes
    if "players" in data:
        for p in data["players"]:
            if "fpts_next_pred" in p and "fpts_next_true" in p:
                preds.append(p["fpts_next_pred"])
                trues.append(p["fpts_next_true"])
            elif "fpts_pred" in p and "fpts_true" in p:
                preds.append(p["fpts_pred"])
                trues.append(p["fpts_true"])
    if not preds and "fpts_pred" in data and "fpts_true" in data:
        preds = data["fpts_pred"]
        trues = data["fpts_true"]
    if not preds:
        # maybe embedding npz style json not vectors
        return None
    return {
        "mae": mae(preds, trues),
        "r2": r2_score(trues, preds),
        "n": len(preds),
    }


def eval_from_npz(path: Path):
    npz = np.load(path, allow_pickle=True)
    pred_key = "fpts_pred" if "fpts_pred" in npz else "fpts_next_pred" if "fpts_next_pred" in npz else None
    true_key = "fpts_true" if "fpts_true" in npz else "fpts_next_true" if "fpts_next_true" in npz else None
    if pred_key is None or true_key is None:
        # fallback to E embeddings only — no eval possible
        return None
    preds = npz[pred_key]
    trues = npz[true_key]
    return {"mae": mae(trues, preds), "r2": r2_score(trues, preds), "n": len(preds)}


def eval_from_csv(path: Path):
    import csv

    preds, trues = [], []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                # flexible column names
                p_col = row.get("pred") or row.get("fpts_pred") or row.get("fpts_next_pred")
                t_col = row.get("true") or row.get("fpts_true") or row.get("fpts_next_true")
                if p_col is None or t_col is None:
                    continue
                preds.append(float(p_col))
                trues.append(float(t_col))
            except:
                continue
    if not preds:
        return None
    return {"mae": mae(trues, preds), "r2": r2_score(trues, preds), "n": len(preds)}


def write_scoreboard(eval_res, source_path):
    out_path = ASSETS / "eval_scoreboard.json"
    existing = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
        except:
            existing = {}

    scoreboard = {
        "built": time.strftime("%Y-%m-%d"),
        "claimed_MAE_next_game": 4.268,
        "claimed_R2": 0.39,
        "claimed_status": "claimed_not_reproducible_offline_train_missing (now train_mtnn.py enables repro)",
        "current_repro": eval_res,
        "current_source": str(source_path) if source_path else "none",
        "metrics": {
            "MAE_next_game": eval_res["mae"] if eval_res else 4.268,
            "R2": eval_res["r2"] if eval_res else 0.39,
            "n": eval_res["n"] if eval_res else None,
        },
        "embedding_dim_code": 32,
        "embedding_dim_legacy": 16,
        "embedding_dim_advertised": 32,
        "embedding_dim_native": 32,
        "embedding_dim_backward_compat": 16,
        "target": "MAE 4.268→3.8 with Procrustes+RealMLP+MoE + TabPFN distill KL T=2 w=0.15",
        "note": "new train_mtnn.py enables repro — run nflverse fetch to get MAE 4.268→3.8 target. 32-d native is primary, 16-d slice+re-L2 legacy for app.js bundle <300KB.",
        "architecture_v2": {
            "d_emb": 32,
            "legacy_16d": "slice first 16 dims re-L2",
            "transformer": "d_model128 n_heads4 n_layers4 CLS→32-d L2",
            "towers": "10 families holistic 160 feats ResidualTower cat([x*m,m]) d_cat*2→96h GELU LN →24d + skip",
            "procrustes": "rotation-only orthogonal Procrustes Q chains season→root drift (hoops same)",
            "realmlp": "per-season RobustScaler median/IQR clip[-3,3] PL embedding sin/cos k=8 d_out16 proj linear 2k→16",
        },
    }
    # merge existing extra keys we care about (preserve claimed_not_reproducible flag transition)
    if existing.get("claimed_status"):
        # upgrade note about new pipeline
        pass

    out_path.write_text(json.dumps(scoreboard, indent=2), encoding="utf-8")
    print(f"[eval] wrote {out_path}")
    print(json.dumps(scoreboard, indent=2))


def main():
    ap = argparse.ArgumentParser(description="Eval next-game MAE R2 for gridiron")
    ap.add_argument("--vectors", type=str, default=str(ASSETS / "vectors.json"), help="vectors.json path")
    ap.add_argument("--npz", type=str, default=str(DATA_DIR / "embedding_gridiron.npz"), help="embedding npz path")
    ap.add_argument("--csv", type=str, default="", help="optional csv predictions")
    ap.add_argument("--scoreboard-only", action="store_true", help="write scoreboard with claimed only if no eval available")
    args = ap.parse_args()

    def try_path(fn, path_str):
        p = Path(path_str)
        if p.exists():
            res = fn(p)
            if res:
                print(f"[eval] from {p}: MAE {res['mae']:.4f} R2 {res['r2']:.4f} n={res['n']}")
                write_scoreboard(res, p)
                return res
        return None

    # priority: csv > npz > vectors
    if args.csv:
        r = try_path(eval_from_csv, args.csv)
        if r:
            return 0
    r = try_path(eval_from_npz, args.npz)
    if r:
        return 0
    r = try_path(eval_from_vectors, args.vectors)
    if r:
        return 0

    # none available — write claimed_not_reproducible + pointer to new train
    print("[eval] No repro predictions found (assets/vectors.json / embedding_gridiron.npz / csv missing)")
    print("[eval] Claimed MAE 4.268 R2 0.39 — not yet reproducible until nflverse fetch")
    print("[eval] Run: python pipeline/train_mtnn.py --synthetic for smoke, or real fetch per docs/DATA_SOURCES.md")
    eval_res_none = None
    # still write scoreboard with claimed
    out_path = ASSETS / "eval_scoreboard.json"
    # if exists keep but mark transition
    existing = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
        except:
            pass
    scoreboard = {
        "built": time.strftime("%Y-%m-%d"),
        "claimed_MAE_next_game": 4.268,
        "claimed_R2": 0.39,
        "claimed_status": "claimed_not_reproducible_offline_train_missing",
        "reproducibility_gate": "new train_mtnn.py enables repro — run nflverse fetch to get MAE 4.268→3.8 target",
        "current_repro": None,
        "metrics": {
            "MAE_next_game": 4.268,
            "R2": 0.39,
            "n": None,
            "status": "claimed",
        },
        "embedding_dim_code": 32,
        "embedding_dim_legacy": 16,
        "embedding_dim_advertised": 32,
        "embedding_dim_native": 32,
        "target": "MAE 4.268→3.8 with Procrustes+RealMLP+MoE + TabPFN distill KL T=2 w=0.15",
        "note": "new train_mtnn.py enables repro — run nflverse fetch to get MAE 4.268→3.8 target. 32-d native primary, 16-d slice+re-L2 legacy for bundle <300KB gz.",
        "architecture_v2": {
            "d_emb": 32,
            "legacy_16d": "slice first 16 dims re-L2",
            "transformer": "d_model128 n_heads4 n_layers4 CLS→32-d L2",
            "towers": "10 families holistic 160 feats ResidualTower cat([x*m,m]) d_cat*2→96h GELU LN →24d + skip",
            "procrustes": "rotation-only orthogonal Procrustes Q chains season→root drift (hoops same)",
            "realmlp": "per-season RobustScaler median/IQR clip[-3,3] PL embedding sin/cos k=8 d_out16 proj linear 2k→16",
        },
        "plan_for_nflverse_fetch": {
            "source": "nflverse 2025 play-by-play roster weather Vegas",
            "files": [
                "nflverse play-by-play 2025",
                "roster 2025",
                "weather wind temp dome",
                "Vegas lines spread total",
                "injury/rest def-vs-pos form lag",
            ],
            "script": "pipeline/fetch_nflverse.py (to be created) → pipeline/data/train_matrix.npz",
            "fetch_steps": [
                "nflreadpy or nfl_data_py load_pbp seasons=[2020..2025]",
                "nflreadpy weekly roster, snap counts, participation",
                "weather API (Open-Meteo) join wind/temp/dome",
                "Vegas lines scrape / nflverse betting data join spread/total/implied",
                "compute lag 1-3 fantasy points PPR, form rolling avg, redzone usage, def-vs-pos SOS_NET analog",
                "build 10 families holistic 160 feats, per-season RobustScaler fit, emit train_matrix.npz X [N,160] M mask Y next-game FPTS",
                "player-split honest, no season-split leakage",
            ],
            "command_target": "python pipeline/train_mtnn.py --epochs 50 --d-emb 32 --scaling robust --era-align procrustes → MAE 4.268→3.8",
        },
    }
    if existing:
        # preserve best known MAE if better?
        scoreboard["_prev"] = {k: existing.get(k) for k in ["best_val_MAE", "claimed_MAE_next_game", "target"] if k in existing}
    out_path.write_text(json.dumps(scoreboard, indent=2), encoding="utf-8")
    print(f"[eval] wrote {out_path} (claimed only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
