"""Promote val-fit per-pos affine on top of bias shrink onto report + boards."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_mtnn import MTNN, SKILL, family_slices, split_by_family
import composite_score as cqs

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "pipeline" / "data"
ASSETS = ROOT / "assets"


def main() -> int:
    npz = np.load(DATA / "train_matrix.npz", allow_pickle=True)
    man = json.loads((DATA / "feature_manifest.json").read_text(encoding="utf-8"))
    ckpt = torch.load(DATA / "mtnn_best.pt", map_location="cpu", weights_only=False)
    Z, M = npz["Z"].astype(np.float32), npz["mask"].astype(np.float32)
    Y = npz["Y"].astype(np.float32)
    seasons = npz["season"].astype(np.int32)
    pos = npz["pos"]
    feats, families = man["features"], man["family_lists"]
    mu = np.asarray(ckpt["mu"], np.float32)
    sd = np.where(np.asarray(ckpt["sd"], np.float32) == 0, 1.0, np.asarray(ckpt["sd"], np.float32))
    ymu = np.asarray(ckpt["ymu"], np.float32)
    ysd = np.asarray(ckpt["ysd"], np.float32)
    n_seasons = int(ckpt.get("n_seasons", seasons.max() - seasons.min() + 1))
    season_min = int(ckpt.get("season_min", seasons.min()))
    slices = family_slices(feats, families)
    model = MTNN(
        {f: len(c) for f, c in slices.items()},
        n_seasons=n_seasons,
        d_emb=int(ckpt.get("d_emb", 32)),
        n_targets=Y.shape[1],
        n_usage=3,
    )
    model.load_state_dict(ckpt["state"])
    model.eval()

    old = json.loads((DATA / "mtnn_report.json").read_text(encoding="utf-8"))
    bc = old.get("bias_calib") or {}
    alpha = float(bc.get("alpha", 0.5))
    b_pos = {k: float(v) for k, v in (bc.get("per_pos") or {}).items()}
    b_g = float(bc.get("global", 0.0))

    def predict_raw(mask):
        idx = np.where(mask)[0]
        Xrz = ((Z[idx] - mu) / sd) * M[idx]
        xs, ms = split_by_family(Xrz, M[idx], slices, torch.device("cpu"))
        sid = torch.tensor(seasons[idx] - season_min, dtype=torch.long).clamp(0, n_seasons - 1)
        with torch.no_grad():
            _, out = model(xs, ms, sid)
            p = out["targets"].numpy() * ysd + ymu
        return p[:, 0], Y[idx, 0], pos[idx]

    def apply_bias(p, pos_arr):
        return np.array([p[i] + alpha * b_pos.get(pos_arr[i], b_g) for i in range(len(p))])

    def fit_affine(y, p, pos_arr):
        params = {}
        for q in SKILL:
            m = pos_arr == q
            if m.sum() < 30:
                continue
            A = np.column_stack([p[m], np.ones(m.sum())])
            coef, *_ = np.linalg.lstsq(A, y[m], rcond=None)
            params[q] = (float(coef[0]), float(coef[1]))
        return params

    def apply_affine(p, pos_arr, params, mix=1.0):
        out = p.copy()
        for i, q in enumerate(pos_arr):
            if q not in params:
                continue
            a, b = params[q]
            out[i] = (1 - mix) * p[i] + mix * (a * p[i] + b)
        return out

    def metrics(y, p, pos_arr):
        abs_e = np.abs(y - p)
        per = {
            q: round(float(abs_e[pos_arr == q].mean()), 3)
            for q in SKILL
            if (pos_arr == q).sum() >= 30
        }
        conf = {
            q: float(np.quantile(abs_e[pos_arr == q], 0.8))
            for q in SKILL
            if (pos_arr == q).sum() >= 30
        }
        gq = float(np.quantile(abs_e, 0.8))
        qrow = np.array([conf.get(pos_arr[i], gq) for i in range(len(y))])
        mape_denom = np.maximum(np.abs(y), 1.0)
        r = {
            "model_fpts_mae": round(float(abs_e.mean()), 3),
            "model_fpts_rmse": round(float(np.sqrt(((y - p) ** 2).mean())), 3),
            "model_fpts_r2": round(
                float(1 - np.sum((y - p) ** 2) / max(1e-9, np.sum((y - y.mean()) ** 2))), 3
            ),
            "model_fpts_medae": round(float(np.median(abs_e)), 3),
            "model_fpts_mape": round(float(np.mean(abs_e / mape_denom)), 3),
            "model_fpts_bias": round(float((p - y).mean()), 3),
            "per_pos_mae": per,
            "conformal_coverage": round(float(np.mean(abs_e <= qrow)), 3),
            "conformal_q": round(gq, 3),
            "conformal_q_by_pos": {k: round(v, 3) for k, v in conf.items()},
            "conformal_level": 0.8,
            "baseline_last4_mae": old["baseline_last4_mae"],
            "baseline_seasontodate_mae": old["baseline_seasontodate_mae"],
            "n_test": old.get("n_test"),
            "n_train": old.get("n_train"),
            "test_season": old.get("test_season"),
            "n_features": old.get("n_features"),
            "n_families": old.get("n_families"),
            "n_params": old.get("n_params"),
            "val_mae": old.get("val_mae"),
            "v1_mae_reference": old.get("v1_mae_reference"),
            "family_drop": old.get("family_drop"),
            "d_emb": old.get("d_emb"),
            "architecture": old.get("architecture"),
            "per_stat_mae": old.get("per_stat_mae"),
            "residual_std": old.get("residual_std"),
            "residual_std_by_pos": old.get("residual_std_by_pos"),
        }
        r["composite"] = cqs.composite_quality(r)
        return r

    pv, yv, posv = predict_raw(seasons == 2024)
    pt, yt, post = predict_raw(seasons == 2025)
    pv_b, pt_b = apply_bias(pv, posv), apply_bias(pt, post)
    params = fit_affine(yv, pv_b, posv)

    base_va = metrics(yv, pv_b, posv)
    best_mix, best_va_cqs = 0.0, base_va["composite"]["cqs"]
    for i in range(0, 21):
        mix = i / 20
        m = metrics(yv, apply_affine(pv_b, posv, params, mix), posv)
        if m["model_fpts_mae"] <= base_va["model_fpts_mae"] + 0.05 and m["composite"]["cqs"] >= best_va_cqs:
            best_mix, best_va_cqs = mix, m["composite"]["cqs"]

    te = metrics(yt, apply_affine(pt_b, post, params, best_mix), post)
    ok, why = cqs.should_promote({**te, "composite": te["composite"]})
    print(f"mix={best_mix} MAE={te['model_fpts_mae']} CQS={te['composite']['cqs']} {why}")
    if not ok:
        return 1

    te["bias_calib"] = bc
    te["affine_calib"] = {
        "mix": best_mix,
        "per_pos": {k: {"a": round(a, 4), "b": round(b, 4)} for k, (a, b) in params.items()},
        "fit": "val_on_bias_calibrated",
        "select": "max_val_cqs_mae_slack_0.05",
    }
    te["promote_metric"] = "cqs"
    te["hill_climb"] = "rz+conformal80+cqs+bias_shrink+affine"
    te["metrics_note"] = "Promote gate = CQS; MAE soft floor. Bias shrink + per-pos affine."
    te["composite"]["baseline_cqs"] = cqs.BASELINE["cqs"]
    te["composite"]["would_promote_vs_baseline"] = True
    te["composite"]["promote_check"] = why
    if "family_ablation" in old:
        te["family_ablation"] = old["family_ablation"]
    (DATA / "mtnn_report.json").write_text(json.dumps(te, indent=2), encoding="utf-8")

    # Boards already have bias; apply affine only (idempotent flag)
    for name in ("nextgame.json", "projections.json"):
        p = ASSETS / name
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("affine_calib_applied"):
            print(f"{name} already affine — skip")
            continue
        for pl in d["players"]:
            if pl.get("source") == "kdst" or pl.get("pos") in ("K", "DST"):
                continue
            ab = params.get(pl["pos"])
            if not ab:
                continue
            a, b = ab
            pl["proj"] = round((1 - best_mix) * float(pl["proj"]) + best_mix * (a * float(pl["proj"]) + b), 2)
            band = float(pl.get("uncertainty") or te["conformal_q_by_pos"].get(pl["pos"], te["conformal_q"]))
            pl["floor"] = round(max(0.0, pl["proj"] - band), 2)
            pl["ceil"] = round(pl["proj"] + band, 2)
        if "model" in d and isinstance(d["model"], dict):
            d["model"]["report"] = te
        d["affine_calib_applied"] = {
            "mix": best_mix,
            "per_pos": te["affine_calib"]["per_pos"],
            "built": time.strftime("%Y-%m-%d"),
        }
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d, separators=(",", ":")), encoding="utf-8")
        tmp.replace(p)
        print(f"updated {name}")

    # Update BASELINE in composite_score.py
    cs = (Path(__file__).resolve().parent / "composite_score.py").read_text(encoding="utf-8")
    cs2 = cs
    # replace mae/cqs numbers in BASELINE block
    import re
    cs2 = re.sub(
        r'BASELINE = \{[^}]+\}',
        f'BASELINE = {{\n    "mae": {te["model_fpts_mae"]},\n    "cqs": {te["composite"]["cqs"]},  # bias+affine\n}}',
        cs,
        count=1,
    )
    (Path(__file__).resolve().parent / "composite_score.py").write_text(cs2, encoding="utf-8")
    print("PROMOTED", te["model_fpts_mae"], te["composite"]["cqs"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
