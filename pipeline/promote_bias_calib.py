"""Promote val-fit bias shrink (α from val) onto live report + boards."""
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

    def predict(mask):
        idx = np.where(mask)[0]
        Xrz = ((Z[idx] - mu) / sd) * M[idx]
        xs, ms = split_by_family(Xrz, M[idx], slices, torch.device("cpu"))
        sid = torch.tensor(seasons[idx] - season_min, dtype=torch.long).clamp(0, n_seasons - 1)
        with torch.no_grad():
            _, out = model(xs, ms, sid)
            p = out["targets"].numpy() * ysd + ymu
        return p[:, 0], Y[idx, 0], pos[idx]

    def metrics(y, p, pos_arr, baselines):
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
            **baselines,
        }
        r["composite"] = cqs.composite_quality(r)
        return r

    pv, yv, posv = predict(seasons == 2024)
    pt, yt, post = predict(seasons == 2025)
    b_pos = {
        q: float((yv - pv)[posv == q].mean())
        for q in SKILL
        if (posv == q).sum() >= 30
    }
    b_g = float((yv - pv).mean())

    old = json.loads((DATA / "mtnn_report.json").read_text(encoding="utf-8"))
    baselines = {
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

    base_va = metrics(yv, pv, posv, baselines)
    best_a, best_cqs = 0.0, base_va["composite"]["cqs"]
    for i in range(0, 21):
        a = i / 20
        pp = np.array([pv[j] + a * b_pos.get(posv[j], b_g) for j in range(len(pv))])
        m = metrics(yv, pp, posv, baselines)
        if m["model_fpts_mae"] <= base_va["model_fpts_mae"] + 0.05 and m["composite"]["cqs"] >= best_cqs:
            best_a, best_cqs = a, m["composite"]["cqs"]

    pp = np.array([pt[j] + best_a * b_pos.get(post[j], b_g) for j in range(len(pt))])
    te = metrics(yt, pp, post, baselines)
    ok, why = cqs.should_promote({**te, "composite": te["composite"]})
    print(f"alpha={best_a} test MAE={te['model_fpts_mae']} CQS={te['composite']['cqs']} {why}")
    if not ok:
        print("NOT PROMOTED")
        return 1

    te["bias_calib"] = {
        "alpha": best_a,
        "per_pos": {k: round(v, 3) for k, v in b_pos.items()},
        "global": round(b_g, 3),
        "fit": "val_season",
        "select": "max_val_cqs_mae_slack_0.05",
    }
    te["promote_metric"] = "cqs"
    te["hill_climb"] = "rz+conformal80+tower_contrib+cqs+bias_shrink"
    te["metrics_note"] = (
        "Promote gate = CQS; MAE soft floor. Val-fit per-pos bias shrink applied."
    )
    te["composite"]["baseline_cqs"] = cqs.BASELINE["cqs"]
    te["composite"]["would_promote_vs_baseline"] = True
    te["composite"]["promote_check"] = why
    # keep family ablation if present
    if "family_ablation" in old:
        te["family_ablation"] = old["family_ablation"]

    (DATA / "mtnn_report.json").write_text(json.dumps(te, indent=2), encoding="utf-8")

    # Update composite baseline constants for future gates
    cqs_path = Path(__file__).resolve().parent / "composite_score.py"
    text = cqs_path.read_text(encoding="utf-8")
    text = text.replace('"mae": 4.258,', f'"mae": {te["model_fpts_mae"]},')
    text = text.replace('"cqs": 61.25,', f'"cqs": {te["composite"]["cqs"]},')
    # only if still old baseline
    if "4.258" in text or "61.25" in text:
        text = text.replace(
            'BASELINE = {\n    "mae": 4.258,\n    "cqs": 61.25,  # CQS of promoted MAE-4.258 checkpoint\n}',
            f'BASELINE = {{\n    "mae": {te["model_fpts_mae"]},\n    "cqs": {te["composite"]["cqs"]},  # bias_shrink α={best_a}\n}}',
        )
    cqs_path.write_text(text, encoding="utf-8")

    # Shift skill projs on boards (idempotent via bias_calib_applied flag)
    for name in ("nextgame.json", "projections.json"):
        p = ASSETS / name
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("bias_calib_applied"):
            print(f"{name} already bias-calibrated — skip shift")
            continue
        for pl in d["players"]:
            if pl.get("source") == "kdst" or pl.get("pos") in ("K", "DST"):
                continue
            off = best_a * b_pos.get(pl["pos"], b_g)
            pl["proj"] = round(float(pl["proj"]) + off, 2)
            band = float(pl.get("uncertainty") or te["conformal_q_by_pos"].get(pl["pos"], te["conformal_q"]))
            pl["floor"] = round(max(0.0, pl["proj"] - band), 2)
            pl["ceil"] = round(pl["proj"] + band, 2)
        if "model" in d and isinstance(d["model"], dict):
            d["model"]["report"] = te
        d["bias_calib_applied"] = {
            "alpha": best_a,
            "per_pos": {k: round(v, 3) for k, v in b_pos.items()},
            "built": time.strftime("%Y-%m-%d"),
        }
        d["count"] = len(d["players"])
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d, separators=(",", ":")), encoding="utf-8")
        tmp.replace(p)
        print(f"updated {name}")

    print("PROMOTED", te["model_fpts_mae"], te["composite"]["cqs"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
