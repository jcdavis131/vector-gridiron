"""Walk-forward weekly rank-quality backtest for the Gridiron MTNN.

The site publishes only the UPCOMING week (nextgame.json) — there is no
timestamped archive of past published projections. So this backtests the
projection METHOD, not a publish history, and says so in the artifact:
the shipped selection-split checkpoint (weights trained on seasons ≤2023,
early-stopped on 2024, bias/affine calibration fit on 2024) scores every
held-out test-season week from features built strictly from PRIOR weeks
(build_features is leakage-safe). Nothing about a scored week — its stats,
its season, or any later data — was available to the weights or the
calibration. Each player's first game of a season has no prior-week form
and is excluded by the feature builder, so coverage is weeks 2+.

Metric: per-week Spearman rank correlation (average-rank ties) between the
projected PPR ordering and actual PPR outcomes, per position (QB/RB/WR/TE)
and overall, averaged across weeks. Baselines ranked the same way:
last-4-average PPR and season-to-date PPR.

Honesty guard: refuses to run if the checkpoint carries final-refit weights
(trained on all labeled seasons, including the test season) — a backtest
with those weights would be retro-fit, not walk-forward.

Run:  python pipeline/build_backtest.py     ->  assets/eval_backtest.json
Requires: pipeline/data/train_matrix.npz + mtnn_best.pt from train_mtnn.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from train_mtnn import MTNN, SKILL, family_slices, split_by_family

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "pipeline" / "data"
ASSETS = ROOT / "assets"
MIN_GROUP = 8  # smallest week×position group we score


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank (ties share the mean rank) — matters because many
    actual weekly lines are identical (0.0, 1.x)."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3:
        return None
    rx, ry = _rankdata(np.asarray(x, np.float64)), _rankdata(np.asarray(y, np.float64))
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def load_checkpoint():
    ck = torch.load(DATA / "mtnn_best.pt", map_location="cpu", weights_only=False)
    deploy = (ck.get("report") or {}).get("deploy") or {}
    if deploy.get("mode") != "selection_only":
        raise SystemExit(
            f"REFUSING backtest: checkpoint deploy mode is {deploy.get('mode')!r} "
            "(weights saw the test season — a walk-forward backtest needs the "
            "selection-split checkpoint; rerun train_mtnn.py --phase select)")
    bias_fit = ck["report"]["bias_calib"].get("fit")
    if bias_fit != "val_season":
        raise SystemExit(
            f"REFUSING backtest: bias calibration fit={bias_fit!r} "
            "(must be val_season, i.e. fit on data before the test season)")
    return ck


def predict_test(ck, npz):
    """Score every test-season row with the frozen checkpoint + its
    pre-test calibration chain (mirrors train_mtnn's export path)."""
    rep = ck["report"]
    test_season = int(rep["test_season"])
    season = npz["season"]
    pos_all = npz["pos"]
    te = (season == test_season) & np.isin(pos_all, list(SKILL))
    X = npz["Z"][te].astype(np.float64)
    M = npz["mask"][te].astype(np.float64)
    y = npz["Y"][te][:, 0].astype(np.float64)  # fpts_ppr
    weeks = npz["week"][te].astype(int)
    pos = pos_all[te]

    feats = ck["feats"]
    families = ck["families"]
    slices = family_slices(feats, families)
    fam_dims = {fam: len(cols) for fam, cols in slices.items()}
    n_targets = len(ck["ymu"])
    model = MTNN(fam_dims, n_seasons=ck["n_seasons"], d_emb=ck["d_emb"],
                 n_targets=n_targets,
                 n_usage=ck["state"]["usage_head.weight"].shape[0])
    model.load_state_dict(ck["state"])
    model.eval()

    mu, sd, ymu, ysd = ck["mu"], ck["sd"], ck["ymu"], ck["ysd"]
    Xz = ((X - mu) / sd) * M
    xs, ms = split_by_family(Xz, M, slices, torch.device("cpu"))
    sid = torch.tensor(season[te] - ck["season_min"], dtype=torch.long)
    sid = sid.clamp(0, ck["n_seasons"] - 1)
    with torch.no_grad():
        emb = model.encode(xs, ms, sid)
        targets = torch.cat([h(emb) for h in model.target_heads], dim=1)
    p = targets.numpy()[:, 0] * ysd[0] + ymu[0]

    # calibration chain fit on the 2024 val season (predates every scored week)
    bc = rep["bias_calib"]
    p = p + bc["alpha"] * np.array([bc["per_pos"].get(pp, bc["global"]) for pp in pos])
    ac = rep["affine_calib"]
    mix = ac["mix"]
    for pp, coef in ac["per_pos"].items():
        m = pos == pp
        p[m] = (1 - mix) * p[m] + mix * (coef["a"] * p[m] + coef["b"])

    base_last4 = X[:, feats.index("f_fpts_ppr")]
    base_std = X[:, feats.index("std_ppr")]
    return {"pred": p, "y": y, "weeks": weeks, "pos": pos,
            "base_last4": base_last4, "base_std": base_std,
            "test_season": test_season}


def rank_table(d):
    """Per-week Spearman per position + overall, averaged across weeks."""
    weeks = sorted(set(d["weeks"].tolist()))
    groups = list(SKILL) + ["ALL"]
    out = {}
    for g in groups:
        rhos, rhos_l4, rhos_std, ns = [], [], [], []
        for w in weeks:
            m = d["weeks"] == w
            if g != "ALL":
                m = m & (d["pos"] == g)
            if m.sum() < MIN_GROUP:
                continue
            rho = spearman(d["pred"][m], d["y"][m])
            if rho is None:
                continue
            rhos.append(rho)
            rhos_l4.append(spearman(d["base_last4"][m], d["y"][m]) or 0.0)
            rhos_std.append(spearman(d["base_std"][m], d["y"][m]) or 0.0)
            ns.append(int(m.sum()))
        out[g] = {
            "spearman": round(float(np.mean(rhos)), 4),
            "spearman_sd": round(float(np.std(rhos)), 4),
            "baseline_last4": round(float(np.mean(rhos_l4)), 4),
            "baseline_std": round(float(np.mean(rhos_std)), 4),
            "weeks": len(rhos),
            "n_rows": int(np.sum(ns)),
            "avg_week_n": round(float(np.mean(ns)), 1),
            "per_week": [round(r, 3) for r in rhos],
        }
    return weeks, out


def main() -> None:
    ck = load_checkpoint()
    npz = np.load(DATA / "train_matrix.npz", allow_pickle=True)
    d = predict_test(ck, npz)
    weeks, table = rank_table(d)
    rep = ck["report"]
    artifact = {
        "computed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metric": {
            "name": "walk-forward weekly Spearman rank correlation",
            "definition": (
                "For each scored week, Spearman rank correlation "
                "(average-rank ties) between the model's projected PPR "
                "ordering and actual PPR fantasy points, computed per "
                "position and overall, then averaged across weeks. "
                "Baselines (last-4 PPR average, season-to-date PPR) are "
                "ranked identically."),
        },
        "season": d["test_season"],
        "weeks": weeks,
        "min_group": MIN_GROUP,
        "positions": {g: {k: v for k, v in row.items() if k != "per_week"}
                      for g, row in table.items() if g != "ALL"},
        "overall": table["ALL"],
        "model": {
            "weights": "selection split (train ≤2023, early stop on 2024)",
            "calibration": "bias shrink + per-pos affine, fit on 2024 val",
            "architecture": rep.get("architecture"),
            "checkpoint_mode": (rep.get("deploy") or {}).get("mode"),
        },
        "caveats": [
            "No timestamped archive of published projections exists — the "
            "site only ships the upcoming week — so this is a backtest of "
            "the projection METHOD: the frozen pre-2025 checkpoint scored "
            "on as-of features built strictly from prior weeks.",
            "Week 1 of the season is not scored: the feature builder "
            "requires at least one prior game in-season.",
            "K/DST are excluded — they use separate season-rate models "
            "(see kdst.json holdout MAE), not the MTNN.",
            "Spearman measures ordering quality only, not point accuracy "
            "(see mtnn_report.json MAE/RMSE for that).",
        ],
    }
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "eval_backtest.json").write_text(
        json.dumps(artifact, separators=(",", ":")), encoding="utf-8")
    print(f"wrote assets/eval_backtest.json - season {d['test_season']} - "
          f"weeks {weeks[0]}-{weeks[-1]} - {table['ALL']['n_rows']} rows")
    for g in list(SKILL) + ["ALL"]:
        r = table[g]
        print(f"  {g:>3}  rho={r['spearman']:+.3f} (sd {r['spearman_sd']:.3f})  "
              f"last4={r['baseline_last4']:+.3f}  std={r['baseline_std']:+.3f}  "
              f"weeks={r['weeks']}  avg n={r['avg_week_n']}")


if __name__ == "__main__":
    main()
