"""Composite Quality Score (CQS) for Gridiron skill MTNN + promote helpers.

CQS is higher-is-better on [0, 100]. It blends MAE with RMSE, R², |bias|,
per-position balance, and conformal coverage so promote decisions aren't
MAE-only (which can hide boom/bust misses or position skew).

Promote rule (see tasks/plan.md):
  CQS_new >= CQS_base + 0.5  AND  MAE_new <= MAE_base + 0.05
"""

from __future__ import annotations

from typing import Any

# Anchors for 0–1 transforms (chosen so current ~4.26 MAE sits mid-high).
MAE_SCALE = 10.0
RMSE_SCALE = 15.0
BIAS_SCALE = 5.0
POS_SCALE = 12.0
COVER_TARGET = 0.80
COVER_TOL = 0.20

WEIGHTS = {
    "mae": 0.35,
    "rmse": 0.20,
    "r2": 0.15,
    "bias": 0.10,
    "pos": 0.10,
    "cover": 0.10,
}

# Promoted baseline (2025 holdout) — update when a trial promotes.
BASELINE = {
    "mae": 4.296,
    "cqs": 63.16,  # bias+affine
}


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def component_scores(report: dict[str, Any]) -> dict[str, float]:
    mae = float(report.get("model_fpts_mae") or 0.0)
    rmse = float(report.get("model_fpts_rmse") or 0.0)
    r2 = float(report.get("model_fpts_r2") or 0.0)
    bias = abs(float(report.get("model_fpts_bias") or 0.0))
    cover = float(report.get("conformal_coverage") or 0.0)
    per = report.get("per_pos_mae") or {}
    pos_vals = [float(v) for v in per.values()] if per else [mae]
    pos_blend = 0.6 * (sum(pos_vals) / len(pos_vals)) + 0.4 * max(pos_vals)

    return {
        "mae": _clip01(1.0 - mae / MAE_SCALE),
        "rmse": _clip01(1.0 - rmse / RMSE_SCALE),
        "r2": _clip01(r2),
        "bias": _clip01(1.0 - bias / BIAS_SCALE),
        "pos": _clip01(1.0 - pos_blend / POS_SCALE),
        "cover": _clip01(1.0 - abs(cover - COVER_TARGET) / COVER_TOL),
    }


def composite_quality(report: dict[str, Any]) -> dict[str, Any]:
    comps = component_scores(report)
    cqs = 100.0 * sum(WEIGHTS[k] * comps[k] for k in WEIGHTS)
    return {
        "cqs": round(cqs, 2),
        "components": {k: round(v, 4) for k, v in comps.items()},
        "weights": dict(WEIGHTS),
        "promote_metric": "cqs",
        "promote_rule": (
            "promote if CQS >= baseline_cqs + 0.5 and MAE <= baseline_mae + 0.05"
        ),
        "baseline_mae": BASELINE["mae"],
    }


def should_promote(
    new_report: dict[str, Any],
    *,
    baseline_mae: float | None = None,
    baseline_cqs: float | None = None,
    cqs_delta: float = 0.5,
    mae_slack: float = 0.05,
) -> tuple[bool, str]:
    """Return (ok, reason). Uses composite block on new_report if present."""
    block = new_report.get("composite") or composite_quality(new_report)
    new_cqs = float(block["cqs"])
    new_mae = float(new_report.get("model_fpts_mae") or 99.0)
    base_mae = float(baseline_mae if baseline_mae is not None else BASELINE["mae"])
    if baseline_cqs is None:
        baseline_cqs = BASELINE.get("cqs")
    if baseline_cqs is None:
        return False, "no baseline_cqs yet — record current CQS as baseline first"
    if new_mae > base_mae + mae_slack:
        return False, f"MAE {new_mae:.3f} > floor {base_mae + mae_slack:.3f}"
    if new_cqs < baseline_cqs + cqs_delta:
        return False, f"CQS {new_cqs:.2f} < promote bar {baseline_cqs + cqs_delta:.2f}"
    return True, f"CQS {new_cqs:.2f} >= {baseline_cqs + cqs_delta:.2f} and MAE ok"


def kdst_rows_for_board(kdst: dict[str, Any], *, band: float = 4.0) -> list[dict[str, Any]]:
    """Normalize K/DST records for nextgame/projections player lists."""
    out: list[dict[str, Any]] = []
    for arr, pos in ((kdst.get("kickers") or [], "K"), (kdst.get("dst") or [], "DST")):
        for p in arr:
            proj = float(p.get("proj") or 0.0)
            out.append({
                "key": p.get("key") or f"{(p.get('name') or '').lower()}|{pos}",
                "name": p["name"],
                "pos": pos,
                "team": p.get("team") or "",
                "opp": "",
                "headshot": "",
                "moved": False,
                "prev_team": "",
                "bye": p.get("bye"),
                "avail": "",
                "proj": proj,
                "floor": round(max(0.0, proj - band), 2),
                "ceil": round(proj + band, 2),
                "uncertainty": band,
                "line": {
                    "rec_yds": 0.0, "rush_yds": 0.0, "pass_yds": 0.0,
                    "rec": 0.0, "td": 0.0,
                },
                "comps": [],
                "tower_contrib": [{"family": "kdst_season_rate", "w": 1.0}],
                "source": "kdst",
            })
    return out


def merge_kdst_into_players(
    players: list[dict[str, Any]],
    kdst: dict[str, Any] | None,
    *,
    band: float = 4.0,
) -> list[dict[str, Any]]:
    """Append K/DST if missing (by key). Idempotent."""
    if not kdst:
        return players
    have = {p.get("key") for p in players}
    extra = [r for r in kdst_rows_for_board(kdst, band=band) if r["key"] not in have]
    if not extra:
        return players
    merged = list(players) + extra
    merged.sort(key=lambda r: float(r.get("proj") or 0.0), reverse=True)
    return merged


def walk_forward_mae(hist_by_id: dict[Any, dict[int, float]]) -> dict[str, float]:
    """Season-rate walk-forward MAE for K or DST histories.

    For each entity and season Y with at least one prior season, project with
    the same 0.65/0.35 last-two rule as build_kdst, compare to actual Y.
    """
    errs: list[float] = []
    n_pairs = 0
    for hist in hist_by_id.values():
        years = sorted(int(y) for y in hist)
        for i, y in enumerate(years):
            prior = years[:i]
            if not prior:
                continue
            if len(prior) == 1:
                pred = hist[prior[-1]]
            else:
                pred = 0.65 * hist[prior[-1]] + 0.35 * hist[prior[-2]]
            errs.append(abs(pred - hist[y]))
            n_pairs += 1
    if not errs:
        return {"mae": None, "n": 0}
    return {"mae": round(sum(errs) / len(errs), 3), "n": n_pairs}
