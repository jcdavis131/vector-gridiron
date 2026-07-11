# Plan — Composite score + full-position preds · 2026-07-10

> `/auto-mode` · `/loop 1m` · parent MAE gate **4.258**

## Locked decisions

| Question | Decision |
|----------|----------|
| **Skill MTNN positions** | QB / RB / WR / TE only (unchanged) |
| **K / DST** | Stay in `build_kdst.py` → `kdst.json`; **also bake** into `nextgame.json` + `projections.json` so every lineup slot has a board row |
| **Promote gate** | **Composite Quality Score (CQS)** primary; MAE soft floor (no promote if MAE > baseline + 0.05) |

## Why not MAE alone

MAE ignores boom/bust misses (RMSE), ranking signal (R²), systematic over/under (|bias|), position hiding (per-pos balance), and floor/ceil honesty (conformal coverage). A start/sit product needs all of those.

## Composite Quality Score (CQS) — higher is better · 0–100

| Component | Weight | Transform (higher = better) |
|-----------|--------|------------------------------|
| MAE | 0.35 | `max(0, 1 − mae/10)` |
| RMSE | 0.20 | `max(0, 1 − rmse/15)` |
| R² | 0.15 | `clip(r2, 0, 1)` |
| \|bias\| | 0.10 | `max(0, 1 − \|bias\|/5)` |
| Pos balance | 0.10 | `max(0, 1 − (0.6·mean + 0.4·max)/12)` over per-pos MAE |
| Conformal cal | 0.10 | `max(0, 1 − \|cover − 0.80\|/0.20)` |

**Promote iff** `CQS_new ≥ CQS_base + 0.5` **and** `MAE_new ≤ MAE_base + 0.05`.

## K/DST quality (separate, not in CQS numerator)

Walk-forward season MAE: for each season Y with ≥1 prior year, project with 0.65/0.35 last-two rule; MAE vs actual Y ppg. Report in `kdst.json` + verify coverage (≥30 K, 32 DST).

## Vertical slices

| # | Deliverable | Done when |
|---|-------------|-----------|
| 1 | Spec + board (this file + todo) | Plan locked |
| 2 | `pipeline/composite_score.py` + wire into train report | `mtnn_report` has `composite` + promote rule |
| 3 | K/DST holdout MAE + bake into next/proj assets | All 6 positions in both boards; kdst report MAE |
| 4 | verify_accuracy G7/G8 + README Methods | Gates green |
| 5 | Hillclimb log + 1m loop | Loop armed; baseline CQS recorded |

## Out of scope

- Training K/DST inside MTNN
- Replacing conformal bands
- Feature expand (still parked for new season data)
