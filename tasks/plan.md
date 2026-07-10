# Plan — Vector Gridiron (user-centric + metrics) · 2026-07-10

> `/auto-mode` continuation · parent: hill-climb board · live MAE **4.258** · conformal q80

## Answers (locked)

| Question | Decision |
|----------|----------|
| **K / DST in MTNN?** | **No.** Skill MTNN = QB/RB/WR/TE only. K/DST ship via `pipeline/build_kdst.py` → `assets/kdst.json` (season-rate) and merge into roster / start-sit / draft / lookback. |
| **Defense family in MTNN?** | Yes as **opponent DvP** features (`defense` family) — not fantasy DST units. Ablation: near-flat on 2025. |
| **Metrics battery?** | **Yes, report-only.** Promote gate stays **MAE**. Add MAPE / RMSE / MedAE / bias / per-pos MAE to `mtnn_report.json` + README Methods. |

## Vertical slices

| # | Deliverable | Done when |
|---|-------------|-----------|
| 1 | Metrics battery in report + Methods | `mtnn_report` has mape/rmse/medae/bias/per_pos; README cites; MAE gate unchanged |
| 2 | Owner-first Lookback | Focused owner (default: you) hero card; owner toggle; draft cards pin focus |
| 3 | Standing-style rankings | Modes: Draft · Titles · PF · Playoffs; career table respects mode |
| 4 | Start/sit ranking honesty | Document: needs weekly lineup history — no fake metric |
| 5 | Verify + ship | verify_accuracy + verify_logic green; deploy |

## Out of scope (this board)

- Training K/DST inside MTNN
- Soft-weight flat families / EWMA span-5 (next hill-climb tick)
- Weekly start/sit efficiency from matchup history (follow-on)
