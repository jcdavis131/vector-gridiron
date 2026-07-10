# Todo — Gridiron hill-climb (E2E post-v2)

- [x] **0. Orient** — plan locked; baseline MAE 4.268 documented
- [x] **1. Methods honesty** — README Methods: MAE/R², NGS+injury, floor/ceil
- [x] **2. Family ablation** — drop-one ΔMAE → `mtnn_report.json` + `family_ablation.py`
- [x] **3. P0 features** — EWMA/DvP/RZ-proxy implemented behind `HILL_CLIMB_FEATURES`
- [x] **4. Reg/HP smoke** — family_drop + expand **did not beat 4.268** → not promoted (`tasks/hillclimb-log.md`)
- [x] **5. Product** — per-pos residual/`uncertainty` in train export; Next Game copy honesty
- [ ] **6. Verify + ship** — gates + commit + prod deploy

## Promoted baseline (do not regress)

| Metric | v2 |
|--------|-----|
| PPR MAE (2025) | **4.268** |
| last-4 MAE | 4.616 |
| STD MAE | 4.523 |
| features | 82 / 13 families |

## Next hill-climb bets (not this board)
- True pbp red-zone shares (not TD-rate proxy)
- Soft-weight / prune flat families (NGS/defense on 2025)
- Quantile / pinball uncertainty heads
