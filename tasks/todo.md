# Todo — Gridiron MTNN v2

- [x] **0. Approval** — user confirmed SPEC (`/auto-mode yes`)
- [x] **1. Ingest** — `nfl_data.py`: depth, ngs, pfr_adv, ep_weekly, draft, combine
- [x] **2. Opportunity** — `build_opportunity.py` + EP index in features
- [x] **3. Features** — multi-family `train_matrix.npz` (49881×82, 13 families)
- [x] **4. Inspect** — `feature_inspect.py` (no NaNs; NGS cov 0.51 expected)
- [x] **5. Train** — `train_mtnn.py` MAE **4.268** (v1 was 4.313); beats baselines
- [x] **6. Rookie path** — draft-capital model retained (val MAE 2.916)
- [x] **7. Verify** — `verify_accuracy.py` 0 fails; `verify_logic.mjs` ALL PASSED
- [x] **8. Wire** — `refresh.py` → train_mtnn; README Methods + injury caveat
- [x] **9. Close** — readiness report below

## Held-out 2025 results
| Metric | v1 | v2 |
|--------|----|----|
| PPR MAE | 4.313 | **4.268** |
| R² | 0.42 | 0.39 |
| last-4 MAE | 4.616 | 4.616 |
| STD MAE | 4.523 | 4.523 |
| features / families | 41 / 1 | **82 / 13** |
