# Todo — Gridiron hill-climb loop

- [x] **RZ pbp shares** — MAE **4.258** promoted (`8cbf1ca`)
- [x] **Prune flat families** — MAE 4.283, **not** promoted
- [x] **Quantile / pinball** — MAE 4.333, **not** promoted (restored 4.258)
- [x] **Zero `conditions`** — MAE 4.320, **not** promoted
- [x] **EWMA-only on RZ** — MAE 4.299, **not** promoted (`EWMA_FEATURES=False`)
- [x] **Conformal q80 floor/ceil** — MAE **4.258** tied, coverage 0.80 — **promoted**
- [ ] **Next: soft-weight flat families** or single EWMA span-5
- Loop armed every **20m** (PID tracked in terminal) until stopped

## Promoted

| Metric | Value |
|--------|-------|
| PPR MAE (2025) | **4.258** |
| R² | 0.402 |
| features | 85 (RZ on) |
