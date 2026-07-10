# Todo — Gridiron (user-centric + metrics)

- [x] **K/DST policy** — skill MTNN = QB/RB/WR/TE; K/DST via `kdst.json`
- [x] **Metrics battery** — MAE gate + MAPE/RMSE/MedAE/bias/per-pos in report (MAE 4.258)
- [x] **Owner-first Lookback** — focus picker, you card, pin ★, standings modes
- [x] **Start/sit ranking** — documented as needing weekly history (no fake metric)
- [x] **Soft-weight flat families ×0.25** — MAE 4.322, **not** promoted
- [x] **EWMA span-5 only** — MAE 4.287, **not** promoted (`EWMA_SPAN5=False`)
- [x] **Soft ngs+defense ×0.25** — MAE 4.310, **not** promoted
- [x] **tower_contrib UI** — MAE **4.258** tied — **promoted**
- [x] **Weekly start/sit ingest** — matchup efficiency + Lookback ranking mode
- [x] **Playoff clutch (wk15+)** — clutch_eff + Clutch standings modes
- [x] **Slow loop to 10m** — 1m outran useful bets; expand paused pending new season
- [x] **10m tick polish** — draft cards + README; expand skipped (documented)
- [x] **Live weekly matchup narratives** — Lookback week-by-week from boxscores
- [ ] Next: expand when new season lands · or user-directed bet
- Loop armed every **10m** until stopped (PID 42588)

## Promoted

| Metric | Value |
|--------|-------|
| PPR MAE (2025) | **4.258** |
| R² | 0.402 |
| RMSE / MedAE | 6.171 / 2.769 |
| Floor/ceil | conformal q80 |
| features | 85 (RZ on) |
