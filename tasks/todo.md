# Todo — Gridiron (user-centric + metrics)

- [x] **K/DST policy** — skill MTNN = QB/RB/WR/TE; K/DST via `kdst.json`
- [x] **Metrics battery** — MAE gate + MAPE/RMSE/MedAE/bias/per-pos in report (MAE 4.258)
- [x] **Owner-first Lookback** — focus picker, you card, pin ★, standings modes
- [x] **Start/sit ranking** — documented as needing weekly history (no fake metric)
- [ ] Hill-climb next: soft-weight flat families / EWMA span-5 (loop)
- Loop still armed every 20m until stopped

## Promoted

| Metric | Value |
|--------|-------|
| PPR MAE (2025) | **4.258** |
| R² | 0.402 |
| RMSE / MedAE | 6.171 / 2.769 |
| Floor/ceil | conformal q80 |
| features | 85 (RZ on) |
