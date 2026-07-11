# Todo — CQS hill-climb

- [x] CQS gate + all 6 positions on boards
- [x] Bias shrink α=0.5 → CQS 62.31
- [x] Affine calib mix=1.0 → **CQS 63.16** / MAE **4.296**
- [x] Blend last4/STD + isotonic — **not** promoted (CQS < 63.66)
- [x] Soft ngs+defense+role+avail ×0.25 — CQS 62.65, **restored**
- [x] Soft conditions ×0.5 — CQS 63.17, **restored** (plateau)
- [x] **Stop loop** — no actionable CQS bets without new data
- [ ] Resume when new season data · or user-directed bet
- Loop **stopped** (was 5m / PID 43172)

## Promoted

| Metric | Value |
|--------|-------|
| CQS | **63.16** |
| MAE | **4.296** |
| R² | 0.422 |
| RMSE | 6.068 |
| bias | −0.311 |
