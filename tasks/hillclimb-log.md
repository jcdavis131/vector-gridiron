# Hill-climb experiment log — 2026-07-10

## Promoted baseline (current)
- **CQS 63.16** · MAE **4.296** · R² **0.422** · RMSE **6.068** · bias **−0.311**
- Calib chain: bias shrink α=0.5 → per-pos affine mix=1.0 (both val-selected)
- Promote: CQS ≥ baseline+0.5 **and** MAE ≤ baseline+0.05
- Positions: QB/RB/WR/TE MTNN + K/DST season-rate on boards

## Trials
| Config | MAE | CQS | Notes |
|--------|-----|-----|-------|
| RZ raw | 4.258 | 61.25 | prior |
| bias α=0.5 | 4.294 | 62.31 | promoted |
| **bias + affine mix=1** | **4.296** | **63.16** | **PROMOTED** |
| blend last4 w=0.15 | 4.283 | 63.27 | not promoted (< +0.5 CQS) |
| blend STD w=0.15 | 4.285 | 63.22 | not promoted |
| isotonic mix=1 | 4.326 | 62.93 | not promoted |
| soft ngs+defense+role+avail ×0.25 | 4.319 | 62.65 | not promoted (restored) |
| soft conditions ×0.5 | 4.292 | 63.17 | not promoted (ΔCQS +0.01; restored) |

## Next bets
1. ~~Post-hoc calib / soft-weight~~ — **plateau** on 2025 holdout
2. Feature expand parked for new season data
3. Optional weekly K/DST or user-directed bet

## Skipped this tick
- Soft conditions ×0.5 — CQS 63.17 < bar 63.66; assets restored
- **Loop stopped** — no actionable CQS bets left without new data / directed work

## Loop
- **stopped** (was 5m / PID 43172)
