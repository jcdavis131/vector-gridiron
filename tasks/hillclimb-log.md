# Hill-climb experiment log — 2026-07-10

## Promoted baseline (current)
- PPR MAE **4.258** · R² **0.402** · **85** feats / 13 families
- `RZ_FEATURES = True` (pbp RZ tgt/carry/inside-5 shares)
- Floor/ceil: **conformal abs-residual q80** (coverage 0.80 on 2025)
- `HILL_CLIMB_FEATURES = False` · `EWMA_FEATURES = False`

## Family ablation (on pre-RZ 4.268 checkpoint)
| Family | ΔMAE when dropped |
|--------|-------------------|
| usage | +0.493 |
| form | +0.399 |
| opportunity | +0.134 |
| meta | +0.127 |
| pedigree | +0.026 |
| market | +0.021 |
| defense / ngs / role / availability / context / conditions / pfr | ~0 |

## Trials
| Config | Test MAE | Notes |
|--------|----------|-------|
| +EWMA +DvP +snapΔ +rz_proxy +WR1 + family_drop=0.1 | 4.278 | not promoted |
| same, family_drop=0.0 | 4.378 | not promoted |
| **+pbp RZ shares only** | **4.258** | **PROMOTED** (beats 4.268) |
| prune ngs+defense+role+availability | 4.283 | not promoted (worse than RZ) |
| +pinball q10/q90 heads (0.25× loss) | 4.333 | not promoted; coverage 0.62 (under); R² 0.364 |
| zero `conditions` only | 4.320 | not promoted |
| +EWMA spans only on RZ (88 feats) | 4.299 | not promoted |
| **conformal q80 floor/ceil** | **4.258** | **PROMOTED** (product; MAE tied; coverage 0.80) |
| soft-weight flat fams ×0.25 | 4.322 | not promoted |
| +EWMA span-5 only on RZ (86 feats) | 4.287 | not promoted |
| soft-weight ngs+defense ×0.25 | 4.310 | not promoted |
| **tower_contrib export + UI** | **4.258** | **PROMOTED** (product; MAE tied) |
| **weekly start/sit ingest** | n/a | **PROMOTED** (product; Sleeper+ESPN matchups) |

## Next bets
1. Revisit feature expand only with stronger prior (new data season)
2. Soft-weight scale sweep only if new evidence
3. Playoff-week start/sit weighting / clutch metric

## Shipped
- README Methods · `family_ablation.py` · per-pos uncertainty
- `build_rz.py` + nflverse pbp parquet · RZ opportunity cols
- Conformal floor/ceil (abs residual q80 by position)
- Metrics battery (MAPE/RMSE/MedAE/bias/per-pos) — MAE remains promote gate
- Owner-first Lookback (focus picker · you card · standings modes)
- Tower contrib (gated attn×gate top-5) on nextgame/proj + profile UI
- Weekly start/sit efficiency from Sleeper/ESPN matchups (latest 1–2 seasons)
