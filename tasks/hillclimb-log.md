# Hill-climb experiment log — 2026-07-10

## Promoted baseline (current)
- PPR MAE **4.258** · R² **0.402** · **85** feats / 13 families
- `RZ_FEATURES = True` (pbp RZ tgt/carry/inside-5 shares)
- `HILL_CLIMB_FEATURES = False` (EWMA bundle still off)

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

## Next bets
1. Soft-prune / zero flat families (NGS, defense, role, availability) at train
2. Quantile / pinball uncertainty heads
3. Revisit EWMA only on top of RZ (not the full failed bundle)

## Shipped
- README Methods · `family_ablation.py` · per-pos uncertainty
- `build_rz.py` + nflverse pbp parquet · RZ opportunity cols
