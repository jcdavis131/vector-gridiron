# Hill-climb experiment log — 2026-07-10

## Promoted baseline (keep)
- PPR MAE **4.268** · R² 0.39 · 82 feats / 13 families
- Assets restored from git; `HILL_CLIMB_FEATURES = False` in `build_features.py`

## Family ablation (on promoted checkpoint)
| Family | ΔMAE when dropped |
|--------|-------------------|
| usage | +0.493 |
| form | +0.399 |
| opportunity | +0.134 |
| meta | +0.127 |
| pedigree | +0.026 |
| market | +0.021 |
| defense / ngs / role / availability / context / conditions / pfr | ~0 |

## Feature expand trials (did **not** promote)
| Config | Test MAE | Notes |
|--------|----------|-------|
| +EWMA +DvP pass/rush +snapΔ +rz_proxy +WR1 + family_drop=0.1 | **4.278** | beats baselines, worse than 4.268 |
| same, family_drop=0.0 | **4.378** | worse than v1 |

Code path retained behind `HILL_CLIMB_FEATURES`. Next bets: true pbp RZ shares; prune flat families; quantile heads — not more redundant form means.

## Shipped this arc
- README Methods honesty
- `pipeline/family_ablation.py` + ablation in `mtnn_report.json`
- Per-pos residual / `uncertainty` export path in train (when retrain promotes)
- Next Game copy notes floor–ceil = residual σ
