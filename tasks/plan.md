# Plan — Vector Gridiron hill-climb (post-MTNN v2)

> Generated 2026-07-10 under `/auto-mode` · baseline `5fe5c03` clean  
> Spec parents: `docs/SPEC.md` · `docs/MTNN_ARCHITECTURE.md` · `docs/DATA_SOURCES_DEEP.md`  
> Live evidence: held-out 2025 PPR MAE **4.268** (v1 4.313); matrix 49881×82 / 13 families

## Goal

Hill-climb the **end-to-end game**: better features → better MTNN → honest uncertainty → UI that explains predictions — without breaking artifact contracts or promoting a worse model.

## Dependency order

```
Methods honesty (README)
    → drop-one family ablation → mtnn_report
        → P0 features (RZ + EWMA + DvP) → rebuild matrix
            → family dropout + small HP smoke → promote if MAE wins
                → uncertainty (floor/ceil) + tower_contrib UI
                    → context/role polish
                        → verify + deploy
```

## Vertical slices (acceptance)

| # | Deliverable | Done when |
|---|-------------|-----------|
| 1 | README Methods honesty | MAE/R², NGS+injury caveats, floor/ceil definition cited |
| 2 | Family ablation report | `mtnn_report.json` has per-family ΔMAE; prune note if any hurt |
| 3 | P0 feature fill | RZ shares + EWMA form + richer DvP in matrix; MAE ≤ 4.268 (prefer ≤4.20) |
| 4 | Reg + HP smoke | family dropout and/or d_emb trial; promote only if G2/G3 pass |
| 5 | Uncertainty + tower_contrib | nextgame players carry calibrated floor/ceil + top family weights; UI shows them |
| 6 | Verify + ship | `verify_accuracy` + `verify_logic` green; prod deploy |

## Out of scope (this board)

- Paid PFF; transformer fusion (only if #3–4 plateau)
- Fixing nflverse injury 2025+ / NGS 2025 unpublished (document only)
- League-connect UX (already shipped)

## Risks

| Risk | Mitigation |
|------|------------|
| Feature fill regresses MAE | Keep prior `train_matrix` / checkpoint; promote only on gate |
| Train >20 min | cost-transparency pause; `--skip-build` for HP loops |
| UI breaks old clients | additive JSON fields only |
