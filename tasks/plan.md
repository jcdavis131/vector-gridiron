# Plan — Gridiron MTNN v2 (hoops rigor)

> Generated 2026-07-09 under `/auto-mode` after session-orient + deep research.  
> Spec: `docs/SPEC.md` · Sources: `docs/DATA_SOURCES_DEEP.md` · Arch: `docs/MTNN_ARCHITECTURE.md`

## Goal

Raise Vector Gridiron’s weekly fantasy MTNN to Vector Hoops rigor: deep cited data sources, masked feature families, multi-tower multi-task net, promotion gates — without breaking the existing UI artifact contract.

## Dependency order

```
docs (SPEC/sources/arch)
    → nfl_data ingest expand
        → build_opportunity (EP/RZ)
            → build_features (families + masks)
                → feature_inspect
                    → train_mtnn + export assets
                        → verify_accuracy + verify_logic
                            → refresh.py + README Methods
```

## Vertical slices

1. **Docs locked** — catalog + SPEC + arch + this plan (approval gate).
2. **Ingest P0/P1 feeds** — depth_charts, NGS, PFR adv, ffopportunity EP (download), wire into `nfl_data.py`.
3. **Family matrix** — rewrite `build_features.py` to emit `Z, mask, manifest` + upcoming rows.
4. **MTNN train/export** — `train_mtnn.py` towers/heads; write nextgame/projections/embedding.
5. **Gates + refresh** — verify scripts; point `refresh.py` at v2; Methods honesty.

## Risks

| Risk | Mitigation |
|------|------------|
| Injury feed dead 2025+ | Mask family; document; depth chart as role proxy |
| EP download missing | Derive RZ shares from pbp subset or defer EP with mask |
| Bigger net overfits | Cap params; early stop; G3 vs v1 MAE |
| Train time long | CPU batch 512; optional CUDA; cost-transparency if >20 min |
| No git repo | Docs only until user asks to init/commit |

## Out of scope

Paid PFF; in-season participation; Transformer fusion until gated towers beat v1.
