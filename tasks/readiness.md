# Readiness report — Gridiron MTNN v2

**Date:** 2026-07-10  
**Status:** Ready to use locally; assets regenerated. Deploy optional.

## Decisions made (auto-mode)

1. Multi-tower + gated fusion (hoops v4 pattern), not Transformer day-one.
2. Free sources only: nflverse + ffopportunity EP (CC-BY-SA).
3. Mask-honest for NGS gaps (2024 empty on nflverse; 2025 unpublished) and injuries post-2024.
4. Kept UI JSON contracts; `train_models.py` → thin wrapper to `train_mtnn.py`.

## Verify commands

```powershell
cd c:\Users\jcdav\vector-gridiron
python pipeline/feature_inspect.py
python pipeline/verify_accuracy.py
node pipeline/verify_logic.mjs
```

All three passed (2026-07-10).

## Key numbers (held-out 2025)

- **PPR MAE 4.268** (v1: 4.313) — beats last-4 4.616 and STD 4.523
- 49,881 player-weeks × 82 features × 13 masked families
- 47,344 params; CUDA train ~2 min

## Artifacts

- `pipeline/data/train_matrix.npz`, `feature_manifest.json`, `mtnn_report.json`, `mtnn_best.pt`
- `assets/nextgame.json`, `projections.json`, `embedding.json`
- Docs: `docs/SPEC.md`, `DATA_SOURCES_DEEP.md`, `MTNN_ARCHITECTURE.md`

## Next action

Deploy when you want live site updated: `vercel deploy --prod --yes` from `vector-gridiron`  
(or wait for Tuesday `VectorGridironWeeklyRefresh`). Optional: init git if you want commits.
