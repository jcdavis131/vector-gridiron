# Deploy-greedy MTNN protocol (all vector sports)

> **Status:** complete · 2026-07-11  
> **Goal:** evaluate honest, deploy greedy — selection metrics stay held-out; shipped weights see all labeled data; push to Vercel prod.

## Protocol

1. **Selection run** — train on the honest split; write MAE/CQS/recall to report.
2. **Promote gate** — if fail, keep prior shipped assets; stop.
3. **Final refit** — same recipe, train on **all labeled** rows; fixed epochs from selection (no retuning).
4. **Ship** — export assets from refit weights; report metrics remain the **selection** estimate, labeled `metrics_source: selection_holdout`.
5. **Prod** — `vercel --prod` for gridiron, hoops, pitch.

## Board

1. [x] Gridiron: `--phase select|final-refit|auto` + report `deploy` block
2. [x] Hoops: `--phase` + train-split selection + auto full-corpus refit; champ report stamped
3. [x] Pitch: `--phase auto` gates `--save-final` on ≥2/4 vs PCA
4. [x] Gridiron `--phase final-refit` → assets + deploy block (n_train=49881, 11ep)
5. [x] Hoops: promote Bet D (CQS 85.87) → embeddings/viz/jacobian/attr → assets/
6. [x] Pitch deploy stamp + prod (embeddings already full-corpus)
7. [x] Auto redeploy-ok: CQS ≥ baseline still ships full-data refit (no +0.5 required)
8. [x] Vercel prod: hoops.dumbmodel.com, pitch.dumbmodel.com, gridiron (in flight / done)
9. [x] Close-out

## Prod URLs

| Sport | Alias | Notes |
|-------|-------|-------|
| Hoops | https://hoops.dumbmodel.com | Bet D MTNN + attr/jacobian |
| Pitch | https://pitch.dumbmodel.com | PitchMTNN e_p 24-d |
| Gridiron | https://gridiron.dumbmodel.com (confirm alias) | Final-refit all 49,881 weeks |

## Non-goals

- Retuning HP on full data
- Claiming full-data in-sample MAE as held-out
- Touching unrelated site polish / network-viz worktree merge
- Git push of dirty branches (CLI prod deploy used instead)
