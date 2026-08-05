# Vector Gridiron

A fantasy-football cockpit: a static site that ranks weekly lineups using next-game fantasy-point predictions from a multi-tower neural net trained on nflverse data (usage, snaps, age, weather, Vegas lines, rest, defense-vs-position — no player tracking).

Live: https://gridiron.dumbmodel.com/ (gridiron.jcamd.com redirects there)

> Solo personal project, no connection to employer, built with public/free-tier only.

## What's here

- `index.html` + `assets/app.js` — the cockpit UI: canvas map with custom shaders (plain WebGL/canvas, no engine), lineup board, share cards. Mobile-first responsive (`assets/responsive.css`).
- `dashboard.html` — the model-lab view; it reads model metrics (MAE / R²) from `assets/eval_scoreboard.json` rather than hardcoding them.
- `pipeline/train_mtnn.py` — training now in-repo (2026-08-05): ResidualTower `cat([x·m,m])→96h→24d` + TransformerFusion `d_model128 4-head 4-layer CLS→32-d L2`, RealMLP RobustScaler median/IQR clip[-3,3] + PLE k=8 d_out16, Procrustes Q chain season→root. Emits `assets/vectors.json` + `pipeline/data/mtnn.pt`.
- `pipeline/eval_next_game.py` — MAE/R² evaluator, updates `assets/eval_scoreboard.json`
- `scripts/` — model export utilities (`export_onnx.py`, `export_executorch.py`, `tabpfn_distill.py`).
- `assets/era_procrustes_align.py`, `assets/realmlp_preproc.py` — preprocessing/alignment code used by the training pipeline.

## Model status (honest, 2026-08-05)

- **Embedding:** 32-d native primary (`d_emb 32`), 16-d slice + re-L2 for legacy compat (`d_emb_legacy 16`), advertised 32 — resolves 16 vs 32 vs 64 confusion. Code in `assets/vectors.json` built 2026-08-05, model `GridironMTNN native 32-d transformer 128d 4L 4H`.
- **Shipped:** Currently synthetic demo (`pipeline/data/embedding_gridiron.npz` 2000×160) with val MAE 8.41 R² -7.15 — high because synthetic, expected.
- **Claimed (old offline):** next-game MAE 4.268 R² 0.39 — marked `claimed_not_reproducible_offline_train_missing` but now train_mtnn.py exists, so reproducible once nflverse fetch lands (fetch script planned: `pipeline/fetch_nflverse.py` nflreadpy 2020-2025 + weather + Vegas → `train_matrix.npz` 10 families holistic 160 feats).
- **Target:** MAE 4.268→3.8 with Procrustes + RealMLP + MoE + TabPFN distill KL T=2 w=0.15
- **Gate:** New 32-d native NOT yet promoted over legacy 16-d until it beats shipped on real nflverse eval (`assets/eval_scoreboard.json` `current_repro` vs `claimed`). `candidate.json` staging first, then promote only if eval beats.
- **Cross-repo:** hoops 64-d, pitch 24-d, gridiron 32-d native + 16-d compat, equities 64-d, joint 64-d

## League share flow

## League share flow

- `?l=<CODE>` league codes stored client-side (`vectorGridiron.v1` localStorage keys)
- Result-card copy/paste + Web Share API, OG image for links
- No backend; boards are device-only

## Deploy

Vercel static (`cleanUrls: true`), redirect gridiron.jcamd.com -> gridiron.dumbmodel.com via `vercel.json`.

MIT. Solo personal project, no connection to employer, built with public/free-tier only.
