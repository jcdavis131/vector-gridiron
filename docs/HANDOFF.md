# Vector Gridiron — HANDOFF

**Date:** 2026-08-09 00:00 CDT  
**Repo:** `jcdavis131/vector-gridiron`  
**Live:** https://gridiron.dumbmodel.com (redirect `gridiron.jcamd.com` →)  
**Zero-deps:** `true` (stdlib inline CSS/JS base64 no network fetch in polish report)

> Solo personal project — NFL fantasy cockpit over 2,000 player-weeks synthetic demo.

## Current State (2026-08-09)

- **Embedding:** 32-d native L2 primary, 16-d compat slice re-L2 for legacy bundle.
  - Transformer fusion `d_model128 n_layers4 n_heads4 CLS→32-d L2` (hoops 64-d, pitch 24-d, gridiron 32-d, equities 64-d, joint 64-d).
  - ResidualTower `cat([x·m,m]) d_cat*2→96h GELU LN→24d` per tower, 10 towers holistic `160 feats` total.
  - RealMLP preprocessing: per-season RobustScaler median/IQR clip[-3,3], PLE `k=8 d_out16` periodic sin/cos proj.
  - Procrustes Q chain season→root rotation-only `QᵀQ=I`, drift via shared players ≥30 residual Frobenius.
- **Shipped synthetic demo:** `pipeline/data/embedding_gridiron.npz` 2000×160, `assets/vectors.json` 2000 rows, MAE 8.47 (R² -7.35) — high because synthetic, expected. See `assets/eval_scoreboard.json`.
- **Claimed:** MAE 4.268 R² 0.39 from old offline run — marked `claimed_not_reproducible_offline_train_missing` in eval JSON. Now `pipeline/train_mtnn.py` exists, but `pipeline/fetch_nflverse.py` still planned, so claimed not yet reproducible until fetch lands.
- **Target:** MAE 4.268→3.8 with MoE sparsity-gated towers + TabPFN distill `KL T=2 w=0.15` + Procrustes + RealMLP gating. Gate: new 32-d NOT promoted over legacy 16-d until it beats shipped on real nflverse eval (`candidate.json` staging).
- **Site:** cockpit UI canvas WebGL no framework, lineup board, dashboard model-lab reads `eval_scoreboard.json`, share cards OG 1200x630 + 1080x1920, PWA v66 CORE24 DENY8 FULL_MTNN15 network-first 1MB cap JSON never cached, `manifest.json` standalone `#1A150F` / `#080A0F`.
- **Pipeline:** `fetch_nflverse.py` (planned) → `train_mtnn.py` (now exists) → `eval_next_game.py` → `export_onnx.py`. Train cmd `python pipeline/train_mtnn.py --epochs 150 --dim 32 --d-model 128 --n-layers 4 --n-heads 4`.

## From ARCHITECTURE_V2 + GRIDIRON_POLISH

- ARCHITECTURE_V2 (2026-08-04) decision: upgrade 16-d legacy → 32-d native, keep 16-d as slice+re-L2 compat for `app.js 116KB` cockpit map drag/pinch 520×520 canvas.
- GRIDIRON_POLISH 2026-08-07 hoops-level parity runId `gridiron-20260807T1020Z`: hero-band eyebrow `2000 player-weeks 32-d native L2 16-d compat`, sky-canvas 800 pts rotating drag pause/reset, map-overlay SHAPE=POS COLOR=ARCH, CTA Play Today/Random/Pack, tri-cards Lab/Players/Trends, viral row Pack Battle 1·3·5 ?pack= toast streak 🔥 countdown midnight UTC, manifest theme `#1A150F` bg `#080A0F`, 40 JS delight suite (41 files inc `app.js`), model.html title How vector knows MTNN weekly network-hero 2000×160→10 towers, methods doctrine every number recomputable, players radar 360×280 Top Similar cosine native/compat, play.html Daily Guess Wordle 6 tries cosine 32-d native, sw.js v66 CORE24 etc.

## How to activate real data

1. Implement `pipeline/fetch_nflverse.py`: `nflreadpy load_pbp seasons=[2020..2025]` + roster snap counts participation + Open-Meteo weather wind/temp/dome + Vegas lines spread/total/implied + injury/rest def-vs-pos form lag 1-3 FPTS PPR rolling avg redzone SOS_NET analog → `pipeline/data/train_matrix.npz` keys `X [N,160] M [N,160] season_ids [N] fpts_next [N]`.
2. `python pipeline/train_mtnn.py --epochs 50 --d-emb 32 --scaling robust --era-align procrustes` → emits `assets/vectors.json` + `pipeline/data/mtnn.pt` + updates `eval_scoreboard.json` current_repro MAE.
3. `python pipeline/eval_next_game.py` → compare `current_repro` vs `claimed` MAE 4.268 gate.
4. If passes, stage `candidate.json`, then promote to `assets/vectors.json` + `assets/eval_scoreboard.json`.

## Verification

```bash
python -m http.server 8000
python -m pytest -q
cat assets/eval_scoreboard.json | python -c "import json,sys; d=json.load(sys.stdin); print(d['embedding_dim_native'], d['current_repro']['mae'], d['claimed_MAE_next_game'])"
```

## Open follow-ups

- `fetch_nflverse.py` missing (planned) → blocks claimed 4.268 repro.
- 5,323 claimed gridiron rows eventual target (nflverse weekly matrix real).
- ONNX/WASM/ExecuTorch export validation INT8 <300KB gz.
- MoE + TabPFN distill to hit MAE 3.8 target.
- PWA v66 verification (standalone display_override, icons 192/512 any+maskable, shortcuts Daily/Lab).
