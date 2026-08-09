# Vector Gridiron

![CI](https://github.com/jcdavis131/vector-gridiron/actions/workflows/ci.yml/badge.svg)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)

A weekly NFL fantasy cockpit over 2,000 player-weeks embedding space.

Live: https://gridiron.dumbmodel.com

> Solo personal project, no connection to employer, built with public/free-tier only.

> **Picking up in-progress work?** Start at [`docs/HANDOFF.md`](docs/HANDOFF.md) — current state, synthetic demo MAE, claimed 4.268 gate, and how to activate the nflverse fetch.

## The embedding

2,000 player-weeks synthetic demo currently (5,323 claimed gridiron rows eventually once `pipeline/fetch_nflverse.py` lands), 32-d native L2 + 16-d compat slice re-L2, transformer fusion `d_model128 4L 4H CLS→32-d`, ResidualTower `cat([x·m,m])→96h→24d` with LayerNorm GELU×2 skip, RealMLP preprocessing RobustScaler median/IQR clip[-3,3], PLE `k=8 d_out16` periodic sin/cos projection, Procrustes Q chain season→root rotation-only `QᵀQ=I`.

Current synthetic val MAE 8.47 (R² -7.3, high because synthetic), claimed MAE 4.268 R² 0.39 from old offline run marked `claimed_not_reproducible`, target MAE 3.8 with MoE sparsity-gated towers + TabPFN distill `KL T=2 w=0.15` + Procrustes era-alignment + 10-family holistic stacking. Source of truth `assets/eval_scoreboard.json` (built 2026-08-05, `embedding_dim_native 32`, `embedding_dim_legacy 16`).

## The site

Cockpit UI: canvas WebGL no framework (plain Canvas/WebGL 800-pt rotating sky, SHAPE=POS COLOR=ARCH overlay), lineup board (Start/Sit weekly FPTS-driven), dashboard model-lab (`dashboard.html` reads `assets/eval_scoreboard.json` for MAE/R² rather than hardcoding), share cards (OG `og-1200x630.png` + `og-1080x1920.png`, Web Share API, `?l=<CODE>` league copy/paste, localStorage-only).

Pages: `/` cockpit, `/play` Daily Guess Wordle 6 tries cosine 32-d + Lab fusion A+B=C, `/players` radar 360×280 Top Similar cosine, `/model` How vector knows, `/methods` doctrine, `/trends` drift, `/dashboard` metrics, PWA `manifest.json` `sw.js` v66 CORE24 DENY8.

## Data pipeline

```bash
python pipeline/fetch_nflverse.py      # planned: nflverse 2020-2025 nflreadpy + weather + Vegas → train_matrix.npz
python pipeline/train_mtnn.py          # now exists: ResidualTower + TransformerFusion 128d 4L4H CLS→32-d L2, emits vectors.json + mtnn.pt
python pipeline/eval_next_game.py      # MAE/R² evaluator → assets/eval_scoreboard.json
python pipeline/export_onnx.py         # ONNX wrapper <300KB gz, INT8 quant path, ExecuTorch stub
```

Features: 10 families holistic 160 feats (`rushing 30, usage 16, form 20, redzone 20, snaps 12, age 8, weather 10, vegas 8, rest 10, def_vs_pos 16`), masking `cat([x·m,m])` per RealMLP pattern. Fetch steps: load_pbp seasons=[2020..2025], roster + snap counts + participation, weather Open-Meteo join wind/temp/dome, Vegas nflverse betting spread/total/implied, compute lag 1-3 FPTS PPR rolling avg redzone def-vs-pos SOS_NET analog, per-season RobustScaler fit, player-split honest.

## Training

AdamW (≈ `lr 1e-3 → cosine`, weight decay MoE L1 `1e-4`), MAE/R² gating (`candidate.json` staging first, promote only if new 32-d beats legacy 16-d on real nflverse eval, `current_repro` vs `claimed`), composite:

```
loss = MSE_fantasy*1.0 + 0.2*aux (arch + pos CE + profile) + SupCon 0.15 + MoE L1 + KL TabPFN T=2 w=0.15
```

Train: `python pipeline/train_mtnn.py --epochs 150 --dim 32 --d-model 128 --n-layers 4 --n-heads 4 --scaling robust --era-align procrustes --family-drop 0.1 --supcon-w 0.15 --arch-w 0.35 --pos-w 0.2`. Synthetic smoke: `--synthetic --epochs 2` validates import + 32-d native + 16-d compat RealMLP path.

## Running locally

```bash
python -m http.server 8000   # open http://localhost:8000
python -m pytest -q          # pipeline gates (needs dev extras in pyproject.toml)
```

Vercel static (`cleanUrls: true`), redirect `gridiron.jcamd.com → gridiron.dumbmodel.com` via `vercel.json`.

## License

MIT — Copyright (c) 2026 J. Cameron Davis. Solo personal project, no connection to employer.
