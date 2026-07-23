# Vector Gridiron

A fantasy-football cockpit: a static site that ranks weekly lineups using next-game fantasy-point predictions from a multi-tower neural net trained on nflverse data (usage, snaps, age, weather, Vegas lines, rest, defense-vs-position — no player tracking).

Live: https://gridiron.dumbmodel.com/ (gridiron.jcamd.com redirects there)

> Solo personal project, no connection to employer, built with public/free-tier only.

## What's here

- `index.html` + `assets/app.js` — the cockpit UI: canvas map with custom shaders (plain WebGL/canvas, no engine), lineup board, share cards. Mobile-first responsive (`assets/responsive.css`).
- `dashboard.html` — the model-lab view; it reads model metrics (MAE / R²) from the data files it is served with rather than hardcoding them.
- `scripts/` — model export utilities (`export_onnx.py`, `export_executorch.py`, `tabpfn_distill.py`).
- `assets/era_procrustes_align.py`, `assets/realmlp_preproc.py` — preprocessing/alignment code used by the training pipeline.

Model status, stated plainly: training happens offline and the training pipeline is not in this repo. The most recent offline run reported next-game MAE 4.268 (R² 0.39); treat that as a claimed number — the eval that produced it is not reproducible from this repo alone (the `pyproject.toml` description says "claimed" for the same reason).

## League share flow

- `?l=<CODE>` league codes stored client-side (`vectorGridiron.v1` localStorage keys)
- Result-card copy/paste + Web Share API, OG image for links
- No backend; boards are device-only

## Deploy

Vercel static (`cleanUrls: true`), redirect gridiron.jcamd.com -> gridiron.dumbmodel.com via `vercel.json`.

MIT. Solo personal project, no connection to employer, built with public/free-tier only.
