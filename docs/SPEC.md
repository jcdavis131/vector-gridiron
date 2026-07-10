# Vector Gridiron — SPEC: MTNN v2 (hoops-rigor weekly fantasy)

> **Status:** Draft for single approval (auto-mode plan gate) · 2026-07-09  
> **Parents:** Vector Hoops MTNN v4/v5 doctrine · current `pipeline/train_models.py` (v1)  
> **Research:** [`docs/DATA_SOURCES_DEEP.md`](./DATA_SOURCES_DEEP.md) · [`docs/MTNN_ARCHITECTURE.md`](./MTNN_ARCHITECTURE.md)

---

## ASSUMPTIONS (correct now or we proceed)

1. **Target product** stays the static Vercel cockpit (`nextgame.json` / `projections.json` / embedding map) — no new paid APIs.
2. **Primary metric** remains held-out **next-game PPR MAE** vs last-4 and season-to-date baselines (honest temporal split).
3. **Architecture** moves from flat shared-trunk → **multi-tower + masked families + multi-task heads**, matching hoops rigor — not a copy-paste of basketball heads (no salary/All-NBA); football-native heads instead.
4. **Free nflverse (+ ffopportunity EP download)** only; PFF/FantasyData out of scope.
5. **Injuries 2025+** stay degraded (nflverse source dead); we mask the family and document honesty.
6. **Repo has no git** today — we add planning docs under `docs/` + `tasks/` without requiring a commit until you ask.
7. **UI contract** for `nextgame.json` / `projections.json` player objects stays backward-compatible; we may add fields (`tower_contrib`, `uncertainty`, richer `line`).

→ Reply **yes** to approve this plan and start implementation top-to-bottom. Reply with corrections to any assumption first if needed.

---

## 1. Objective

Build a **Vector-Hoops-class MTNN** for Vector Gridiron that:

- Ingests a **deep, cited, leakage-safe** player-week feature matrix (families + masks).
- Trains a **multi-tower multi-task net** whose shared embedding powers comps/map and whose heads predict weekly fantasy + component lines.
- Beats v1 on held-out season PPR MAE (and does not lose to last-4 / STD baselines).
- Ships Methods-honest documentation of every data source and known gap (injuries 2025+, no in-season routes).

**Users:** fantasy managers using draft / start-sit / next-game tabs.  
**Success:** promotion gate green + `verify_logic.mjs` still passes + MAE improves or ties within noise with richer uncertainty.

---

## 2. Commands

```powershell
cd c:\Users\jcdav\vector-gridiron

# Data / features
python pipeline/nfl_data.py              # smoke: fetch helpers (if __main__)
python pipeline/build_features.py --offline
python pipeline/build_opportunity.py     # EP / RZ (when implemented)
python pipeline/feature_inspect.py

# Train / export
python pipeline/train_mtnn.py --epochs 40
python pipeline/export_projections.py    # or train_mtnn writes assets directly

# Gates
python pipeline/verify_accuracy.py
node pipeline/verify_logic.mjs
```

---

## 3. Project structure

```
vector-gridiron/
  docs/
    SPEC.md                 # this file
    DATA_SOURCES_DEEP.md    # source catalog
    MTNN_ARCHITECTURE.md    # towers / heads / gates
  tasks/
    plan.md
    todo.md
  pipeline/
    nfl_data.py             # expand: depth, ngs, pfr, ep
    build_features.py       # → family matrix + mask + manifest
    build_opportunity.py    # NEW
    feature_inspect.py      # NEW
    train_mtnn.py           # NEW (replaces core of train_models.py)
    train_models.py         # keep as v1 fallback or thin wrapper
    verify_accuracy.py      # NEW promotion gates
    cache/                  # gitignored raw
    data/                   # train_matrix.npz, feature_manifest.json, reports
  assets/
    nextgame.json, projections.json, embedding.json  # contracts preserved
```

---

## 4. Code style

Match existing pipeline: stdlib + numpy + torch, module-level constants, leakage comments in docstrings, `norm_key(name, pos)`, no new heavy deps unless justified (`document-non-action`). Prefer clear family names over clever abbreviations.

---

## 5. Testing strategy

| Level | What |
|-------|------|
| Unit | feature row: prior-week only; mask zeros pre-coverage; shape asserts |
| Inspect | `feature_inspect.py` coverage / NaN / corr flags |
| Train gate | val early-stop on PPR MAE; test season report vs baselines |
| Artifact | `verify_logic.mjs` existing UI contracts |
| Ablation | drop-one-family Δ MAE (document; full sweep optional) |

---

## 6. Boundaries

**Always do**

- Leakage-safe trailing features; temporal train/val/test by season.
- Mask missing families; cite sources in README/Methods.
- Keep `nextgame` / `projections` keys the UI already reads.
- Beat or match v1 baselines before promoting assets.

**Ask first**

- Adding paid data; changing scoring away from PPR primary; deleting v1 path; deploying prod mid-train.

**Never do**

- Train on same-week box score as features.
- Impute targets with league averages.
- Commit secrets / raw HTML scrapes of ToS-hostile sites.
- Claim injury-aware 2025+ predictions without a live injury feed.

---

## 7. Acceptance criteria (Definition of Done)

1. `docs/DATA_SOURCES_DEEP.md` + `MTNN_ARCHITECTURE.md` + this SPEC committed in tree.
2. `feature_manifest.json` lists ≥8 families with per-feature masks.
3. `train_mtnn.py` multi-tower model trains end-to-end offline from cache.
4. Held-out test: `model_fpts_mae` ≤ v1 (4.313) **or** within 0.05 with documented tradeoff + richer heads; must still beat last-4 baseline.
5. `assets/nextgame.json` + `projections.json` regenerate; `verify_logic.mjs` passes.
6. README Methods section lists new sources + injury caveat.

---

## 8. Non-goals (this pass)

- In-season route participation (feed dead).
- PFF grades.
- Per-position separate models as the only path (optional ablation later).
- Full Transformer fusion v5 until concat/gated towers beat v1 (hoops lesson: justify capacity).
