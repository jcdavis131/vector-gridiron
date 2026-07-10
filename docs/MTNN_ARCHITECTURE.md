# Vector Gridiron MTNN v2 — Architecture

> **Status:** Design for implementation · 2026-07-09  
> **Rigor bar:** Vector Hoops `pipeline/train_mtnn.py` (residual towers, family masks, multi-task heads, promotion gates)  
> **v1 baseline:** `pipeline/train_models.py` — flat 41→128→64→32 trunk, 6 linear heads, MAE 4.313 / R² 0.42 on 2025

---

## 1. Why upgrade (falsifiable)

v1 is a competent weekly regressor but is **not** hoops-rigorous:

| Dimension | Hoops MTNN | Gridiron v1 | v2 target |
|-----------|------------|-------------|-----------|
| Input | ~120 feats, 17–18 masked families | 41 flat feats, no mask | 80–120 feats, ≥8–12 families + masks |
| Encoder | Residual towers → gated/concat fusion → L2 emb | Shared MLP trunk | Residual towers → gated fusion → L2 emb |
| Tasks | archetype, skills, next profile, aux… | 6 regression heads | fantasy + lines + usage recon + pos + aux |
| Leak protocol | player/temporal splits, documented | temporal seasons ✔ | keep temporal; add family coverage report |
| Docs | DATA_SOURCES_DEEP + promote gates | README paragraph | full catalog + gates |
| Promote | purity/recall gates | MAE vs baseline only | MAE + family ablation note + artifact verify |

**House rule (from hoops RESEARCH):** statistical baselines first; ship a bigger net only if held-out MAE improves (or ties with clearly better calibration/uncertainty).

---

## 2. Feature families → towers

See [`DATA_SOURCES_DEEP.md`](./DATA_SOURCES_DEEP.md). Each family `f` is a slice of `Z` with mask `M`:

```
x_f = Z[:, cols_f] * M[:, cols_f]
tower_f: ResidualMLP( [x_f ∥ m_f] → d_tower )
fusion: GatedAttention or Concat( towers ∥ week_emb ) → emb (32–48d, L2-norm)
```

**Default fusion:** gated (hoops v4 default). Transformer fusion is **optional ablation**, not day-one — capacity must earn its keep on ~40k player-weeks.

---

## 3. Heads (football-native)

| Head | Target | Loss | Weight |
|------|--------|------|--------|
| `fpts_ppr` | week PPR | SmoothL1 | 1.0 |
| `rec_yds` / `rush_yds` / `pass_yds` | yards | SmoothL1 | 0.25 each |
| `receptions` | receptions | SmoothL1 | 0.25 |
| `total_td` | TDs | SmoothL1 | 0.35 |
| `usage_recon` | same-week target_share, snap_pct, carries (masked) | SmoothL1 | 0.15 |
| `position` | QB/RB/WR/TE | CE | 0.10 |
| `role` | depth_rank bucket (if present) | CE masked | 0.08 |
| `pedigree` | draft pick quality z (career) | SmoothL1 masked | 0.05 |

Embedding = fusion output (pre-head). Export PCA(3) for map like v1; nearest neighbors in L2 space.

**Rookies:** keep separate draft-capital MLP (v1) until pedigree+combine towers cover them; merge later if MAE wins.

---

## 4. Training protocol

- **Rows:** REG skill player-weeks with ≥1 prior game, seasons 2016→latest.
- **Split (default):** train ≤2023, val 2024, test 2025 (match v1 honesty). Revisit rolling year when 2026 stats exist.
- **Standardize:** μ/σ on train only; per-target y μ/σ for head balance.
- **Optim:** AdamW, wd 1e-4, batch 512, early stop on val PPR MAE, patience 25.
- **Regularization:** dropout 0.15–0.2; optional family-token dropout 0.1.
- **Device:** CPU OK; CUDA if available.

---

## 5. Artifacts

| Artifact | Producer | Consumer |
|----------|----------|----------|
| `pipeline/data/train_matrix.npz` | `build_features.py` | train |
| `pipeline/data/feature_manifest.json` | build | train + inspect |
| `pipeline/data/mtnn_report.json` | train | verify / Methods |
| `assets/nextgame.json` | train/export | UI |
| `assets/projections.json` | train/export | UI |
| `assets/embedding.json` | train/export | map |
| `pipeline/data/mtnn_best.pt` | train | retrain resume |

UI JSON **must** keep: `key,name,pos,team,proj,floor,ceil,line,conditions,avail,bye,…`.

---

## 6. Promotion gates (`verify_accuracy.py`)

| Gate | Pass |
|------|------|
| G1 Shapes | matrix rows = meta; no NaN in unmasked cells |
| G2 Baseline | test PPR MAE < last-4 MAE and < season-to-date MAE |
| G3 Regress | test PPR MAE ≤ v1 MAE + 0.05 |
| G4 Coverage | each active family mask rate documented; no silent 0% family |
| G5 Artifacts | `verify_logic.mjs` exit 0 |
| G6 Honesty | README cites new sources + injury 2025 caveat |

If G3 fails after reasonable epochs/HP: **keep v1 assets**, leave v2 in `pipeline/data/`, document (hoops promote pattern).

---

## 7. Implementation phases

1. **Docs** — SPEC + DATA_SOURCES + this file + tasks *(this gate)*  
2. **Ingest** — nfl_data expand (depth, ngs, pfr, ep download)  
3. **Matrix** — build_features families + masks + opportunity  
4. **Inspect** — feature_inspect  
5. **Train** — train_mtnn + export assets  
6. **Gates** — verify_accuracy + verify_logic  
7. **Wire** — refresh.py calls new path; README Methods  

---

## 8. Explicit non-copy from hoops

Do **not** port: salary head, All-NBA votes, BBRef BPM bridge, InfoNCE career pairs as primary (optional later for comps), Chimera puzzle purity. Football success = **weekly PPR accuracy + honest conditions**, not retrieval purity.
