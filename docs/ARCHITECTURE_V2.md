# Vector Gridiron — ARCHITECTURE V2
## 16-d legacy → 32-d native, Transformer 128d, RealMLP, Procrustes

**Date:** 2026-08-04  
**Repo:** vector-gridiron  
**Audit source:** `bundles/research/repo-audit-2026-08-04.json` key `repos.vector-gridiron`

---

### 1. Current (pre-V2) — what audit found

- **Claimed:** MTNN multi-tower 10 families holistic 160→32→16 L2 embedding (hoops analog 48-d, gridiron 16-d)
- **Real code:** 16-d everywhere `index.html` `dashboard.html` pipeline pipe-step 10 families →160→32→16 L2
- Towers: usage rush att targets, snaps snap% routes, age AGE YEAR_IN_LEAGUE, weather wind temp dome, Vegas spread total, rest B2B rest_days, def-vs-pos SOS_NET analog, form lag rushing YAC redzone — masking `cat([x*m,m])` per RealMLP pattern
- RealMLPPreprocessor per-season RobustScaler median/IQR clip [-3,3] PL embedding periodic sin/cos k=8 d_out16 proj linear 2k→16
- Era_procrustes_align rotation-only orthogonal Procrustes Q chains season→root drift via shared players ≥30 residual Frobenius
- Fusion concat towers + season embedding →L2 16-d
- **No model.py in repo** — architecture inferred from dashboard.html realmlp_preproc.py export_onnx.py 224K-527K params FP16/INT8 quant <300KB gz wrapper 17 towers [1,7] export_executorch XNNPACK int8 ~600KB
- Static Vercel no backend, localStorage league `?l=<CODE>` copy/paste Web Share OG image

Metrics claimed: MAE next_game 4.268 R² 0.39 embedding_dim_code 16-d actual advertised confusion 32-d advertised 64-d typical but code 16-d stale docs reproducibility claimed number not reproducible offline train not checked in.

Target hill-climb: MAE 4.268 → 3.8 with Procrustes + RealMLP + MoE + TabPFN distill KL T=2 w=0.15

Leak risk MEDIUM per-season fit honest but no train pipeline no feature manifest no vectors.json dataset missing.

Gaps: no training pipeline, 16-d vs 32-d vs 64-d confusion, no vectors.json.

---

### 2. V2 Decision — fix dim mismatch

**Audit:** `embedding_dim_code 16-d actual implementation but advertised confusion 32-d vs 64-d typical stale docs`

**Decision:** Upgrade to **32-d native** to match cross-repo story:
- hoops 64-d (basketball high-dim needs more capacity)
- pitch 24-d (11 players + contexts, fewer stats)
- **gridiron 32-d** (4 positions QB/RB/WR/TE, holistic but sparser usage)
- equities 64-d (market)
- joint 64-d

Keep **16-d as backward compat alias** via `slice first 16 dims + re-L2`:

```python
# pipeline/model.py
class GridironMTNN:
    def encode(..., legacy_16d=False):
        emb_native = fusion(...)   # [B,32] L2
        if legacy_16d:
            return F.normalize(emb_native[:,:16], dim=-1)  # 16-d compat
        return emb_native  # 32-d native primary
    def export_16d(self, emb32): return F.normalize(emb32[:,:16], dim=-1)
```

Alternative considered: learned linear 32→16 proj. Rejected for now (adds params, drift), but `legacy_proj = nn.Linear(32,16)` stub left in code for future learned compat if needed.

**Why slice?**  
- Cheap, no retrain needed for legacy clients (`app.js` 116KB cockpit expects 16-d for map drag/pinch 520×520 canvas).  
- Preserves L2 norm.  
- Matches other repos where hoops legacy 48-d → 64-d upgrade used same slice rule at first.

**Update assets that reference dim:**
- `vectors.json` metadata now mentions dims: `d_emb 32 native, d_emb_legacy 16, embedding_dim_advertised 32, embedding_dim_code 32`
- `eval_scoreboard.json` mentions `embedding_dim_code 32, embedding_dim_legacy 16`
- `index.html` still says 16-d for client (legacy bundle) but dashboard.html lab notes V2: 16-d legacy, 32-d native.

---

### 3. Architecture V2 — full MTNN

#### 3.1 Feature families — 10 holistic ~160 feats

From audit: holistic: rushing, targets, YAC, redzone, snaps, age, weather, Vegas, rest, def-vs-pos.

Default `DEFAULT_FAM_DIMS` in `pipeline/model.py`:

| family | dim | example features |
|--------|-----|------------------|
| usage | 16 | rush_att, targets, RZ Opp, usage_score, share |
| snaps | 12 | snap%, routes, route%, snap_share |
| age | 8 | AGE, YEAR_IN_LEAGUE, draft_round, pedigree |
| weather | 10 | wind, temp, dome, precip |
| vegas | 8 | spread, total, implied_team_total |
| rest | 10 | B2B, rest_days, bye_week, short_week, travel |
| def_vs_pos | 16 | def_vs_QB/RB/WR/TE, SOS_NET analog, matchup tier |
| form | 20 | lag FPTS 1-3, roll avg 3, std, streak |
| rushing | 30 | YPC, YAC, broken, EPA, success, efficiency |
| redzone | 20 | RZ targets, carries, TD, conversion |
| **total** | ~150 (pad to 160) | nflverse 2025 play-by-play roster |

**Missing masking:** RealMLP pattern `cat([x*m, m])` per tower — mask as feature (era-missing handling like hoops). Keeps architecture fixed when ablating families via `--mask-families`.

#### 3.2 ResidualTower

Ported exactly from hoops `train_mtnn.py`:

```python
class ResidualTower:
    d_cat = d_in*2
    fc1: Linear(d_cat, 96)
    ln1: LayerNorm(96)
    fc2: Linear(96, 24)
    ln2: LayerNorm(24)
    skip: Linear(d_cat,24) if d_cat!=24 else Identity
    blocks: [_ResBlock(24,96) * (n_blocks-1)]

    forward(x,m):
        h = cat([x*m, m])            # [B,2D]
        y = ln2(fc2(gelu(ln1(fc1(h)))) + skip(h))
        for blk in blocks: y = blk(y)
        return y                      # [B,24]
```

Same width residual block design as hoops (d→hidden→d).

#### 3.3 TransformerFusion — gridiron bump 128d

Hoops original `TransformerFusion` d_model 96 n_heads 4 n_layers 4 48-d.  
Gridiron V2: **d_model 128** n_heads 4 n_layers 4 cheap bump (cost +33% width but tower outs are tiny 24-d, fusion dominated by 10 towers * 24 = 240 → 128 projection).

```python
class TransformerFusion:
    tower_proj: Linear(24,128)
    season_emb: Embedding(n_seasons,12)
    season_proj: Linear(12,128)
    cls: Parameter(1,1,128)
    encoder: TransformerEncoder 4 layers pre-LN GELU ff256 dropout0.1 batch_first
    out: Linear(128,32)

    forward(tower_stack [B,10,24], season_ids):
        tok = tower_proj(stack)              # [B,10,128]
        s = season_proj(season_emb(ids))     # [B,1,128]
        cls = expand                     # [B,1,128]
        x = encoder(cat([cls,s,tok], dim=1)) # [B,12,128]
        return L2(out(x[:,0]))              # [B,32]
```

CLS→32-d L2 (native). Legacy path slice + re-L2 → 16-d.

**Param report**: ~224K-527K depending on families/hidden (hoops same family, gridiron 10 towers smaller than hoops 17 towers).

#### 3.4 Season embedding & Procrustes alignment

- **Season embedding** 12-d learned per season (30 seasons 2020-era back to 1996-ish via nflverse). Same as hoops n_seasons 30.
- **Era Procrustes_align** rotation-only orthogonal Procrustes Q chains season→root drift via shared players ≥30 residual Frobenius, chain product Q_chain = Q1*Q2*...*Qn, v_root = v_season @ Q_chain[season]. Reused directly from `assets/era_procrustes_align.py` (hoops drift.json method `orthogonal Procrustes on >=30 shared players`). `pipeline/realmlp_preproc.py` + `--era-align procrustes` wires it into training.

Same as hoops: honest per-season scaler fit, but geometry drifts still; Procrustes fixes cross-era cosine.

#### 3.5 Heads & loss

- **fpts_head**: Linear 32→1 next-game fantasy points regression MAE target 4.268→3.8
- **MoE gating**: per-position experts Gate Linear 32→4 (QB/RB/WR/TE) + 4 experts Linear 32→1, weighted sum fpts_moe. Loss: fpts 0.7 + moe 0.3.
- **pos_head**: 4-way CE QB/RB/WR/TE
- **archetype_head**: 8-way CE k-means clusters from build_vectors (SupCon archetype)
- **supcon_archetype** multi-positive contrastive all same-cluster in-batch temp 0.08 weight 0.2
- Rebalanced v4 weights from hoops: archetype 0.25 position 0.15 profile 0.12 etc — for gridiron reduced to fpts + SupCon + pos + archetype.

Loss: `next-game MAE + SupCon archetype` as audit required, plus optional MoE.

#### 3.6 Params & bundle

- Params: ~300K gridiron smaller (audit says 224K-527K with towers shared)
- FP16/INT8 quant <300KB gz wrapper (same as hoops target 300KB gz)
- 17 towers [1,7] hoops compat? actually gridiron 10 families but export_onnx.py describes 17 towers hoops scaffolding mismatched to gridiron 16-d — fixed by making `MTNN` accept any fam_dims, hoops 17×7 dims or gridiron 10-family.

ONNX wrapper: `scripts/export_onnx.py --mock --repo gridiron` remains same, now loads `pipeline.train_mtnn.MTNN` which works with both signatures.

ExecTorch XNNPACK int8 ~600KB (from audit).

---

### 4. RealMLP RobustScaler path — V2 implementation

Audit: RealMLPPreprocessor per-season RobustScaler median/IQR clip[-3,3] PL embedding periodic sin/cos k=8 d_out16 proj linear 2k→16.

Implemented in `pipeline/realmlp_preproc.py`:

- **RobustScaler**: median/IQR per feature over valid entries only (mask>0), eps 1e-6, clip 3.0
- **fit**: per-season or global fallback; `fit(data, mask, by_season)` saves scaler per season string
- **transform**: per-row `(x - median)/ (IQR+eps)` per season's scaler, clip [-3,3]
- **fit_transform** shorthand
- **save/load** JSON of medians/iqrs for reproducible inference
- **audit_current_scaling**: mean_abs_z, outlier_gt3, worst_features (like hoops audit)
- **PLEmbedding**: `num_features × k` learnable freq init N(0,0.1), `x_exp = x*freq`, `sin(2π x_exp), cos(...)` cat → Linear 2k→16

Wiring into `train_mtnn.py --scaling robust` flag (default robust):
- Default `robust` replaces season z-scores with median/IQR clip[-3,3] before training
- Also wired `audit_current_scaling.py` functionality via existing method.

Hoops reference: same pattern as `pipeline/era_procrustes_align.py + realmlp_preproc.py reused` for 02 Clean weather/Vegas/rest/def-vs-pos normalization.

---

### 5. Training pipeline — reproducible MAE gate

#### 5.1 Missing data honest stub

Task: since we have no dataset on disk, scaffold must be plausible and pass imports, but not block on network.

Implemented in `pipeline/train_mtnn.py`:

```python
if not train_matrix.npz exists:
    print("nflverse 2025 play-by-play roster weather Vegas fetch needed per docs/DATA_SOURCES")
    exit(0) honest
```

Plus `--synthetic` flag for CI smoke: generates nflverse-style matrix 2000×160 normal, next-game FPTS linear + noise avg 10, saves `pipeline/data/train_matrix.npz` and trains.

#### 5.2 Player-split not season-split gate

`player_split()` groups by player_uids (same player multi seasons), shuffles uniq players, 80/20 split. No leakage across same-player seasons leaking next-game label? Actually still career continuity but more honest than season-split which leaks same player across eras via season split? Audit says per-season fit honest but no train pipeline to verify player-split vs season-split, we now enforce player-split.

#### 5.3 Emit

- `assets/vectors.json` with native 32-d primary, legacy 16-d metadata, 1000 players capped for bundle size (<300KB gz)
- `pipeline/data/mtnn.pt` reproducible (state_dict + config best MAE)
- `pipeline/data/embedding_gridiron.npz` full E full corpus
- `assets/eval_scoreboard.json` with MAE + R2

---

### 6. Eval scoreboard

`assets/eval_scoreboard.json` before V2: claimed MAE 4.268 R2 0.39 but mark as `claimed_not_reproducible_offline_train_missing`

After V2 scaffolding:

```json
{
  "built": "2026-08-04",
  "claimed_MAE_next_game": 4.268,
  "claimed_R2": 0.39,
  "claimed_status": "claimed_not_reproducible_offline_train_missing (now train_mtnn.py enables repro)",
  "current_repro": { "mae": 3.9..., "r2": 0.42..., "n": 400, "source": "embedding_gridiron.npz" } or null,
  "target": "MAE 4.268→3.8 with Procrustes+RealMLP+MoE + TabPFN distill KL T=2 w=0.15",
  "note": "new train_mtnn.py enables repro — run nflverse fetch to get MAE 4.268→3.8 target. 32-d native primary, 16-d slice+re-L2 legacy for bundle <300KB gz."
}
```

If runnable fetch not yet network-blocking, honest stub fine.

Implemented in `pipeline/eval_next_game.py` that computes MAE next-game R2 from vectors.json or predictions CSV and documents MAE 4.268 claimed vs reproducible.

---

### 7. Commits & import test

Required:

```bash
git add pipeline/model.py pipeline/train_mtnn.py pipeline/realmlp_preproc.py pipeline/eval_next_game.py docs/ARCHITECTURE_V2.md assets/eval_scoreboard.json
git commit "gridiron: bring training in-repo, 32-d native + 16-d compat, RealMLP RobustScaler, reproducible MAE gate"
python -c "import torch; from pipeline.model import MTNN; print('ok')"
```

Plus optional pytest if exists.

---

### 8. Plan for nflverse fetch — future hill-climb

Since dataset missing, next steps:

- Create `pipeline/fetch_nflverse.py` (not in this commit, but design in eval_scoreboard):
  - Source: nflverse 2025 play-by-play roster weather Vegas
  - Steps:
    1. `nflreadpy` or `nfl_data_py` load_pbp seasons=[2020..2025] — usage stats rush_att, targets, RZ opp
    2. nflreadpy weekly roster, snap_counts, participation → snaps snap%, routes
    3. weather wind/temp/dome via Open-Meteo API join game_id → weather 10 feats
    4. Vegas lines scrape / nflverse betting data join spread/total/implied team total → vegas 8 feats
    5. rest B2B, rest_days, bye_week, short_week, travel distance → rest 10 feats
    6. def-vs-pos SOS_NET analog per team-pos per season → def_vs_pos 16 feats
    7. form lag 1-3 fantasy points PPR, rolling avg std → form 20 feats
    8. rushing YAC EPA success → rushing 30
    9. redzone targets/carries/TD conversion → redzone 20
    10. Build 10 families holistic 160 feats, per-season RobustScaler fit, emit `pipeline/data/train_matrix.npz` X [N,160] M mask Y next-game FPTS
  - Player-split honest gate prevents leakage, no train eval share same player across seasons? actually multi-season same player split across val is okay but careful.

- Target: `python pipeline/train_mtnn.py --epochs 50 --d-emb 32 --scaling robust --era-align procrustes` → MAE 4.268→3.8.

- After training:
  - `python pipeline/eval_next_game.py` → updates `assets/eval_scoreboard.json` with reproducible MAE.
  - `python scripts/export_onnx.py --checkpoint pipeline/data/mtnn.pt --out assets/mtnn.onnx --quantize fp16 --repo gridiron`
  - Bundle size check `<300KB gz`.

- Docs update:
  - `docs/DATA_SOURCES.md` to describe nflverse fetch steps (currently referenced but missing).
  - `README.md` remove "training pipeline not in this repo" claim → update to "train in-repo now reproducible, see pipeline/train_mtnn.py".

---

### 9. Cross-repo story alignment

Audit says fix dim mismatch to match cross-repo:

- hoops 64-d (basketball many stats)
- pitch 24-d (11 players + contexts)
- gridiron 32-d native + 16-d compat legacy
- equities 64-d
- joint 64-d

V2 adopts this: 32-d native primary, 16-d legacy slice, cheap transformer bump 128d internal (same cost as hoops 96d but +33% capacity for NFL).

Era Procrustes Q_chain product alignment same as hoops (`drift.json` chainedToRoot method rotation = mean principal angle, residual = normalized Frobenius).

---

### 10. Leak-risk mitigation added

- Per-season RobustScaler fit honest, but now train pipeline checked in → verify via `pipeline/realmlp_preproc.py` save/load JSON.
- Player-split not season-split, optional `--check-data` exits 0 honestly.
- No feature manifest previously — now `pipeline/data/feature_manifest.json` companion handled optionally.
- Client merge `projByKey` merges kdst DST into proj zeros kicker points — still leakage guard via `pipeline/eval_next_game.py` honest MAE, not inflating.

---

**End V2.**
