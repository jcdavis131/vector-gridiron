# Vector Gridiron — Data Model

**Date:** 2026-08-09  
**Repo:** `jcdavis131/vector-gridiron`  
**Embedding:** 32-d native L2 + 16-d compat slice re-L2  
**Synthetic demo:** 2,000 player-weeks (`pipeline/data/embedding_gridiron.npz` 2000×160) → target 5,323 gridiron rows nflverse weekly

## 1. Feature Families — 10 holistic 160 feats

| family | dim | example features | notes |
|--------|-----|------------------|-------|
| rushing | 30 | rush_att, rush_yds, rush_td, ypc, explosives, first_downs, broken_tackles, usage_score | holistic rushing |
| usage / targets | 16 | targets, receptions, air_yards, slot%, routes, share | WR/TE usage |
| form | 20 | lag PPR 1-3, rolling avg 3g, stdev, momentum, YAC, recency weight | lag fantasy |
| redzone | 20 | RZ opp, RZ TD, inside-5 att, goal-line share, conversion | scoring opp |
| snaps | 12 | snap%, routes, route%, snap_share, participation, o-snap% | snap context |
| age | 8 | age, year_in_league, exp_curve, peak delta, 90th age curve | age/experience |
| weather | 10 | wind, temp, dome flag, precip, humidity, altitude adj | game context |
| vegas | 8 | spread, total, implied points, over/under leverage, fav flag | Vegas lines |
| rest | 10 | rest_days, B2B, Thursday short, bye, travel miles, mini-bye | rest/travel |
| def_vs_pos | 16 | SOS_NET analog, opponent def rank vs QB/RB/WR/TE, allowed PPR last 5g | matchup |

Sum: 30+16+20+20+12+8+10+8+10+16 = 160.

Masking: `cat([x·m,m])` where `m∈{0,1}^{160}` missing mask per RealMLP pattern → `d_cat*2 →96h GELU LN→24d` per ResidualTower + skip depth blocks.

## 2. Holistic Usage Context

- **Snaps:** snap% routes route% snap_share participation — prevents counting backup weeks same as workhorse.
- **Age:** age + YEAR_IN_LEAGUE exp_curve — handles rookie vs vet peak curves (RB age cliff, WR prime).
- **Weather:** wind temp dome precip — cold/wind kills passing, dome boosts.
- **Vegas:** spread total implied — favorite + high total = volume/upside.
- **Rest:** rest_days B2B Thursday short bye travel — rest freshness affects explosive usage.
- **Defense-vs-Position:** SOS_NET analog opponent allowed PPR last 5g vs position — matchup difficulty not uniform.
- **Form:** lag 1-3 PPR rolling avg momentum — captures recent hot/cold without leaking next-game.

## 3. Procrustes Era Chain

Season→root rotation-only orthogonal Procrustes drift correction:

```
shared = players ≥30 w/ ≥2 seasons both sides
Q_season = argmin_R ||E_root_shared·R - E_season_shared||_F  s.t. RᵀR=I
E_aligned = E_season · Q_chain_season→root
```

Frobenius residual tracked per chain link. Hoops same pattern adapted for NFL 2020-2025.

## 4. RealMLP Preprocessing

- Per-season RobustScaler median/IQR per tower (not global StandardScaler) → honest era separation, no train/eval leakage across seasons when fit per-season.
- Clip [-3,3] after scaling.
- PLE `k=8 d_out16`: periodic sin/cos embedding `k=8` frequency bins → `2k→16` proj linear, preserves numeric continuity for age/weather/Vegas scalars.
- Missing mask preserved `cat([x·m,m])` → tower learns explicit missing vs zero.

## 5. Fusion + Embedding

- Towers 10×24d (5,323→160→10×24) + season embedding 12-d → `544+12=556→CLS128 4L4H`.
- TransformerFusion `d_model128 n_heads4 n_layers4 pre-LN dropout0.15 ff512` → CLS token `128→512→32-d L2` native.
- Legacy compat `emb_native[:,:16]` `F.normalize` 16-d.

## 6. Targets

- Next-game FPTS PPR (float32) primary.
- Aux heads: archetype 8, position 4 CE, profile 24-d next-week style.

## 7. Evaluation

`assets/eval_scoreboard.json`:

- `current_repro` MAE 8.475 R² -7.35 synthetic expected high.
- `claimed_MAE_next_game` 4.268 R² 0.39 old offline not reproducible until fetch.
- Target 3.8 via MoE + TabPFN distill KL T=2 w=0.15.

## 8. Provenance

- Source hashes 7 `da3a047…` in `assets/data/gridiron.json` → 7/7 honest DM_PROVENANCE pattern dumbmodel.com hub.
- No backend, static Vercel, localStorage league codes `vectorGridiron.v1`.
