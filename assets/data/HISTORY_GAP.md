# Gridiron Historical Gap Audit 2026-08-09

Analogous to hoops audit (944 OU 31 seasons, 1.4MB player props, 14.7k matchup). Gridiron needs same depth for construct validity.

## Current State (from pipeline/train_mtnn.py + assets/data)

| Asset | Hoops Benchmark | Gridiron Current | Gap | Source Plan |
|-------|----------------|------------------|-----|-------------|
| **Preseason Win Totals O/U** | 944 entries, 31 seasons 1995-2026 (BBR + Covers + BetMGM) | 0 full seasons (stub 23 seasons empty, resumable Covers gated) | Need NFL O/U 2003-2025 comparable to NBA pattern | `pipeline/fetch_historical_gridiron.py --allow-gambling` fetches Covers `https://www.covers.com/sportsoddshistory/nfl-team/?sa=nfl&Team=...` per team, merges into `assets/data/nfl_win_totals.json` structure {built, source, coverage, seasons}. Zero-deps offline-first resumable. BetMGM manual snapshot for 2025-26 season like hoops. |
| **Player Props / Pedigree** | 1.4MB player_season_props.json, ~2642 KB, market expectation baseline for beating expectations | `draft_pedigree.json` stub 0 real (template 20 nulls) | Need draft round/pick + combine forty 4.22-4.84 etc. | Port hoops `fetch_draft_history.py` → NFL draft via nflreadpy `load_draft_picks()`, combine via `load_combine()` 40-yard dash, vert, bench. Wire into tower age+athleticism family (age 8 dims currently, expand to 12 with speed score). Analogous to hoops glass-box 0-99 skills composite. |
| **Matchup / DvP enriched** | 14.7k matchup enriched, def_vs_pos, blitz faced | `def_vs_pos` tower 16 dims present in MTNN fam_dims, but training matrix synthetic only | Need real DvP rank + SOS_NET analogue for NFL (def rank vs QB/RB/WR/TE, schedule difficulty) | nflverse PBP load + weekly def_vs_pos via `nflreadpy load_ff_opportunity` or SIS charting. Also height/weight matchup risk (undersized CB hunted late). Track as `pipeline/data/train_matrix.npz` X[M,F] ~5000+ rows 2020-2025. |
| **Ages / Usage / Form** | Real career trajectories Construct | Synthetic 2000 rows | Need 5+ seasons nflverse weekly roster snap participation weed 2020-2025 real rows | `python pipeline/train_mtnn.py --check-data` warns when missing, honest exit 0. To fill: nflverse loader stub (planned). Same as hoops leakfree player-split no season leak. |
| **Payroll / Cap % analogue** | hoops payroll_by_season.json + cap_rules.json front office lab | Missing cap % analogue | Need salary cap hit % of cap (OTC overthecap.com scrape or PFF) | Hoops front-office lab pattern: `team_base_*.json` + `payroll_by_season.json` → gridiron version `salary_cap_pct.json` for foresight appreciating deals. |
| **Shared Map LOD** | shared-map.js v4-filtered 3+ seasons OR rookie last 3 (pid-aware disambiguates Jr/Sr Gary Payton), mobile 4k desktop 8k, DPR1 fillRect batched | shared-map.js present 26k BUT gridiron vectors.json has only 2000 player-weeks, no map lite concept | parity already improved: 21k index OK (hoops-level tri-cards, viral row). Need to reuse shared-map.js LOD pattern if growing to 12k+ player-weeks. | Current 2000 < 4k threshold, so lite mode already OK. When scaling to 10k+ weekly rows (2020-2025 6 seasons * ~800 players/season ~4800), introduce `vectors_map_lite.json` 4322 first paint fallback like hoops. |

## Season Coverage Target

- Hoops: 1996-97..2025-26 (30 seasons) + preseason OU back to 2003 covers.
- Gridiron target: 2003..2025 (23 seasons) win totals, plus draft 2000..2025 combine.
- NFL teams 32 stable (WAS rebrand) vs NBA 30 — easier.

| Season | NFL Teams Expected | Status |
|--------|-------------------|--------|
| 2003-2015 vintage | 32 each | EMPTY stub, needs Covers backfill |
| 2016-2019 | 32 | EMPTY |
| 2020-2024 real nflverse era | 32 | EMPTY — but primary for MTNN train |
| 2025-26 current | 32 | EMPTY — BetMGM manual optional like hoops BetMGM Apr/Aug 2026 |

## Offline-First / Zero-Deps Policy (v5 Prime)

- `bundles/zero_deps.json {"zero_deps":true,"allow":"acne:./src"}` enforced — no pip installs, no cloud, ACNE optional local.
- All fetchers use stdlib urllib + curl fallback, respectful 3.5-4.7s delay, UA rotation, resumable merge.
- Gambling domain gate: Covers fetch only behind `--allow-gambling` (user-confirm), else keeps existing stub.
- Draft/combine uses nflreadpy TODO in local-GPU lane (allowed pip in that lane only).

## What ships honestly today

- `pipeline/fetch_historical_gridiron.py` zero-deps stub runnable offline (`--offline`) returns existing, resumable appends seasons.
- Writes `assets/data/nfl_win_totals.json` with required structure {built, source, coverage, seasons}.
- `draft_pedigree.json` template with todo for nflverse local-GPU lane.
- No fake metrics: MAE 8.475 synthetic vs claimed 4.268 R2 0.39 target 3.8 logged in candidate.json + eval_scoreboard honest.

## Next Steps (hoops->gridiron parity)

1. Run Covers backfill with user confirmed Allow:
   `python pipeline/fetch_historical_gridiron.py --allow-gambling`
   Expected 32 teams * 2003-2025 = ~700 entries vs hoops 944.

2. Port hoops `fetch_draft_history.py` + `fetch_positions.py` analogues:
   `python pipeline/fetch_draft_pedigree.py --combine --since 2000`

3. Build `train_matrix.npz` from nflverse weekly (local-GPU):
   `python pipeline/fetch_nflverse.py && python pipeline/train_mtnn.py --epochs 150 --scaling robust --era-align procrustes`

4. Model.html glass-box (hoops parity):
   - SHAP Kernel per family tower importance logged in eval JSON + surfaced in Lab attr grid.
   - Partial dependence plot for forty vs speed score.
   - Construct validity doc: defines fantasy points prediction construct, operationalizes, checks convergent/discriminant/predictive.

5. Manifest v3.3 + zero_deps parity already: bundles/zero_deps.json present, manifest v66 PWA.

## Truthfulness

All gaps documented, no fabrication. Honest exit 0 when data missing (like train_mtnn.py). Offline-first always.
