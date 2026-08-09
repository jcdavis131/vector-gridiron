# vector-gridiron — REAL-data multi-target benchmark

End-to-end, reproducible pipeline that wires **real nflverse data** into the
vector-bench multi-target harness, trains the repo's MTNN with **three heads**
on CPU, and scores it against the full baseline gauntlet — one honest verdict
per registry target.

## What's here

| file | what it is |
| --- | --- |
| `build_dataset.py` | Fetches real nflverse weekly player stats (2019–2024) via the repo's committed `pipeline/fetch_nflverse.py`, constructs forward-shifted labels for `next_game_fpts` / `next_game_yards` / `next_game_tds`, writes `data/gridiron_bench_dataset.npz` + `data/datasheet.json`. Fails loudly if the fetch fails — **no synthetic fallback**. |
| `run_real_benchmark.py` | Loads the dataset, trains `pipeline/model.py`'s MTNN (family towers + TransformerFusion) with 3 regression heads (seeded, CPU, early-stopped on val), runs `vector_bench.runner.run_domain_benchmark` (full ladder + MTNN rung per target), writes `benchmark_report.json` (schema 1.1) + `training_config.json`. |
| `data/gridiron_bench_dataset.npz` | The wired dataset (2.7 MB): raw `X`/`M` (31,670 rows x 160), per-target labels/masks, current-game stats, split indices, entity/time ids. |
| `data/datasheet.json` | Provenance: source URLs, seasons, per-target label construction + observed stats, split spec, leakage notes. |
| `benchmark_report.json` | Schema-1.1 domain report produced by `run_real_benchmark.py` — every number in it comes from that run. |
| `training_config.json` | Exact training config + early-stopping outcome of the committed run. |

## Data (REAL, verified)

- Source: [nflverse-data](https://github.com/nflverse/nflverse-data) GitHub release
  CSVs (`player_stats/stats_player_week_{season}.csv`), CC-BY 4.0. No API key.
- Seasons 2019–2024, positions QB/RB/WR/TE: 31,670 player-game rows,
  1,108 distinct players. 2025 files are not published under the pipeline's known
  URL templates (HTTP 404 as of 2026-08-09), so 2024 is the most recent season.
- Spot-checked against reality: e.g. Patrick Mahomes' 2023 week-1 row carries a
  next-game label equal to his actual week-2 PPR line; 2024 PPR leaders are
  Lamar Jackson / Saquon Barkley / Josh Allen.

## Label construction (forward-shifted, leakage-safe)

For a row (player, game *g*): features come from `engineer_features` using only
information up to and including game *g*; each label is the stat in the player's
**next recorded game within the same season** (`groupby(player, season).shift(-1)`):

- `next_game_fpts` — nflverse `fantasy_points_ppr` (used directly; NaN→0 before shift)
- `next_game_yards` — `rushing_yards + receiving_yards + passing_yards` (each NaN→0)
- `next_game_tds` — `rushing_tds + receiving_tds + passing_tds` (each NaN→0)

Rows whose player has no following game in that season are dropped (same rule as
the committed pipeline's `fpts_next`). An invariant check over all 28k+ adjacent
kept pairs confirms `label(row g) == current-game stat(row g+1)` exactly.

## Split (temporal, strictly forward)

`time_key = season*100 + week`, harness `time_cut = 202400`:

| slice | seasons | rows | who fits on it |
| --- | --- | --- | --- |
| train | 2019–2022 | 20,672 | MTNN gradient steps; all preprocessing stats |
| val | 2023 | 5,471 | MTNN early stopping only (baselines fit on train+val) |
| test | 2024 | 5,527 | evaluation only — never visible at fit time |

## Ladder

vector-bench defaults (`dummy_mean`, `persistence` = last train label per player,
`ridge`, `pca_ridge(n=16)`, `knn(k=10)`, `hist_gbm`, `mlp`) **plus** the mandated
`persistence_current_stat` rung (predict next-game stat = current-game stat), and
the trained MTNN rung per target. Primary metric per target: `spearman_ic`.

## Results (committed run)

Domain headline: **MTNN beats best baseline on 2/3 judged targets (primary
metric spearman_ic); baseline wins the rest.** Wins and the loss, plainly:

| target | best baseline | baseline IC | MTNN IC | delta | MTNN beats? |
| --- | --- | --- | --- | --- | --- |
| `next_game_fpts` | hist_gbm | 0.6820 | **0.6842** | +0.0022 | **yes** |
| `next_game_yards` | **hist_gbm** | **0.7389** | 0.7377 | −0.0012 | **no** |
| `next_game_tds` | hist_gbm | 0.4632 | **0.4663** | +0.0031 | **yes** |

Full per-method metrics (mae/rmse/r2 included) are in `benchmark_report.json`.
Notable context: the MTNN also posts the best mae/rmse/r2 on `next_game_fpts`
(mae 4.585 vs hist_gbm 4.620), while on `next_game_yards` hist_gbm is better
across the board (mae 26.42 vs 27.04) — an honest baseline win. The mandated
`persistence_current_stat` rung lands mid-ladder everywhere (IC 0.583 / 0.665 /
0.329), well above naive persistence but far below the learned rungs.

MTNN training (committed run, `training_config.json`): early-stopped at epoch
48, best epoch 33 (val MSE(z) 0.4626), 172,528 params, ~5 min on 2 CPU threads.

## Reproduce

```bash
# deps: numpy, pandas, torch (CPU), scikit-learn, plus editable installs of
# vector-core + vector-bench from the vector-hub monorepo:
#   pip install -e <vector-hub>/packages/vector-core -e <vector-hub>/packages/vector-bench
OMP_NUM_THREADS=2 python bench/build_dataset.py        # ~1 min (network fetch)
OMP_NUM_THREADS=2 python bench/run_real_benchmark.py   # ~10–20 min on 2 CPU threads
```

Both scripts are fully seeded (`SEED = 0`); the dataset build is deterministic
given the (static) published season files, and the benchmark re-run reproduces
the committed report on the committed dataset.
