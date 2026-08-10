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

## Results (current committed run — post improvement-pass)

Domain headline: **MTNN beats best baseline on all 3/3 judged targets**
(primary metric `spearman_ic`) — up from 2/3 in the original merged run (PR #5).
The target that used to lose, `next_game_yards`, now wins:

| target | best baseline | baseline IC | MTNN IC | delta | MTNN beats? |
| --- | --- | --- | --- | --- | --- |
| `next_game_fpts` | hist_gbm | 0.6820 | 0.6829 | +0.0009 | **yes** |
| `next_game_yards` | hist_gbm | 0.7389 | **0.7396** | **+0.0007** | **yes** (was **no**, −0.0012) |
| `next_game_tds` | hist_gbm | 0.4632 | 0.4665 | +0.0033 | **yes** |

Full per-method metrics (mae/rmse/r2) are in `benchmark_report.json`. Honest
caveat: the `next_game_yards` win is **IC-only** — on mae/rmse/r2 hist_gbm is
still slightly ahead (mae 26.42 vs 26.64, rmse 39.51 vs 39.97, r2 0.732 vs
0.726), same as before the improvement pass. All margins here remain tiny
(|delta IC| ≤ 0.0033) — this is a genuine, reproducible, but modest shift, not
a decisive win. The mandated `persistence_current_stat` rung is unaffected and
still lands mid-ladder everywhere.

## Improvement pass (`bench/mtnn-improvement-pass`)

Starting from the honest 2/3 result above, a follow-up pass searched for a
genuine improvement **without touching the dataset or the train/2019-22 /
val/2023 / test/2024 split** — same `bench/data/gridiron_bench_dataset.npz`,
same leakage discipline.

**Hyperparameter search (selected on VAL loss only, never test).** Reference
metric: equal-weighted mean of the 3 raw z-scored-target MSEs at each config's
best checkpoint (val, seed 0) — kept weighting-scheme-invariant so every
candidate is comparable on the same footing as the committed baseline's loss.

| config (vs. committed baseline) | val ref MSE | vs. baseline |
| --- | --- | --- |
| baseline (lr=1e-3, wd=1e-4, d_model=64, equal loss) | 0.4626 | — |
| lr=5e-4 | 0.4683 | worse |
| weight_decay=3e-4 | 0.4629 | ~tied, slightly worse |
| d_model=96 | 0.4635 | worse |
| **per-target uncertainty loss weighting** (same hparams) | **0.4603** | **better** |
| uncertainty weighting + weight_decay=3e-4 | 0.4608 | better, but worse than uncertainty alone |

Width, learning rate, and weight-decay changes did **not** beat the committed
config on validation loss. The one change that did was **how the joint loss
combines the three targets.**

**Per-target loss weighting (the fix for `next_game_yards`).** The committed
run's loss was a flat `mean(MSE_fpts, MSE_yards, MSE_tds)` over z-scored
targets — an equal 1/3 share of gradient regardless of how each target's
optimum trades off against the others. This pass replaces it with
**homoscedastic uncertainty weighting** (Kendall, Gal & Cipolla, *"Multi-Task
Learning Using Uncertainty to Weigh Losses"*, CVPR 2018,
https://arxiv.org/abs/1705.07115):

```
loss = sum_i [ exp(-log_var_i) * MSE_i + log_var_i ]
```

`log_var_i` is one learned scalar per target (3 extra parameters total,
initialized to 0 — equal weighting at init), so the network discovers its own
per-task weighting during training instead of a fixed 1/3-1/3-1/3 split. Across
all 3 seeds it converged to almost the same weights:

| target | learned weight `exp(-log_var)` (seed 0 / 1 / 2) |
| --- | --- |
| `next_game_fpts` | 1.614 / 1.617 / 1.612 |
| `next_game_yards` | **2.799 / 2.811 / 2.884** |
| `next_game_tds` | 1.561 / 1.570 / 1.568 |

The network consistently gives `next_game_yards` **~1.7-1.8x more weight**
than the other two targets — direct, reproducible evidence for the "drowned
out" hypothesis: under equal weighting, `next_game_yards`' gradient share was
too small relative to what it needed, even though its raw (unweighted)
z-scored MSE is the *smallest* of the three (~0.26 vs ~0.56 for fpts/tds) —
it's the "easiest" target in absolute loss terms, so equal weighting was
implicitly under-investing in squeezing out its last bit of accuracy.

**Ensembling (disclosed).** The reported MTNN numbers average the **test
predictions of 3 independently-seeded, independently-trained models** (seeds
0, 1, 2), same architecture and loss, `bench/training_config.json`'s
`member_diagnostics` has each seed's own val loss/epoch/learned weights. This
is a plain seed ensemble, not a new mechanism — disclosed so the win isn't
mistaken for a single lucky seed (seed 0 alone already had a lower val loss
than baseline; the ensemble is the reported, final number).

**What did NOT change:** dataset, split, feature families, architecture
(d_emb=32, d_model=64, 2 fusion layers, 4 heads, `d_tower=24`), optimizer
(Adam, lr=1e-3, weight_decay=1e-4), batch size (1024), preprocessing
(`vector_core.RobustScaler` fit on train only, targets z-scored on train
only). Only the loss's per-target weighting and the seed-ensembling are new.

**Honest framing.** This is a genuine, reproducible improvement on the
harness's judged metric (all 3 targets now beat their best baseline on
`spearman_ic`, up from 2/3), achieved by a documented multi-task technique and
selected without ever looking at test during the search. It is also a modest
one: IC deltas are all still <0.004, `next_game_yards`' absolute-error metrics
(mae/rmse/r2) still favor hist_gbm, and per-seed val loss varied enough
(0.4603–0.4653) that a single unlucky seed could still lose the IC verdict —
the ensemble is what makes the result stable across seeds, not any one run.

MTNN training (current run, `training_config.json`): 3-seed ensemble, each
seed early-stopped around epoch 33-36 (patience 15, max 150), 172,528 params
per member, ~14-15 min per seed on 2 contended CPU threads (~45 min total for
the 3-seed ensemble; the original single-seed run was ~5 min).

## Reproduce

```bash
# deps: numpy, pandas, torch (CPU), scikit-learn, plus editable installs of
# vector-core + vector-bench from the vector-hub monorepo:
#   pip install -e <vector-hub>/packages/vector-core -e <vector-hub>/packages/vector-bench
OMP_NUM_THREADS=2 python bench/build_dataset.py        # ~1 min (network fetch)
OMP_NUM_THREADS=2 python bench/run_real_benchmark.py   # ~30-45 min on 2 CPU threads (3-seed ensemble)

# To reproduce the ORIGINAL (pre-improvement-pass, single-seed, equal-loss) run:
OMP_NUM_THREADS=2 python bench/run_real_benchmark.py --loss-mode equal --seeds 0   # ~5 min
```

Both scripts are fully seeded; the dataset build is deterministic given the
(static) published season files, and the benchmark re-run reproduces the
committed report on the committed dataset. `--loss-mode` (`equal` | `uncertainty`,
default `uncertainty`) and `--seeds` (default `0 1 2`) let you reproduce either
the original committed numbers or the improvement-pass numbers from the same
script.
