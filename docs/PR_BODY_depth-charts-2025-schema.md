# weekend/depth-charts-2025-schema

**What and why.** nflverse re-sourced `depth_charts_2025.csv` from ESPN in a new,
timestamped-dump schema (no `week` column), so the existing week-keyed parser
(`index_depth()`) required `week > 0` and silently dropped the entire role family for
every 2025 row (coverage 0.000) while every training-era row (2023/2024) still had it —
the 2025 test split trained blind to a feature family the model relies on. This fix
teaches `index_depth()` to dispatch on the header and as-of join the new dump format onto
`games.csv`, restoring the role family for the test season without touching any row that
already had it.

**Measured evidence** (offline rebuild against `pipeline/cache`, 2026-09-01 mirror; baseline
first, unmodified worktree rebuild is byte-identical to the main tree's `train_matrix.npz`,
sha256 `10f245a6...`):
- role coverage: 2023 `0.9513 → 0.9513` (unchanged), 2024 `0.9411 → 0.9411` (unchanged),
  2025 `0.0000 → 0.9830` (5,335 of 5,427 rows).
- availability coverage 2025: `0.0000 → 0.0000` — untouched; this is the injuries feed
  (`INJURIES_LAST_SEASON=2024`, no `injuries_2025.csv` in cache), not depth charts.
- ngs coverage 2025: `0.0000 → 0.0000` — untouched; no `ngs_2025` asset, and the 2024 files
  are truncated stubs with the same schema as 2023 (missing data, not schema drift).
- rows `<= 2024`: `np.array_equal` on Z, mask, Y, Y_usage, season, week, gsis, name, pos,
  team → all True across 44,448 rows (nothing pre-2025 changed).
- 2025 rows: 5,427 before and after; non-role columns of Z and mask `array_equal` True;
  masked cells with `Z != 0`: 0 (no leakage into masked slots).
- as-of join: 16,078 (week, gsis) keys resolved over 570/570 team-weeks from 64 daily
  dumps; 692 keys dropped as ambiguous (691 two-slots-one-team, 1 two-teams-one-week); 371
  offense rows with an empty `gsis_id` skipped; 0 team-weeks left without a snapshot. 12 of
  the 5,427 2025 matrix rows (all TE, listed at both TE and FB) are dropped by the
  ambiguity rule; the other 80 role-masked 2025 rows are players genuinely absent from
  their team's chart that day.

**Verified, and how.**
- `pipeline/test_depth_chart_schema.py` (new, 308 lines, 8 tests) pins the dispatch, the
  frozen weekly parser (dict-equal to the original 14,749-key 2024 output), the exact join
  logic, the ambiguity drops, and the staleness bound (snapshot must be ≤7 days before
  gameday) — all on schema-only fixtures. 8 passed.
- `pipeline/test_feature_hygiene.py` passes on the rebuilt matrix — all feature-hygiene
  gates passed.
- `nfl_data.py` gained a streaming reader so the 52 MB dump (554,215 rows) is not
  materialized as 554k Python dicts (RAM guard).

**Explicitly NOT done.** Availability (injuries) and NGS coverage for 2025 remain 0.000 —
this fix only restores the depth-chart/role family; those are separate missing-data
questions (different upstream feeds), not schema drift this branch's scope covers.

**Merge target and blocker.** Base: `origin/master` (`a6fc91ea`, the shipped-regime commit),
1 commit ahead, clean. No git-level blocker — this is the branch the operator-facing plan
names explicitly for a PR. **Operational caveat, not a merge blocker:** the GPU queue's
`Protocol.hash()` ignores `repo_path` and the commit, so once the runner's fixed-regime
baseline job (j0035) runs against this branch's worktree, it overwrites
`baselines.json:vector-gridiron.a6fc91ea` with fixed-regime numbers under the same key as
the shipped-regime baseline (62.0900 ± 1.2078) — the shipped panel survives only in
`results/j0005.json`, `results.tsv` (2026-09-06T05:38), and
`results/baseline_snapshot_gridiron_shipped_17dfb97c450c.json`. This is a runner-side
bookkeeping fact for the operator's Monday handoff, not a reason to withhold the PR.
