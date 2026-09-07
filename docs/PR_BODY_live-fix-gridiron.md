# weekend/live-fix-gridiron

**What and why.** gridiron.dumbmodel.com's onboarding tour showed static literals
("9 seasons", "4.78 model MAE", "trained on 2016-22 ... tested on (2024)") that never
updated as the model was retrained/re-evaluated since the initial commit (`e63a4bb`).
None of the three literals matches any served JSON. This fix replaces them with neutral
placeholders filled at runtime from the same served data the rest of the page already
uses.

**Measured evidence** (each old literal checked against the live-served JSON; none match):
- "9 seasons" vs. `vectors.json`'s real `seasons` array — 27 (1999-2025).
- "4.78 model MAE" / "5.13 guess-the-average" vs. `projections.json`'s real
  `model.report` — `model_fpts_mae` 4.296, `baseline_seasontodate_mae` 4.523. Neither 4.78
  nor 5.13 appears in any served JSON, exact or ±0.005.
- "trained on 2016-22 ... tested on (2024)" vs. the model's own `selection.split`
  ("train<=2023 / val 2024 / test 2025") and `test_season` (2025) — grepped all
  `assets/*.json`; no file anywhere produces a "2016" training-start figure (the only hit
  was an unrelated affine-calib coefficient, 1.2016).

New `nuxUpdateStats()` in `assets/app.js`, called right after `labelSeasons()` in `boot()`
(same ordering guarantee `labelSeasons()` already relies on), fills 5 placeholder spans:
`nux-seasons` (27), `nux-mae-model` (4.296), `nux-mae-baseline` (4.523, chosen over the
trailing-4-game baseline as the closer match to "eyeballing the averages"),
`nux-train-cutoff` (2023, via regex on `selection.split`), `nux-test-season` (2025).

**Verified, and how.**
- Served a copy of the actual output dir (repo root — `vercel.json` sets no
  `outputDirectory`) via `python -m http.server 8791`; curled `/`, the new-token
  `assets/app.js`, `assets/{gridiron,motion}.css`, `assets/{vectors,projections}.json` —
  all 200, byte counts matching L1's recorded live "before" curls.
- Grepped served `/` for the 5 new element ids (all present) and for
  `4.78|5.13|9 seasons|2016.?22|\(2024\)` (none found).
- Ran `nuxUpdateStats()`'s exact logic in `node` against the worktree's own
  `assets/{vectors,projections}.json`: `{seasons:27, model_mae:4.296, baseline:4.523,
  cutoff:"2023", test_season:2025}` — matches the served spans.
- `node --check assets/app.js`: syntax OK.
- Server killed; `Get-NetTCPConnection` confirmed port 8791 no longer listening; a
  follow-up curl failed with connection refused (exit 28).
- Bumped `assets/app.js`'s cache-bust token (`?v=backtest-1 → ?v=nux-honesty-1`): live
  `app.js` is served `Cache-Control: public, max-age=3600, stale-while-revalidate=86400`
  (verified via curl against gridiron.dumbmodel.com), so an unbumped token would leave
  already-cached visitors served the new HTML's "-" placeholders forever.
- Repo test suite (`pipeline/test_feature_hygiene.py`, the only test file) requires
  `pipeline/data/{train_matrix.npz,feature_manifest.json}`, gitignored and absent from a
  fresh worktree — ran it anyway per guard 11 (home checkout porcelain empty, no gridiron
  job running, no hardcoded home path in the script); it exits 1 with a "run
  build_features.py" message, a pre-existing gap unrelated to this frontend-only fix.

**Explicitly NOT done.** D2 from the L1 audit (5 of 7 served JSON assets are live with
dates newer than any git ref, deployed by an out-of-git weekly-refresh path) is a
product/process decision, not a code defect in this repo's files — left untouched per the
brief.

**Merge target and blocker.** Base: `origin/master` (`a6fc91ea`), 1 commit ahead, clean —
same base commit as `weekend/depth-charts-2025-schema`; the two branches touch disjoint
files (`assets/app.js`+`index.html` here vs. `pipeline/build_features.py`+`pipeline/nfl_data.py`+
a new test file there) so they do not conflict with each other. No blocker.
