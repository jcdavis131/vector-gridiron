# The availability family measures the wrong side of the injury report

> **Status:** diagnosed, not yet fixed (2026-08-14)
> **Finding:** 3 of the 4 `availability` features are structurally dead — not
> buggy, but definitionally near-zero for the rows the matrix contains. The
> injury data they draw on is rich and is being discarded.

## What the audit flagged

`pipeline/audit_features.py` reports three near-constant columns, all in one
family:

```
a_inj_out         sd 0.004743   coverage 0.8915
a_inj_doubtful    sd 0.009486   coverage 0.8915
a_games_missed_4  sd 0.009486   coverage 0.8915
```

Measured against the matrix, across all 44,448 observed rows:

| feature | rows flagged | share |
|---|---|---|
| `a_inj_out` | **1** | 0.002% |
| `a_inj_doubtful` | 4 | 0.009% |
| `a_games_missed_4` | 4 | 0.009% |
| `a_inj_questionable` | 2,041 | 4.6% |

One player OUT across 44,448 player-weeks is not a plausible NFL rate.

## Why, and why it is not a simple bug

The source is not thin. `pipeline/cache/injuries_2024.csv`, one season:

```
report_status:  Out 1,116 · Questionable 1,513 · Doubtful 194 · blank 3,386
```

The matrix is built from `stats_player_week`, which contains only players who
**appeared in the game**. A player designated OUT does not play, so he has no
weekly stat row — and therefore no row here for the flag to be set on. The three
dead columns ask "was this player unavailable?" of a population defined by
having been available.

`a_inj_questionable` survives at 4.6% precisely because Questionable players
often do play.

So the family is not broken code. It is a correct join answering a question the
row population has already answered.

## The signal being thrown away

Those 1,116 OUT designations describe exactly who is **not** playing — which is
the input that moves everyone else's opportunity. A teammate sitting is what
redistributes carries, targets and snaps, and opportunity is the only thing that
measured as signal in this estate:

- NBA, shuffle-controlled: summed absent-teammate minutes bought −0.108 MAE on
  minutes and −0.051 on DK points, against a control of exactly +0.000.
- NFL, this repo's own data: trailing opportunity (targets/carries/attempts)
  was worth −0.102 MAE against a season-average bar, while trailing *production*
  was worth only −0.035.
- This repo's shipped report agrees without knowing it: `baseline_seasontodate_mae`
  4.523 beats `baseline_last4_mae` 4.616, so recent production is noise.

The availability family currently encodes self-status, which is null by
construction. It should encode **teammate** absence, which is not.

## Proposed fix

Replace or supplement the three dead columns with team-level absence built from
the injury report already cached:

```
a_team_out_snapshare      summed prior snap share of teammates listed Out
a_team_out_same_pos       the same, restricted to the player's position group
a_team_out_count          how many rotation teammates are Out
```

Snap share rather than raw counts, because a backup being Out is not the same
event as a starter being Out. `snap_counts_{year}.csv` is already cached for ten
seasons.

Two honest cautions, from the NBA measurement rather than from theory:

1. **Expect the gain to be small.** In NBA the equivalent feature bought 2% of
   the available headroom. Absence is common enough that a trailing-form
   baseline already absorbs much of it.
2. **Position-weighting did not help in NBA** — a flat team-level sum beat a
   position-matched one, because the whole rotation shifts up rather than only
   the positional peers. NFL may differ, since roles are more rigid there, which
   is exactly why `a_team_out_same_pos` is worth measuring separately rather
   than assuming.

## How to judge it

Through the harness, not by eye:

```
python gpu/climb.py vector-gridiron --desc "team-absence availability features"
```

Baseline is recorded: **CQS 62.0967 ± 1.1260**, `neg_fpts_mae` −4.3502, six
seeds `[5, 7, 13, 21, 42, 99]`.

Note the honest bar is 4.3502, not the 4.268 quoted in `ALIENWARE_HANDOFFS.md`.
4.268 is better than every one of those six seeds — a lucky draw, 1.3 sd below
the mean, not a level the recipe reproduces.
