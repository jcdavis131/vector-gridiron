# Data Sources

Vector Gridiron's training matrix is built from **real nflverse public data** by
`pipeline/fetch_nflverse.py`. A synthetic generator
(`train_mtnn.py --synthetic`) remains available as an offline fallback.

## nflverse — player weekly stats

- **Project:** [nflverse](https://github.com/nflverse) — a community-maintained
  set of open NFL data packages (`nflverse-data`, `nflreadr`, `nflfastR`).
- **Dataset used:** weekly player stats from the `player_stats` GitHub release of
  [`nflverse/nflverse-data`](https://github.com/nflverse/nflverse-data/releases/tag/player_stats).
- **Access:** plain HTTPS CSV download from the GitHub release assets. **No API
  key, no authentication, no rate-limited API.** The fetcher tries the known file
  templates in order (the project has renamed these over time):

  ```
  .../releases/download/player_stats/stats_player_week_{season}.csv
  .../releases/download/player_stats/player_stats_{season}.csv
  .../releases/download/player_stats/player_stats_{season}.csv.gz
  ```

### What we fetch and how it maps to feature families

The model (`pipeline/model.py`) expects 10 feature families summing to ~150
columns, padded to a 160-wide matrix. Weekly player stats populate the families
that can be derived from box-score usage and scoring; families that live in other
nflverse releases (or external sources) are emitted **masked** (value `0`,
mask `0`) so the RealMLP `cat([x*m, m])` masking path and the on-disk schema stay
identical to the synthetic matrix.

| Family       | Source                          | Status  |
|--------------|---------------------------------|---------|
| `usage`      | targets, receptions, target/air-yards share, WOPR, RACR, pass usage | **real** |
| `rushing`    | carries, rushing yards/TDs/EPA/first downs/fumbles, YPC | **real** |
| `form`       | lagged & rolling fantasy-point form (prior games only) | **real** |
| `rest`       | schedule-derived (week, weeks-since-last, bye, playoffs) | **real** |
| `snaps`      | `snap_counts` release (not yet wired) | masked |
| `age`        | `rosters` release (not yet wired)     | masked |
| `weather`    | games/weather source (not yet wired)  | masked |
| `vegas`      | betting-lines source (not yet wired)  | masked |
| `def_vs_pos` | matchup aggregation (not yet wired)   | masked |
| `redzone`    | play-by-play aggregation (not yet wired) | masked |

**Target (`fpts_next`)** is the player's *following-game* PPR fantasy points
(`fantasy_points_ppr`, falling back to `fantasy_points`). Form features use only
prior games, so there is no target leakage. Rows without a following game (a
player's last game in a season) are dropped.

### Output schema

`fetch_nflverse.py` writes `pipeline/data/train_matrix.npz` with the **same keys
and dtypes** the synthetic generator emits, so `train_mtnn.py` consumes it
unchanged:

```
X [N,160] f32 · M [N,160] f32 · fpts_next [N] f32 · pos [N] i64 ·
seasons [N] str · season_ids [N] i64 · features [160] str · player_ids [N] str
```

A sibling `pipeline/data/train_matrix.meta.json` records provenance
(`source: "nflverse"` vs `"synthetic"`). The trainer prefers the real matrix when
present and reports which source it loaded.

### Usage

```bash
# real data (preferred)
python pipeline/fetch_nflverse.py --seasons 2021 2022 2023
python pipeline/train_mtnn.py

# quick offline check (tiny slice, writes nothing)
python pipeline/fetch_nflverse.py --dry-run

# synthetic fallback (offline / smoke)
python pipeline/train_mtnn.py --synthetic
```

## License / attribution

nflverse data is released under the **Creative Commons Attribution 4.0
International (CC-BY 4.0)** license. When publishing results or redistributing
derived data, attribute nflverse (see the
[nflverse-data repository](https://github.com/nflverse/nflverse-data) and
[nflverse.com](https://nflverse.com)). Underlying NFL statistics are aggregated
from public sources by the nflverse community; this project fetches only the
published open CSV releases and stores no credentials.

This repository's own code is MIT-licensed (see `LICENSE`). The nflverse data it
fetches at runtime is **not** vendored into the repo and retains its CC-BY 4.0
terms.
