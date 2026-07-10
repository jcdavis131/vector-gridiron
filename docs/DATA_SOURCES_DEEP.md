# Vector Gridiron — Deep data sources for weekly fantasy MTNN

> **Status:** Research + implementation plan (2026-07-09)  
> **Parent:** [`SPEC.md`](./SPEC.md) · [`MTNN_ARCHITECTURE.md`](./MTNN_ARCHITECTURE.md)  
> **Doctrine:** Leakage-safe (prior weeks only), mask-honest for era-gated feeds, free/OSS first, cite every source in Methods. Neural net only when it beats honest baselines on a held-out season.

This catalog is the football counterpart to Vector Hoops `docs/DATA_SOURCES_DEEP.md`. Each track owns: source URL, coverage, legal/terms, recommended features, mask strategy, MTNN tower family, and blocked-by.

---

## Global join conventions

| Key | Format | Used by |
|-----|--------|---------|
| Player | `gsis_id` (primary); fallback `pfr_id` / `norm_name\|pos` | all tracks |
| Season / week | `(season:int, week:int)` REG only for training | form, matchup |
| Team | nflverse abbrev (`KC`, `LA`, …) as in `games.csv` | context, DvP |
| Composite | `(gsis_id, season, week)` | train matrix rows |

**Mask rule:** missing or pre-coverage rows get `mask=0` on that feature family; never impute league averages into **training targets**. League-average fallbacks are allowed only for **context features** when explicitly documented (e.g. early-season DvP).

**Leakage rule:** every form / usage / DvP / NGS trailing feature uses weeks **strictly before** the target week. Upcoming-week rows use end-of-prior-season form + real schedule/Vegas for the target week.

---

## What we already use (baseline — keep)

| Feed | Path / URL | Features today |
|------|------------|----------------|
| `stats_player_week` | nflverse `stats_player` release | trailing form (PPR, targets, carries, WOPR, EPA, …), targets |
| `snap_counts` | nflverse | offense snap % |
| `games.csv` | Lee Sharpe / nfldata | home/away, rest, div, roof, surface, temp, wind, kickoff, Vegas spread/total → implied pts |
| `players.csv` | nflverse | age, height, weight, draft, latest_team, status |
| `injuries` | nflverse (⚠ **no 2025+** — source died) | Out / Doubtful / Questionable for UI avail |
| FFC + MFL ADP | public APIs | consensus ADP for draft value flags |

**Gap vs literature / peer OSS:** no depth chart role, no red-zone shares, no NGS tracking, no PFR advanced contact/pressure, no expected-points opportunity layer, no EWMA multi-window form, no teammate/QB context, flat 41-d trunk (no towers/masks).

---

## Evidence: what predicts weekly fantasy

Synthesized from fantasy-analytics practice + OSS models ([ffopportunity](https://github.com/ffverse/ffopportunity), [alexanderdfree/Fantasy_Football_ML_AWS](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS), Fantasy Analytics Authority / Projection Lab methodology):

| Signal class | Why it matters | Priority |
|--------------|----------------|----------|
| **Opportunity / usage** — target share, air yards share, WOPR, carry share, snap %, route participation | Leading indicators of volume; more stable than yards/TDs | P0 |
| **Red-zone / goal-line share** | Highest-variance TD component; must regress | P0 |
| **Vegas implied team total + spread** | Game script / scoring pace prior | P0 (have) |
| **Weather (wind, outdoor)** | Suppresses pass volume outdoors | P0 (have) |
| **Defense vs position** | Matchup adjustment | P0 (have, expand) |
| **Depth chart / injury ahead** | Role shocks week-to-week | P0 |
| **NGS separation / YAC / RYOE / CPOE** | Efficiency quality beyond box score | P1 |
| **PFR pressure / YBC / broken tackles** | QB pressure + RB contact profile | P1 |
| **ffopportunity expected points** | Situation-normalized opportunity | P1 |
| **QB / teammate context** | Pass-catcher depends on QB quality | P1 |
| **Combine / draft capital** | Rookies + pedigree aux head | P1 (partial) |
| **Market ADP / rankings** | Calibration / residual vs market | P2 |
| **FTN charting** | Play design (CC-BY-SA; cite FTN) | P2 |
| **Participation / routes** | Gold for WR/TE; **not in-season** after 2022 (postseason dump only) | P3 / deferred |

---

## Per-source implementation specs

### Track A — nflverse weekly production (expand in-place)

| Field | Detail |
|-------|--------|
| **Fetcher** | `pipeline/nfl_data.py` → `weekly_stats` (exists) |
| **Source** | `https://github.com/nflverse/nflverse-data/releases/tag/stats_player` |
| **Coverage** | 1999+; we train **2016+** (snap/NGS alignment) |
| **Legal** | nflverse open data; attribute nflverse |
| **New features (from existing cols)** | `f_receptions`, `f_racr`, `f_pacr`, `f_cpoe`, `f_yac`, `f_air_yards`, first downs, fumbles lost, pass INTs (QB), multi-window EWMA (3/5/8) of key usage |
| **Mask** | Always on for REG skill rows with ≥1 prior game |
| **Tower** | `form`, `usage` |
| **LOC** | ~80 (feature expand in `build_features.py`) |

---

### Track B — Snap counts (expand)

| Field | Detail |
|-------|--------|
| **Fetcher** | `nfl_data.snaps` (exists) |
| **Source** | `snap_counts` release (PFR-derived) |
| **Coverage** | ~2012+ |
| **New features** | trailing snap %, Δ snap vs prior 3, ST snap % (noise flag) |
| **Tower** | `usage` |
| **Update cadence** | 0/6/12/18 UTC in-season ([schedule](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html)) |

---

### Track C — Schedule / Vegas / weather (expand)

| Field | Detail |
|-------|--------|
| **Fetcher** | `nfl_data.games` (exists) |
| **Source** | `https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv` |
| **New features** | moneylines → win prob proxy; over/under odds; coach continuity flag; stadium_id hash (optional); short-week rest already present |
| **Tower** | `conditions`, `market` |
| **Cadence** | every ~5 min in-season |

---

### Track D — Depth charts (NEW — P0)

| Field | Detail |
|-------|--------|
| **Fetcher** | `pipeline/nfl_data.py` → `depth_charts(year)` |
| **Source** | `https://github.com/nflverse/nflverse-data/releases/tag/depth_charts` |
| **Coverage** | 2001+; **2025+ schema change**: timestamped ESPN updates, not week-keyed — join as-of gameday |
| **Legal** | nflverse; ESPN-sourced post-2024 |
| **Features** | `depth_rank` (1/2/3), `is_starter`, `formation` one-hots (offense), same-position depth ahead count |
| **Mask** | mask family if no chart row as-of kickoff |
| **Tower** | `role` |
| **LOC** | ~200 |
| **Note** | Sample 2024 cols: `gsis_id, week, depth_team, depth_position, formation, club_code` |

---

### Track E — Injuries (NEW join into features — P0 / degraded 2025+)

| Field | Detail |
|-------|--------|
| **Fetcher** | `nfl_data.injuries` (exists for UI) |
| **Source** | `injuries` release |
| **Coverage** | ~2009–**2024**; **2025+ unavailable** (nflverse: source died, no ETA) |
| **Features** | prior-week status one-hot; games missed trailing 4; teammate-same-pos Out count |
| **Mask** | **entire `injury` family mask=0 for season≥2025** until feed returns |
| **Tower** | `availability` |
| **Honesty** | Methods must say 2025+ avail is roster-status / depth-chart only |

---

### Track F — Next Gen Stats weekly (NEW — P1)

| Field | Detail |
|-------|--------|
| **Fetcher** | `nfl_data.ngs(stat_type, year)` |
| **Source** | `nextgen_stats` release (`ngs_{year}_{passing\|receiving\|rushing}.csv.gz`) |
| **Coverage** | **2016–2023** weekly (full); **2024 files on nflverse are empty placeholders** (~0.5KB); **2025 not published** as of 2026-07 — family mask handles gaps |
| **Legal** | NFL NGS via nflverse; attribute |
| **Receiving** | avg_separation, avg_cushion, avg_yac, avg_yac_above_expectation, percent_share_of_intended_air_yards, catch_percentage |
| **Rushing** | efficiency, RYOE, RYOE/att, % 8+ defenders in box, avg_time_to_los |
| **Passing** | CPOE, aggressiveness, avg_time_to_throw, avg_intended_air_yards |
| **Mask** | family mask pre-2016; position-appropriate submask (WR/TE get receiving, etc.) |
| **Tower** | `ngs` |
| **Cadence** | nightly ~3–5am ET in-season |
| **LOC** | ~220 |

---

### Track G — PFR advanced weekly (NEW — P1)

| Field | Detail |
|-------|--------|
| **Fetcher** | `nfl_data.pfr_adv(stat, year)` |
| **Source** | `pfr_advstats` (`advstats_week_{pass\|rec\|rush}_{year}.csv`) |
| **Coverage** | ~2018+ weekly files present |
| **Pass** | times_pressured_pct, bad_throw_pct, drop_pct, times_blitzed |
| **Rush** | yards_before_contact_avg, yards_after_contact_avg, broken_tackles |
| **Rec** | drops, drop_pct, broken_tackles, receiving_rat |
| **Join** | `pfr_player_id` ↔ `players.pfr_id` |
| **Tower** | `pfr_adv` |
| **Cadence** | daily 7AM UTC |
| **LOC** | ~180 |

---

### Track H — Red zone / opportunity from play-by-play (NEW — P0)

| Field | Detail |
|-------|--------|
| **Fetcher** | `pipeline/build_opportunity.py` (derive from pbp or pre-agg) |
| **Source** | nflverse `pbp` release **or** precomputed [ffopportunity](https://github.com/ffverse/ffopportunity) EP tables |
| **Coverage** | pbp ~1999+; ffopportunity models trained historically, releases on GitHub |
| **Features** | trailing RZ target share, RZ carry share, inside-5 carry share, expected fantasy points (EP), EP over expected |
| **Legal** | pbp nflverse; ffopportunity **CC-BY-SA** — cite in Methods |
| **Mask** | EP family mask if release missing for season |
| **Tower** | `opportunity` |
| **LOC** | ~300 (prefer ffopportunity CSV download over full pbp if lighter) |
| **Blocked-by** | decide EP download vs local xgboost rebuild (prefer download) |

---

### Track I — Defense vs position (expand — P0)

| Field | Detail |
|-------|--------|
| **Fetcher** | derived in `build_features` (exists) |
| **Expand** | separate DvP for pass/rush fantasy; rolling 4-week DvP; vs WR vs TE split; pace-adjusted points allowed |
| **Tower** | `defense` |

---

### Track J — Teammate / QB context (NEW — P1)

| Field | Detail |
|-------|--------|
| **Fetcher** | derived from weekly_stats + games QB ids |
| **Features** | team QB trailing PPR/EPA; RB committee Herfindahl; WR1 target share on team; pass rate over expected proxy from team implied |
| **Tower** | `context` |
| **LOC** | ~150 |

---

### Track K — Draft / combine / contracts (NEW — P1 rookies + aux)

| Field | Detail |
|-------|--------|
| **Fetcher** | `draft_picks.csv`, `combine.csv`, optional `contracts` |
| **Features** | pick, round, forty/vertical (combine), cap hit percentile (optional) |
| **Tower** | `pedigree` |
| **Use** | rookie model inputs + aux head (like hoops pedigree) |

---

### Track L — Market consensus (expand — P2)

| Field | Detail |
|-------|--------|
| **Fetcher** | `build_adp.py` (exists) + optional `ff_rankings` / FantasyPros via nflreadr when tagged |
| **Features** | ADP rank, ADP vs model residual (post-hoc calibration, not train target) |
| **Tower** | none in train (avoid circularity); optional calibration layer |

---

### Track M — FTN charting (P2 / optional)

| Field | Detail |
|-------|--------|
| **Source** | `ftn_charting` 2022+ |
| **Legal** | **CC-BY-SA 4.0 — must credit "FTN Data via nflverse"** |
| **Use** | play-action rate, motion, etc. aggregated to player-week |
| **Mask** | pre-2022 full mask |
| **Tower** | `ftn` |

---

### Track N — Participation / routes (P3 — deferred)

| Field | Detail |
|-------|--------|
| **Reality** | In-season participation **dead**; 2023+ FTN dump **after postseason only** |
| **Decision** | Do **not** block MTNN v2 on routes; document as future when feed returns |

---

## Recommended tower families (v2 matrix)

| Family | Example features | Coverage |
|--------|------------------|----------|
| `form` | trailing / EWMA PPR, yards, TDs, EPA | 2016+ |
| `usage` | targets, carries, snap%, shares, WOPR | 2016+ |
| `opportunity` | RZ shares, EP, EPOE | when EP available |
| `role` | depth rank, starter flag | depth charts |
| `availability` | injury status, games missed | ≤2024 |
| `meta` | age, exp, size, pos one-hot | always |
| `conditions` | weather, roof, rest, weekday, primetime | games.csv |
| `market` | implied pts, spread, total | games.csv |
| `defense` | DvP variants | derived |
| `ngs` | separation, RYOE, CPOE, … | 2016+ |
| `pfr_adv` | pressure, YBC/YAC contact | ~2018+ |
| `context` | QB quality, committee | derived |
| `pedigree` | draft/combine | career-constant |

---

## Free vs paid (explicit non-goals)

| Source | Status | Why not |
|--------|--------|---------|
| PFF grades | paid | cost + redistribution |
| FantasyData / Sleeper projections API | ToS / paid tiers | we build our own |
| ESPN proprietary projections | scrape risk | ADP only via public FFC/MFL |
| Spotrac full | ToS | OTC via nflverse contracts only |

---

## Update cadence (operator)

| Cadence | Feeds |
|---------|-------|
| Weekly Tue refresh (existing task) | stats, snaps, games, injuries (if any), depth, NGS, PFR, retrain/export |
| Daily in-season (optional later) | games Vegas, depth, injuries |
| Offseason | draft_picks, combine, ADP, full retrain |

---

## Attribution (Methods / README)

Must cite when features ship:

- [nflverse](https://nflverse.com) / nflreadr data schedule  
- Lee Sharpe `games.csv` (nfldata)  
- NFL Next Gen Stats (via nflverse)  
- Pro-Football-Reference advanced (via nflverse)  
- FTN Data via nflverse (CC-BY-SA) when Track M ships  
- ffopportunity / ffverse when EP ships  
- Fantasy Football Calculator / MyFantasyLeague for ADP  
