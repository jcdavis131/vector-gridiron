# SPEC — League context on every page (Lookback + Trades + owners)

> **Status:** Approved via `/auto-mode` feature ask · 2026-07-10  
> **Repo:** vector-gridiron · baseline `e63a4bb` clean

## Objective

When a league is connected (ESPN / Sleeper / demo), **every cockpit view** uses that league — especially **Lookback** (real draft history + owner evaluation) and **Trades** (roster + historical owner context). Managers can judge owners across seasons, not only the current roster snapshot.

## Current state (live evidence)

| Area | Status |
|------|--------|
| My Team / Start-Sit / Waivers / Draft board | Uses `STATE.league` |
| Lookback live fetch + VOR grading | Implemented (`refreshLookbackFromLeague`) |
| Lookback owner **career** rollup | Missing |
| Trades | Current-roster heuristic only — no lookback grades |
| Season picker live vs mock labeling | Weak |
| Highlight “my” lookback team | Name-only (fragile) |

## Acceptance

1. After connect (or session restore), Lookback shows real graded seasons when drafts exist; status line is honest when empty/loading/err.
2. Lookback includes an **Owners** board: multi-season draft grade, titles, PF, seasons graded — keyed to current league teams when names/ids match.
3. Trades shows partner **historical draft grade / titles** when lookback live is ready; still works without history.
4. Season picker marks `● live` vs mock seasons when a league is connected.
5. `afterLeagueLoaded` + lookback completion both refresh Trades / Lookback.
6. `node pipeline/verify_logic.mjs` still passes; add a small lookback-owner unit check in `lookback_live_verify.mjs` or a new `league_context_verify.mjs`.

## Non-goals

- Full week-by-week matchup narratives from ESPN (already noted as future).
- Writing cookies to a server.
- Changing MTNN / projection math.

## Commands

```powershell
cd c:\Users\jcdav\vector-gridiron
node pipeline/verify_logic.mjs
node pipeline/league_context_verify.mjs
```
