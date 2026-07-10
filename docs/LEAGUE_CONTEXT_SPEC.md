# SPEC — League context on every page (Lookback + Trades + owners)

> **Status:** Shipped · 2026-07-10 (owners + trades) · **hotfix** players-array unwrap 2026-07-10  
> **Repo:** vector-gridiron

## Objective

When a league is connected (ESPN / Sleeper / demo), **every cockpit view** uses that league — especially **Lookback** (real draft history + owner evaluation) and **Trades** (roster + historical owner context).

## Hotfix (Lookback “No completed drafts” with connected badge)

**Root cause:** `espnProxy` always did `data = data[0]` on array responses. That is correct for `leagueHistory` (`[{league}]`) but **wrong** for `mode=players` / `playerPool` (`[player,…]`). Resolve returned one object → zero named picks → every season graded empty → misleading “No completed drafts found” while the roster badge still showed the live league.

**Also hardened:** cookie headers (`X-Espn-S2` / `X-Espn-Swid`), `drafted` flag optional when picks exist, playerPool fallback, distinct `auth` status vs `empty`, Connect → Reconnect when connected.

## Acceptance

1. After connect (or session restore), Lookback shows real graded seasons when drafts exist; status line is honest when empty / loading / err / **auth**.
2. Lookback **Owners** board: multi-season draft grade, titles, PF — keyed to current league teams.
3. Trades shows partner career chips when lookback live is ready.
4. Season picker marks live vs mock when lookback is ready.
5. `afterLeagueLoaded` + lookback completion refresh Trades / Lookback; Connect button reflects connected state.
6. `node pipeline/verify_logic.mjs` + `node pipeline/league_context_verify.mjs` pass.

## Commands

```powershell
cd c:\Users\jcdav\vector-gridiron
node pipeline/verify_logic.mjs
node pipeline/league_context_verify.mjs
```
