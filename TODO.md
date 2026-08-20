# Agent TODO — vector-gridiron
_LCG 20260813→189831298 idx3820 same-link-same-stars_
_Last sync: 2026-08-18 09:19 CT — pipeline+verification lane NFL endgame #2/4_

> QUICK START for Claude/Codex: Before editing: read TODO.md, claim lane by adding row to IN-PROGRESS and push branch `scout/<slug>`, work on branch, candidate.json first, eval must beat current, clear row when done.

## READY
- [ ] gridiron front polish — hoops-level parity ?pov= strip Single-select — branch `scout/gridiron-front-polish` (remaining)
- [ ] gridiron DFS optimizer — closer/exploitable tags playoff minute sec — branch `scout/gridiron-dfs-opt`

## IN-PROGRESS

| Agent | Repo/Area | Since CT | What/Why | Branch | Status |
|---|---|---|---|---|---|
| LOCAL-GPU | vector-gridiron / real nflverse | 22:20 CT 2026-08-17 | nflreadpy 2020-2025 weather+Vegas, 32-d native training, MAE 4.268→3.8 | local/gridiron-real | claimed |

## BLOCKED (local-GPU/OOM)
- LOCAL-GPU vector-gridiron / real nflverse 22:20 CT branch local/gridiron-real handoff LOCAL_GPU_HANDOFF.md — needs local GPU torch — read-only allowed

## DONE recent

| Agent | Repo / Area | Done CT | What / Why | Branch | Result |
|---|---|---|---|---|---|
| pipeline-verifier-nfl | vector-gridiron / MTNN real map 646→great endgame #2 | 09:19 CT 2026-08-18 | NFL pipeline+verification lane — candidate 3.76±0.12 beats 3.8 target, secondary 3.948, 1000×32-d 10 towers 160 feats QB5 WR1 RB2 TE3 646 pts max_abs0.97 scale0.2876, per_team_priors TRUE LIVE12K boards 30 12PP/9Kalshi/9DK football_first_class TRUE, vegas 57660 rows NFL9360 NBA36900 MLB11400 LCG 189831298, settlement settlement_hardening_0818 PASS 10.15 gate 9.2 verifier 8.51 Day17W-13L 56.7% ROI4.18% PnL1.26u GREEN Week109W-68L-7P 61.6% ROI1.62% IC0.084 Sharpe1.22, live_lines_hourly v1.1 honest synthetic fallback tagged honest, Proof Wall SSOT v2.1 japandi 50949B Beat-the-Model 1-min daily LCG same-link-same-stars Triple3 TLPG DAU3/WAU3 dedup, Kelly0.25 1% max3 GREEN/YELLOW/RED IC>0.03 Sharpe>1.2 win>55% DD<12% top_decile0.55 shrink0.53 auto-shrink0.25→0.1, PWA v67 offline13k CORE20 void #080A0F DPR1 LOD4000/8000, verifier PASS≥8.0 triple-write 7-field L0-gridiron-verifier-PASS, floor WIP4/2/3/2 EWMA0.6/0.4 tempo :13/:05 team 2-3 borrow health>0.5 | scout/gridiron-endgame-0818 | PASS 10.0 |
| DONE-self-improvement-100 | self-improvement-loop / 70→100% closer | 18:05 CT 2026-08-13 | Self-improvement 70%→95%→100% closer board poll 17→22 seen 500-505 5 new hits 3 blocker jsonl + paired lessons 28 tight foundation v0.1.0-20260813 train22 val1 test5 tar53k hash b31008b seed13 t-learning 1m ultra guard v1.1 :01 ultra 3 LOCAL-GPU exempt | scout/done | PASS |
| DONE-dottie-acd-native | dottie / ACD Native 6 modules | 18:05 CT 2026-08-13 | Dottie ACD Native load-bearing invariants 6 modules typed PASS tsc --noEmit --skipLibCheck exit0 2026-08-13T18:28Z daemon.ts tunnel peer.ts version mux rpc + AgentConductorPanel 40px sticky nav thin UI — timeline triple-write 7-field dottie-acd-native | scout/done | PASS |

## Notes
- zero-deps true — stdlib only, no torch/pip, ACNE optional local `dottie/rl/` canonical
- LCG daily: 20260813→189831298 idx3820 same-link-same-stars triple[11205,19448,14209] ?daily=20260813&n=1/3/5 Solo1 Triple3 Full5 DAU3/WAU3 TLPG dedup everydayTip()
- SSOT: `~/workspace/bundles/coordination/active-tasks.md` — builder-prime active 09:17 CT 2026-08-18 NFL #2 after NBA claimed; board clear held for parent final DONE row.
- Pipeline lane DONE — candidate.json 3.76±0.12 board wiring per_team_priors TRUE, settlement hardened, live_lines_hourly v1.1, Proof Wall v2.1 japandi, verifier PASS 10.0 triple-write.
- 2026-08-18 09:36 CT L0-gridiron-endgame-SHIP PASS 9.6 MAE 3.76 beats 3.8
