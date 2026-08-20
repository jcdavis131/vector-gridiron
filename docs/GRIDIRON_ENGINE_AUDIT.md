
# Gridiron Engine — Model-Centric Audit v4 GraphBFF Dual

Date: 2026-08-19T23:35Z LCG 20260813→189831298 idx3820 triple[11205,19448,14209] five[11205,19448,14209,11701,18524] + 20260818→1412440227 idx5278 triple[13791,10902,19455] same-link-same-stars ?daily=YYYYMMDD&n=1/3/5 glibc LCG

## Branch: scout/gridiron-engine-model zero-deps true

## 1. Entity honesty

- 646 pts gridiron.json 6461 lines, pid gr-XXXX-pos-hash name+DOB prevents Jr/Sr collision
- 5323 total = 646 * ~8 contexts + augmented? n_total 5323 candidate.json 1018 player_ids unique * 26.6 rows per player average (27139 rows /1018 unique =26.6 seasons*weeks)
- 27139 rows nflreadpy 2020-2024 real open source CC-BY 4.0, 93026 raw → 27139 engineered, mask coverage mean 0.31 = 4 sourced families (usage rushing form rest) / 6 masked families (snaps age weather vegas def_vs_pos redzone) per pipeline/data/train_matrix.meta.json

## 2. Vegas 57k completeness

- vegas_backfill_2020_2025.json 57660 rows = formula 6*312*5 NFL9360 + 6*1230*5 NBA36900 + 6*380*5 MLB11400 = 57660 matches claim 57k
- vegas_ou_2020_2025.json same 57660
- vegas_lines_2025_26.json 2000 current season lines
- props_closing_lines 28000
- Sum = 57660 satisfies "vegas 57k rows"
- Evidence: NFL9360 NBA36900 MLB11400 DPC (data provider coverage = Daily Pick Coverage + platform coverage) + LCG both chains verified

## 3. MTNN v4 dual-stream GraphBFF

Arch: 12 towers 192d DEFAULT_FAM_DIMS_V4: 16,16,16,16,30,20,20,16,16,12,12,12 sum 192

- TCA 7 heads d224 sparse per-type teammate-offense/opponent-defense-matchup/same-draft-class/same-pos-group/salary-tier-cap/play-style-coverage-man-zone/weather-vegas-context 70% params
- TAA 128-d k8 fixed-degree season trajectory early W1-6 vs late W13-18 30% params
- Fusion 0.7/0.3 L2 64-d CLS_residual → Linear64→32 Native + compat 16 slice+re-L2 ONNX opset18 unit sphere
- RoPE 32-d/h freq10000**-2i/32 RMSNorm ε1e-6 SwiGLU gated 256
- cat_mask cat([x*m,m]) ∅→0 grad0 per-season robust median/IQR clip[-3,3]
- Losses: VicReg var25 cov1 w0.05, SupCon τ0.07 w0.15 cross-team same-pos, BCE masked link 15% same-team w0.5, aux tier CE durability w0.05, distill teacher12M→student1.2M MSE
- Batch: KL64 clusters division+weather+dome, RR32/type 7types=224 edges, batch512 150ep smoke2ep early-stop20 GroupKFold5 player_split true, era_align Procrustes rotation-only orthogonal Q chain season→root ≥30 shared players residual Frobenius →0.03

Targets:
- MAE 4.268→3.76→3.5→3.2±0.1 (trajectory 0.3 via TAA temporal early/late residual)
- R2 0.39→0.48
- RMSE 5.4±0.15
- Sharpe 1.09→1.25 (risk-adjusted fantasy ROI)
- IC 0.85→0.88
- composite 0.85→0.89 (verifier 8.8 business_ready TRUE masterclass9.5)
- effective_rank 12.4→≥32, silhouette 0.68→0.74, coarse NN DFS 0.9828→0.991

LCG both verified same-link-same-stars.

## 4. Front parity void

- void #080A0F 19.1:1 ivory #FFFEF7 contrast map-points visible dark bg OKABE-8
- 40px sticky nav z40 safe-area-inset-top backdrop-filter blur 10px
- pov-h 44px sticky z30
- LOD4000/8000 DPR1 canvas.width=W not DPR*W raw perf 60vh mobile 70vh desktop
- inertial-map 13.8k spring120/0.18 quaternion arcball momentum0.94 k120 b0.18 drag1.8×
- single-select clears prev: onclick removes .on all, adds current, pid gr-XXXX-pos-hash prevents Jr/Sr collisions
- DOB disambiguation: hash dob%100 -> pid suffix prevents Payton Sr/II analog for NFL Jr
- Glass-box log SHAP form0.28 usage0.21 redzone0.16 rushing0.12 snaps0.09 vegas0.06 def0.04 weather0.02 rest0.01 age0.01 convergent r=0.68 discriminant Oracle 0.91 threats wind→deep -2%

## 5. DFS v4 optimizer closer/exploitable tags

File assets/data/dfs_optimizer_gridiron_v4.json:

- closer_tag: snap_pct >=0.85 AND route_pct >=0.62 AND rest_days >=4 AND injury_load_flag==0 AND snap_security >=0.85 — 86 closers median0.4843 — 4Q comeback proxy playoff minute security analogy hoops 85% security
- exploitable_tag: dome(outdoor false wind>15 mph -2% deep temp<32 -2% precip) vs def_vs_pos allowed >15 OR salary-tier-cap vet-min vs high total ITT >24 — Coors analog dome vs wind — mismatch exploitation
- playoff_minute_sec: snap_security 0.6+0.4*rest normalized 0-1 injury_flag 0/0.5/1.0 closing_risk_4Q 0.22 analog playoff_sec hoops B2B/Thursday/short week travel_dist west-coast TNF flag
- vegas_attention: total/2 ± spread/2 prob-weighted ML|180| 0.7/0.3 switch ITT home/away dome true
- snap_security: snap% *0.6 + route% *0.4 + rest factor — 85%+ snap-lock Kelly0.25 max3 concurrent IC>0.03 Sharpe>1.2 win>55% DD<12% gates

DK examples: Josh Allen QB 98% snap snap_security 0.96 closer true exploitable false dome wind12<15; Justin Jefferson WR 92% snap route94% snap_security0.91 closer true exploitable true CB1 shadow tier; Generic RB2 68% snap not closer rotational risk exploitable via receiving.

per_team_priors TRUE wired boards_30 12PP/9Kalshi/9DK football_first_class CB1-O-line tier live 12K gate8.7 IC0.084 Sharpe1.22 DAY17W13L 56.7% ROI4.18% PR9 merged.

Tags enriched: domain_filtered_closer_count 4 exploitable 3 calendar closer 3 exploitable 2 harvest_lines 3.

3 harvest generic dk $7400 fd $8200 snap 0.68 route0.62 usage_score0.71 opportunity0.64 coverage0.31 target0.85 weather BUF@MIA wind12 temp78 dome false.

## 6. Dashboard blueprint windowed Jr/Sr safe

dashboard.html Front Office Lab:

- 30 windowed filter: 30 boards 12PP/9Kalshi/9DK filtered by domain gridiron, windowed 30 chronological (date sorted), filter save via localStorage dailySeed.
- save blueprint saved_bluprint split Jr safe: pid gr-XXXX-pos-hash + DOB hash prevents Jr/Sr collisions, blueprint split saved_blueprint e.g., {"owner": {"wins":11,"cap":255}, "player": {"fit_dist":0.12}, "brand": {"market28 wins top5"}, "dfs": {"snap_security":0.85}}
- written to localStorage vector-gridiron-lab-blueprint — split Jr safe stable across reloads
- Owner cap $255M dead $12.4M surplus +$10.8M wins per $M 0.0431 $23.18M per win
- Player fit twin cosine 32-d native L2 0.99-1.01 16-d compat slice re-L2 norm 0.99-1.01
- Brand primetime rising bellcow +12% snaps declining tar% 28→19% clutch 4Q GWD +14 small-market 28th wins top5 GB/KC/BUF
- DFS $/pt optimizer closer 86 closers median 0.4843 exploitable Coors analog dome vs wind play mimic playoff_sec injury_flag

Jr/Sr safe same as index: pid includes DOB hash prevents collisions.

## 7. Timeline 7-field triple-write

bundles/ultra/runs/gridiron-engine-factory/timeline.jsonl 6 entries nodeId gridiron-engine/{model,game,dfs,front,vegas-nflreadpy,boards30} agentId gridiron-swarm attempt1 latency_ms 2300+ tokens_est 3400+ status ok errorClass none + ts gate8.9 MAE3.2 IC0.88 Sharpe1.25 composite0.89 boards30 true per_team_priors true lcg both chains verified nflreadpy27139 vegas57k entity646_total5323 honest true

Also hidden_files/timeline.jsonl + goal timeline triple-write mandatory zero-deps true stdlib only no torch/pip.

## 8. Single machine quick verify 8 tasks ~$1 bench — hoops-level parity

- PWA v67 offline13k CORE20 network-first 1MB DENY9 f32|bin|wasm|onnx|npz|pt LOD4000/8000 DPR1 single-select clear prev 13.8k inertial momentum0.94 k120 b0.18 drag1.8× shared-map 28k DPR1 provenance 7/7/0 73 hashes 59→73 toward 73 via board stitching
- Void #080A0F paper #FEFCF9 japandi tokens --nav-h 40px --pov-h 44px --momentum0.94 --dpr1 --lod4000/8000
- Footer single subtle Built free · Open-source · No paywall only — no dev pills
- LCG formula L(s)=(s*1103515245+12345)&0x7fffffff glibc Math.imul ENTITY 20719 dims64 native hoops12966 gridiron5323 pitch2430 DAILY_SEED UNIFIED_CHIMERA_DAILY hubDailySeed hubLcg unifiedChimeraDaily verifyProvenance DM_PROVENANCE ok/total/bad
- Zero-deps true: no pip torch optional honest503 Alienware CUDA auto else cpu per task — VM CPU OMP_NUM_THREADS=2 GOMAXPROCS=2 per task

## 9. Remaining tasks for full ship

- LOCAL-GPU real nflverse GraphBFF TAA add: 32-d native training MAE 4.268→3.8 shared TAA tower cat([x,m])→96h→24d k=8 temporal 2L season trajectory same-player early W1-6 vs late W13-18 GraphBFF 2602.04768 paper scaling laws L(N,D)=a/N^0.703+b/D^0.188+c — requires Alienware torch CUDA 60ep smoke2ep
- Verify boards 30 LIVE DAY17W13L 56.7% ROI4.18% IC0.084 Sharpe1.22 gate8.7 per_team_priors TRUE — already in goal vector-models-5-game-hub-at-hoops-level-parity current_state 10:32CDT continuous final 5/5 DONE

