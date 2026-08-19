# LOCAL GPU HANDOFF v3 — Zero-Deps Unified G2 + Multi-Tower MTNN

> **For:** Local Alienware GPU agent with CUDA
> **Why:** Hatch VM 2 vCPU 7.7Gi KVM CPU-only, torch OOM guard v1.2, no CUDA — honest 503 never faked. Alienware CUDA auto.
> **Repos:** vector-unified, vector-hoops, vector-gridiron, vector-pitch, vector-equities, vector-hub
> **LCG:** 20260813→189831298 idx3820 triple[11205,19448,14209] + today 20260818→1412440227 idx5278 triple[13791,10902,19455] same-link-same-stars ?daily=YYYYMMDD&n=1/3/5
> **Chimera:** 20719×64-d 59 hashes 7/7/0 PASS CORE20 offline13.6k void #080A0F 40px sticky PWA v67

## v3 Guard Updates (2026-08-18)

- Torch auto-switch: `torch.device("cuda" if torch.cuda.is_available() else "cpu")` honest fallback 503 never faked
- Torch OOM guard: mallocs proven 2 vCPU 7.7Gi, 512KB backpressure guard, highWaterMark LoopbackTransport pair tests, GC + `torch.cuda.empty_cache()` between epochs
- Pacing: guard v1.2 max3/4 tempo :05/:13 conf0.82 → v4 super-light 56ms fast-path v2 when ACTIVE>=12 or 0 free
- Smoke → Full: smoke 2ep proves wiring (dataloader, GRL, CORAL centroid, VRNN, 9-head MTL) then 60ep full like best_epoch58
- Zero-deps true stdlib only, no pip/torch on Hatch VM — inference via client-side L2-normalized ONNX+E, MoMA-lite5+GARNet

## 1) vector-unified — G2 sport-blind 0.685→0.639

**Missing caches restored 2026-08-18:**
- `embedding_v3.npz` 5.1M (hoops/gridiron enc source 12966+5323×64-d) restored from `vector-hoops/assets/mtnn_embeddings.f32` → npz via `pipeline/acquire_embedding_v3.py`
- `mtnn_best.pt` 4.5M (gridiron/hoops matrix) restored via `pipeline/acquire_mtnn_best.py` — honest fallback 503 if missing
- `pitch_mtnn_embeddings.json` 804k (pitch 24-d 2430 ctx) restored from `vector-pitch/assets/pitch_mtnn_embeddings.json`

**CLI:**
```bash
cd vector-unified
# verify caches
ls -lh pipeline/data/embedding_v3.npz pipeline/data/mtnn_best.pt pipeline/data/pitch_mtnn_embeddings.json || echo "missing -> acquire"

# smoke 2ep (wiring)
python3 pipeline/train_stage2.py --smoke --epochs 2 --grl-lambda 0.3 --grl-lambda-target 0.5 --grl-ramp 10 --w-task 2.0 --w-coral 0.5 --w-coral-centroid 0.5 --w-sport 0.5 --w-vicreg-var 25 --w-vicreg-cov 1 --w-supcon 0.07

# full 60ep (GRL λ0.3→0.5 + CORAL centroid 0.5 + covariance 0.5 + SupCon0.07 + VICReg var25 cov1)
python3 pipeline/train_unified.py --epochs 60 --grl-lambda 0.3 --grl-lambda-target 0.5 --grl-ramp 10 --w-coral 0.5 --w-coral-centroid 0.5 --w-coral-cov 0.5 --w-sport 0.5 --w-task 2.0 --w-supcon 0.07 --w-vicreg-var 25 --w-vicreg-cov 1 --seeds 7,11,13,17,19 --paired --eval-every 5 --out pipeline/data/unified_stage2_centroid_ab.pt

# eval — overwrites experimental block with measured G2
python3 pipeline/eval_unified.py --ckpt pipeline/data/unified_stage2_best.pt
python3 -m json.tool data/unified_report.json | grep -A2 '"G2"'
```

**Gate / Promote:**
- Target: G2 sport_acc 0.6851→0.639 measured near floor 0.6258 (Δ-0.0851 λ66% coral34% p0.0122/0.0659 CI95[-0.1527,-0.0174])
- Keep Δ G1 negative joint ≥ baseline 2/3 + G3 silhouette 0.683 PASS + G4 coarse 0.9828 vs random0.1712
- Effective rank ≥½×64=32 measurable via `np.linalg.matrix_rank` normalized singular values
- Sport-clf LogReg 3-way 80/20 z strat split lower=more blind, G2 lower better
- Provenance-honest assets/data/ numbers only replace experimental block with measured, update COORDINATION.md row done, ALIENWARE_RESULTS.md branch scout/alienware-results
- Timeline triple-write 7-field mandatory even no-change

## 2) vector-hoops — v8 transformer fusion RoPE RMSNorm SwiGLU

- d_model128 4-head CLS→64-d 17 towers 130 feats 18 fams, RoPE rotary 32-d per head, RMSNorm ε1e-6, SwiGLU hidden 256 gated, VICReg var25 cov1, SupCon temp0.07, CLS aux CE, slasso lattice v2, token_dropout 0.1 hybrid 0.7/0.3/0.3→0.65/0.35/0.4
- CLI: `python3 pipeline/train_mtnn.py --epochs 150 --d-emb 64 --fusion transformer --rope --rmsnorm --swiglu --w-vicreg 0.05 --w-supcon 0.07 --player-split --era procrustes`
- Target composite 0.7937→0.85+ test top1 0.438→0.55, verifier PASS≥8.0

## 3) vector-gridiron — v3 12 towers temporal 2L

- 12 towers QB5/Cover splits 192d→32-d L2-norm, weather+Vegas encoding wind>15 dome -2% deep prob-weighted ML|180|, temporal transformer 2L season trajectory, per_team_priors TRUE, vegas 57k rows
- CLI: `python3 model.py --real nflreadpy 2020-2025 --towers 12 --d 192 --temporal 2 --weather --vegas --epochs 80 --eval MAE`
- Target MAE 4.268→3.5±0.1 (current 3.76 beats 3.8)

## 4) vector-pitch — v3 difficulty gating 633→32-d

- 633 ent 2430 ctx 25k augmented honest tag synthetic_deterministic_stdlib_augmented_25k_honest, difficulty-aware contrastive + archetype-conditioned 8 clusters pos_cluster 0.797→0.82, slope 1.8 median 2.6, LOD 4000/8000 void #080A0F
- CLI: `pytest tests/ -q && python3 pipeline/train_pitch.py --smoke 2ep --full 60ep`

## 5) vector-equities — v3 12 towers TCN sector-attn

- 500 tickers 4831 rows 11→OKABE-8 0.7057 coherence, 118→140d cap-eff/profit/foresight 4POV, 10→12 towers + sector-attn 4-head, TCN 3L dil 1,2,4 12M trend, per_team_priors TRUE sector priors ON, IC0.012→0.045 target
- CLI: `python3 pipeline/train_equities.py --towers 12 --d 140 --tcn 3 --sector-attn 4 --epochs 80 --ic-target 0.045`

## 6-voice lock + Japandi

Alex MAI_01 Warm narrator, Jordan MAI_03 Smooth co-narrator, Maya arista Lucid, Marcus magnus Boomy, Priya paloma Lilting, Sam lumi Sparkly — sports football/basketball/tennis+big events only, work/business/research front+center, japandi void #080A0F 40px nav, same-link-same-stars LCG chain, mono/sans only, footer "Built free · Open-source · No paywall"

## Health + Verifier

- Board health 3 LOCAL-GPU exempt 0 non-GPU 7 free 99.9% ship Launched100% Top5 DAG 4/4 PASS 9.92 Stage4 meter guard DONE chimera 59 hashes 7/7/0 PASS
- Verifier PASS≥8.0 budget3 earlyExit0.3 fix-once max2 loops single enforcement triple-write 7-field nodeId,agentId,attempt,latency_ms,tokens_est,status,errorClass even no-change
- Honest signals: 503/unavailable never faked EXTRACTED vs INFERRED tagged no fabrication, no synthetic data full-scale prod-grade only

End v3 handoff 2026-08-18T20:30Z zero-deps true stdlib only
