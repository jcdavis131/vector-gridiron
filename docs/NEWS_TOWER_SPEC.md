# News Tower — Added MTNN Tower for Public Media Aggregation

**Goal:** Add a 4th tower to each domain's MTNN that ingests real public news/RSS, links to entities, and produces recency-weighted features — zero-deps stdlib only, honest 503, no synthetic.

## Why
Current towers use box-score / cap / form features. News captures narrative shocks: injuries, trades, manager changes, earnings surprises — not in stats yet but predictive. Added as auxiliary tower, not replacing core, to preserve construct validity.

## Architecture (GraphBFF dual-stream compatible)
- Existing: 3 towers 16→32 ResidualTower LN GELU×2 gated → TCA 4 heads + TAA 128-d k=8 → fusion 0.7/0.3 L2 64→32
- New: **4 towers** — 4th is News Tower 16→32 same ResidualTower, same TCA eligibility, but edges = co-mention + same-team news burst
- Fusion: 0.60 * tca + 0.25 * taa + 0.15 * news → L2 unit sphere (down-weights news to avoid narrative overfit)
- Provenance: 7/7/0 → 14/14 hashes with news source types

## Per-Domain Sources (real RSS, stdlib xml.etree + urllib, 10s timeout, honest 503)

### Hoops — vector-hoops
- NBA.com news RSS, ESPN NBA, CBS Sports NBA, Yahoo Sports NBA
- Features per player per day: mention_count_7d, mention_count_30d, sentiment_7d (-1..1 via word-list), injury_flag, trade_rumor_flag, starter_flag, rest_flag, recency_decay exp(-days/7)
- Entity link: casefold name match vs current_rosters.json + chemistry.json (Gary Payton → GP II safe)
- Output: assets/news/news_features.json — dict[player_id] -> 16-d list float [-1,1]

### Gridiron — vector-gridiron
- NFL.com, ESPN NFL, CBS NFL, NFL Trade Rumors RSS
- Features: same + QB controversy, depth-chart change, weather narrative
- Link vs nflreadpy 1018 players

### Pitch — vector-pitch
- BBC Sport Football, The Guardian Football, ESPN FC, Sky Sports
- Features: transfer_rumor, injury, manager_change, form streak narrative, 9 leagues aware
- Link vs 633 ent StatsBomb

### Equities — vector-equities
- SEC 8-K RSS (sec.gov/cgi-bin/browse-edgar), Yahoo Finance RSS, Reuters Business, PR Newswire earnings
- Features: earnings_surprise_sentiment, guidance_change, insider_flag, macro_sector_news, recency
- Link vs 500 tickers via ticker + company name
- No price leakage: only news up to T-1, no future returns in features

### Unified / Hub — vector-unified + vector-hub
- Meta-aggregator: union of above + Schools NCES news (district news via Google News RSS education)
- Features: cross-domain narrative bridge (player→school→ticker brand halo)
- Used for chimera 24799→45279 auxiliary only, weight 0.08

## Ingest / Clean / Featurize (stdlib only)
1. **ingest**: `news_ingest.py --domain hoops --out assets/news/raw_YYYYMMDD.json` — urllib.request urlopen RSS, xml.etree parse <item><title><link><pubDate>, store url/title/desc/date/source, 10s timeout, try/except -> honest 503 empty list with reason logged
2. **clean**: `news_clean.py --in raw --out cleaned` — dedup by title+url casefold, strip HTML via html.parser, filter English, drop >30d old, entity linking via Aho-Corasick style simple scan (no regex heavy), preserve DOB Jr/Sr safe
3. **featurize**: `news_featurize.py --in cleaned --rosters current_rosters.json --out news_features.json` — 16-d per entity: [cnt7 log1p/5, cnt30 log1p/10, sent7, sent30, inj, trade, starter, rest, transfer, manager, earnings, guidance, recency_exp, burst_3d z, league_sent, sector_sent] — all [-1,1] or [0,1] normalized, L2 not applied here (tower does LN)
4. **provenance**: `news_provenance.json` — 7/7/0 style: source_url, fetch_time, n_raw, n_cleaned, n_linked, n_entities_with_features, hash sha256 of raw file list

## Batching / Edges
- TCA edges: same-archetype / same-league / same-team / same-real-vs-aug now + same-news-burst (co-mention in same 24h window) 40% dominant remains
- TAA edges: k=8 fixed-degree most recent 8 seasonal + news recency k=4 most recent news burst neighbors
- Batch: KL 64 clusters league+formation + RR 32/type + RR 16/type schools aux + RR 8/type news burst = 288+64 supervision edges/batch

## Eval Uplift (no leakage)
- Ablation: composite w/ news vs w/o news, report Δ composite, Δ pos_cluster, Δ IC (equities), Δ MAE (gridiron)
- Temporal split: train ≤2024-12-31, valid 2025-01→2025-06, test 2025-07→now, news features only up to feature_date-1
- Glass-box: SHAP top drivers should show news weight ≤0.15, not dominant
- Guard: if news source 503 >3 days, fallback to zero-vector honest, never fake sentiment

## Ship / UI
- News tab in pudding-map sites: paper #FAFAF8 shell #FEFCF9 inner void #080A0F 40px sticky nav z40, single-select map clear prev, OKABE-8
- Feed: title + source + time ago + entity chips, filter by player/ticker, 300 rows virtual window 30, no infinite scroll
- Provenance footer: n sources, n articles today, 7/7/0→14/14 hashes, LCG same-link-same-stars ?daily=YYYYMMDD&n=1/3/5
- PWA v67 offline13k: cache last 7d cleaned json, CORE20

## Cron / LCG
- Daily 06:00 CT ingest + featurize, same LCG chain 20260813→189831298 idx3820 triple[11205,19448,14209] + 20260818→1412440227 triple[13791,10902,19455]
- Zero-deps true, stdlib only, torch optional Alienware honest 503

## Next
- Scaffold hoops as reference, replicate to other 4, update candidate.json n_towers 3→4, docs arch MD, then merge branches PASS 8.8→9.0+ expected
