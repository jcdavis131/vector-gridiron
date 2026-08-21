#!/usr/bin/env python3
"""
News featurize — 16-d per entity, stdlib only, no torch
Features: cnt7, cnt30, sent7, sent30, inj, trade, starter, rest, transfer, manager, earnings, guidance, recency_exp, burst_3d z, league_sent, sector_sent
All normalized [-1,1] or [0,1], honest 0 if no news
"""
import argparse, json, os, math, sys
from datetime import datetime, timezone, timedelta
from collections import defaultdict, Counter

POS_WORDS = set("win won victory champion great excellent hot streak clutch elite breakout surge upbeat".split())
NEG_WORDS = set("loss lost injury injured trade rumor rest bench slump down cold out questionable doubtful".split())
INJ_WORDS = set("injury injured out questionable doubtful surgery concussion".split())
TRADE_WORDS = set("trade rumor rumors deal deadline".split())
STARTER_WORDS = set("starter starting lineup starting five".split())
REST_WORDS = set("rest resting load management dnp".split())

def sentiment(text):
    w = text.casefold().split()
    pos = sum(1 for x in w if x in POS_WORDS)
    neg = sum(1 for x in w if x in NEG_WORDS)
    tot = pos+neg+1
    return (pos-neg)/tot  # -1..1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.inp) as f:
        data = json.load(f)
    items = data.get("items", [])
    now = datetime.now(timezone.utc)

    # aggregate per entity
    per_entity = defaultdict(list)
    for it in items:
        for ent in it.get("entities", []):
            per_entity[ent].append(it)

    features = {}
    for ent, lst in per_entity.items():
        cnt7 = sum(1 for x in lst)  # simplified: assume all within 30d, real would parse pubDate
        cnt30 = len(lst)
        # sentiment avg
        sents = [sentiment(x.get("text","")) for x in lst]
        sent7 = sum(sents)/len(sents) if sents else 0.0
        sent30 = sent7
        txt_all = " ".join(x.get("text","") for x in lst).casefold()
        inj = 1.0 if any(w in txt_all for w in INJ_WORDS) else 0.0
        trade = 1.0 if any(w in txt_all for w in TRADE_WORDS) else 0.0
        starter = 1.0 if any(w in txt_all for w in STARTER_WORDS) else 0.0
        rest = 1.0 if any(w in txt_all for w in REST_WORDS) else 0.0
        # placeholders for pitch/equities reuse
        transfer = trade
        manager = 0.0
        earnings = 0.0
        guidance = 0.0
        # recency exp(-days/7) — assume 0 days for now
        recency = 1.0
        # burst 3d z-score: cnt7 vs cnt30 mean
        burst = (cnt7 - cnt30/4.3) / (math.sqrt(cnt30)+1) if cnt30 else 0.0
        burst = max(-1,min(1,burst/3))
        league_sent = sent7
        sector_sent = 0.0

        # normalize cnt log1p
        cnt7_n = math.log1p(cnt7)/5.0
        cnt30_n = math.log1p(cnt30)/10.0

        vec = [cnt7_n, cnt30_n, sent7, sent30, inj, trade, starter, rest, transfer, manager, earnings, guidance, recency, burst, league_sent, sector_sent]
        # clip to [-1,1]
        vec = [max(-1,min(1,float(x))) for x in vec]
        features[ent] = vec

    # also include zero-vectors for entities with no news? Caller merges with roster list
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = {"featurized_at": now.isoformat(), "n_entities": len(features), "n_items": len(items), "features": features, "dim": 16, "fusion": "0.60*tca+0.25*taa+0.15*news L2", "lcg": "20260813->189831298 idx3820 same-link-same-stars"}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"featurized entities={len(features)} items={len(items)} out={args.out} dim=16")

if __name__ == "__main__":
    main()
