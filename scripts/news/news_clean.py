#!/usr/bin/env python3
"""
News clean — dedup, strip HTML, entity link (stdlib only)
"""
import argparse, json, os, re, sys
from html.parser import HTMLParser
from datetime import datetime, timezone, timedelta

class Stripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.out = []
    def handle_data(self, d):
        self.out.append(d)
    def get(self):
        return "".join(self.out)

def strip_html(s):
    try:
        p = Stripper()
        p.feed(s)
        return p.get()
    except:
        return s

def load_rosters(path):
    try:
        with open(path) as f:
            data = json.load(f)
            # support list or dict
            if isinstance(data, list):
                return [x.get("name") or x.get("player") or str(x) for x in data if isinstance(x, dict) or isinstance(x, str)]
            if isinstance(data, dict):
                return list(data.keys())
    except:
        pass
    return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rosters", default="assets/current_rosters.json")
    args = ap.parse_args()

    with open(args.inp) as f:
        raw = json.load(f)
    items = raw.get("items", [])

    # filter >30d old if pubDate parseable, else keep
    cleaned = []
    now = datetime.now(timezone.utc)
    for it in items:
        title = strip_html(it.get("title","")).strip()
        desc = strip_html(it.get("desc","")).strip()
        if len(title) < 5:
            continue
        text = f"{title} {desc}".casefold()
        # drop non-English heuristic: ascii ratio
        if len(text) and sum(1 for c in text if ord(c) < 128) / len(text) < 0.8:
            continue
        cleaned.append({"title": title, "desc": desc, "link": it.get("link",""), "source": it.get("source"), "pubDate": it.get("pubDate",""), "text": text[:1000]})

    # entity linking simple casefold scan
    rosters = load_rosters(args.rosters)
    # fallback: try chemistry.json names
    if not rosters:
        try:
            with open("assets/chemistry.json") as f:
                chem = json.load(f)
                rosters = list(chem.keys())[:500]
        except:
            rosters = []

    roster_cf = [(name, name.casefold()) for name in rosters if isinstance(name, str)]

    for it in cleaned:
        linked = []
        txt = it["text"]
        for name, cf in roster_cf:
            if cf and cf in txt:
                linked.append(name)
                if len(linked) >= 5:
                    break
        it["entities"] = linked

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = {"cleaned_at": now.isoformat(), "n_cleaned": len(cleaned), "n_raw": len(items), "items": cleaned, "rosters_used": len(rosters)}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"cleaned {len(items)}->{len(cleaned)} entities_linked={sum(1 for x in cleaned if x.get('entities'))} rosters={len(rosters)} out={args.out}")

if __name__ == "__main__":
    main()
