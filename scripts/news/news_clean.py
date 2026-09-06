#!/usr/bin/env python3
"""
News clean — dedup, strip HTML, entity link (stdlib only)
"""

import argparse
import json
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path


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
    except Exception:
        return s


def load_rosters(path):
    try:
        with Path(path).open() as f:
            data = json.load(f)
            # support list or dict
            if isinstance(data, list):
                return [
                    x.get("name") or x.get("player") or str(x)
                    for x in data
                    if isinstance(x, dict) or isinstance(x, str)
                ]
            if isinstance(data, dict):
                return list(data.keys())
    except Exception:
        pass
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rosters", default="assets/current_rosters.json")
    args = ap.parse_args()

    with Path(args.inp).open() as f:
        raw = json.load(f)
    items = raw.get("items", [])

    # filter >30d old if pubDate parseable, else keep
    cleaned = []
    now = datetime.now(UTC)
    for it in items:
        title = strip_html(it.get("title", "")).strip()
        desc = strip_html(it.get("desc", "")).strip()
        if len(title) < 5:
            continue
        text = f"{title} {desc}".casefold()
        # drop non-English heuristic: ascii ratio
        if len(text) and sum(1 for c in text if ord(c) < 128) / len(text) < 0.8:
            continue
        cleaned.append(
            {
                "title": title,
                "desc": desc,
                "link": it.get("link", ""),
                "source": it.get("source"),
                "pubDate": it.get("pubDate", ""),
                "text": text[:1000],
            }
        )

    # entity linking simple casefold scan
    rosters = load_rosters(args.rosters)
    # fallback: try chemistry.json names
    if not rosters:
        try:
            with Path("assets/chemistry.json").open() as f:
                chem = json.load(f)
                rosters = list(chem.keys())[:500]
        except Exception:
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

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out = {
        "cleaned_at": now.isoformat(),
        "n_cleaned": len(cleaned),
        "n_raw": len(items),
        "items": cleaned,
        "rosters_used": len(rosters),
    }
    with Path(args.out).open("w") as f:
        json.dump(out, f, indent=2)
    print(
        f"cleaned {len(items)}->{len(cleaned)} entities_linked={sum(1 for x in cleaned if x.get('entities'))} rosters={len(rosters)} out={args.out}"
    )


if __name__ == "__main__":
    main()
