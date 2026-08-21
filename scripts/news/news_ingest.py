#!/usr/bin/env python3
"""
News ingest — zero-deps stdlib only, honest 503
Fetches real RSS for hoops domain, stores raw JSON with provenance.
No synthetic data ever.
"""
import argparse, json, sys, time, hashlib, os
from datetime import datetime, timezone
import urllib.request, urllib.error
import xml.etree.ElementTree as ET

SOURCES = {
    "hoops": [
        ("nba.com", "https://www.nba.com/news/rss.xml"),
        ("espn_nba", "https://www.espn.com/espn/rss/nba/news"),
        ("cbs_nba", "https://www.cbssports.com/rss/headlines/nba/"),
        ("yahoo_nba", "https://sports.yahoo.com/nba/rss/"),
    ],
    "gridiron": [
        ("nfl.com", "https://www.nfl.com/news/rss.xml"),
        ("espn_nfl", "https://www.espn.com/espn/rss/nfl/news"),
        ("cbs_nfl", "https://www.cbssports.com/rss/headlines/nfl/"),
    ],
    "pitch": [
        ("bbc_football", "http://feeds.bbci.co.uk/sport/football/rss.xml"),
        ("guardian_football", "https://www.theguardian.com/football/rss"),
        ("espn_fc", "https://www.espn.com/espn/rss/soccer/news"),
    ],
    "equities": [
        ("sec_8k", "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&company=&dateb=&owner=include&start=0&count=40&output=atom"),
        ("yahoo_finance", "https://feeds.finance.yahoo.com/rss/2.0/headline?s=&region=US&lang=en-US"),
        ("reuters_business", "http://feeds.reuters.com/reuters/businessNews"),
    ],
    "unified": [
        ("espn_all", "https://www.espn.com/espn/rss/news"),
        ("reuters_sports", "http://feeds.reuters.com/reuters/sportsNews"),
    ],
}

def fetch_rss(url, timeout=10):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (dumbmodel news bot) honest-503"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
            return data, None
    except Exception as e:
        return None, str(e)

def parse_rss(data):
    items = []
    try:
        root = ET.fromstring(data)
        # RSS 2.0 <channel><item> or Atom <entry>
        for item in root.findall(".//item") + root.findall(".//{http://www.w3.org/2005/Atom}entry"):
            title = item.findtext("title") or item.findtext("{http://www.w3.org/2005/Atom}title") or ""
            link = item.findtext("link") or ""
            # Atom link href
            if not link:
                link_el = item.find("{http://www.w3.org/2005/Atom}link")
                if link_el is not None:
                    link = link_el.attrib.get("href", "")
            desc = item.findtext("description") or item.findtext("{http://www.w3.org/2005/Atom}summary") or item.findtext("{http://www.w3.org/2005/Atom}content") or ""
            pub = item.findtext("pubDate") or item.findtext("published") or item.findtext("{http://www.w3.org/2005/Atom}published") or ""
            items.append({"title": title.strip(), "link": link.strip(), "desc": desc.strip()[:500], "pubDate": pub.strip()})
    except Exception as e:
        return [], f"parse_error:{e}"
    return items, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="hoops", choices=["hoops","gridiron","pitch","equities","unified"])
    ap.add_argument("--out", default=None, help="out json path")
    args = ap.parse_args()
    domain = args.domain
    out_path = args.out or f"assets/news/raw_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    all_items = []
    provenance = {"domain": domain, "fetch_time": datetime.now(timezone.utc).isoformat(), "sources": [], "n_raw": 0, "errors": []}

    for name, url in SOURCES.get(domain, []):
        data, err = fetch_rss(url)
        if err:
            provenance["sources"].append({"name": name, "url": url, "status": "503", "error": err, "n": 0})
            provenance["errors"].append(f"{name}:{err}")
            continue
        items, perr = parse_rss(data)
        if perr:
            provenance["sources"].append({"name": name, "url": url, "status": "parse_error", "error": perr, "n": 0})
            continue
        h = hashlib.sha256(data).hexdigest()[:12]
        provenance["sources"].append({"name": name, "url": url, "status": "200", "hash": h, "n": len(items)})
        for it in items:
            it["source"] = name
            it["source_url"] = url
            it["fetched_at"] = provenance["fetch_time"]
        all_items.extend(items)

    provenance["n_raw"] = len(all_items)
    # dedup by title+link casefold
    seen = set()
    deduped = []
    for it in all_items:
        key = (it["title"].casefold().strip(), it["link"].casefold().strip())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(it)

    output = {"provenance": provenance, "items": deduped, "domain": domain, "lcg": "20260813->189831298 idx3820 triple[11205,19448,14209] same-link-same-stars"}
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"ingest {domain}: raw={provenance['n_raw']} deduped={len(deduped)} out={out_path} errors={len(provenance['errors'])}")
    # honest 503 if all failed: still writes empty items with provenance
    if len(deduped) == 0 and provenance["errors"]:
        print(f"WARNING honest 503: all sources failed for {domain}, writing empty with provenance", file=sys.stderr)

if __name__ == "__main__":
    main()
