#!/usr/bin/env python3
import json, time, urllib.request, urllib.parse, sys, os

API = "https://ka.wikipedia.org/w/api.php"
OUT = "/home/normie/Documents/Projects/GeoScribe/corpus/raw_extracts.txt"

def fetch_batch():
    params = {
        "action": "query",
        "generator": "random",
        "grnnamespace": "0",
        "grnlimit": "20",
        "prop": "extracts",
        "explaintext": "1",
        "exlimit": "20",
        "format": "json",
    }
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "GeoScribe-corpus-builder/1.0 (buildable.dev@gmail.com)"})
    with urllib.request.urlopen(req, timeout=40) as r:
        data = json.load(r)
    texts = []
    for pg in data.get("query", {}).get("pages", {}).values():
        ex = pg.get("extract")
        if ex:
            texts.append(ex)
    return texts

def main():
    passes = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    seen_titles = set()
    total_chars = 0
    with open(OUT, "a", encoding="utf-8") as f:
        for i in range(passes):
            try:
                texts = fetch_batch()
            except Exception as e:
                print("err", e, file=sys.stderr)
                time.sleep(2)
                continue
            for t in texts:
                f.write(t.replace("\x00", " ") + "\n\x1e\n")  # record sep
                total_chars += len(t)
            f.flush()
            if i % 10 == 0:
                print(f"pass {i}/{passes} total_chars={total_chars}", file=sys.stderr)
            time.sleep(0.3)
    print("done total_chars", total_chars, file=sys.stderr)

if __name__ == "__main__":
    main()
