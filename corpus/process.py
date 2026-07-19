#!/usr/bin/env python3
import re, random, sys, os
from collections import Counter

BASE = "/home/normie/Documents/Projects/GeoScribe/corpus"
RAW = os.path.join(BASE, "raw_extracts.txt")

# Georgian Mkhedruli letters U+10D0-U+10F0 (ა-ჰ), 33 letters
GEO = set(chr(c) for c in range(0x10D0, 0x10F0 + 1))
DIGITS = set("0123456789")
PUNCT = set(".,!?;:…()-„“«»")  # note: two double-quote chars used in Georgian: „ and “
SPACE = set(" ")
# Full allowed charset
ALLOWED = GEO | DIGITS | PUNCT | SPACE

# Regex-safe sentence splitter: split after . ! ? … when followed by whitespace/newline
SENT_SPLIT = re.compile(r'(?<=[.!?…])\s+')

def geo_word_count(s):
    words = re.findall(r'[ა-ჰ]+', s)
    return len([w for w in words if len(w) >= 1])

def main():
    raw = open(RAW, encoding="utf-8").read()
    # split records then normalize whitespace within
    records = raw.split("\x1e")
    sentences = []
    for rec in records:
        # collapse newlines to spaces for sentence splitting but keep sentence boundaries
        text = rec.replace("\n", " ")
        text = re.sub(r'\s+', ' ', text).strip()
        for sent in SENT_SPLIT.split(text):
            sentences.append(sent.strip())

    kept = []
    for s in sentences:
        if not (20 <= len(s) <= 120):
            continue
        if not s:
            continue
        # charset check: all chars in ALLOWED
        if any(ch not in ALLOWED for ch in s):
            continue
        # must contain at least 3 Georgian words
        if geo_word_count(s) < 3:
            continue
        kept.append(s)

    # exact dedup preserving nothing (set), then sort for determinism then shuffle seed 42
    uniq = list(dict.fromkeys(kept))
    uniq = sorted(set(uniq))
    rng = random.Random(42)
    rng.shuffle(uniq)

    with open(os.path.join(BASE, "sentences.txt"), "w", encoding="utf-8") as f:
        for s in uniq:
            f.write(s + "\n")

    # word list: only Mkhedruli letters, length 2-20, by frequency top 8000
    words = re.findall(r'[ა-ჰ]+', raw)
    cnt = Counter(w for w in words if 2 <= len(w) <= 20)
    top = [w for w, _ in cnt.most_common(8000)]
    with open(os.path.join(BASE, "words.txt"), "w", encoding="utf-8") as f:
        for w in top:
            f.write(w + "\n")

    print("sentences", len(uniq))
    print("words", len(top))

if __name__ == "__main__":
    main()
