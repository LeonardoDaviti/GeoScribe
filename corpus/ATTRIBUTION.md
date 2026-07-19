# Corpus Attribution

The text in `sentences.txt` and `words.txt` is derived from the **Georgian Wikipedia** (https://ka.wikipedia.org).

- **Source:** Georgian Wikipedia article plain-text extracts, retrieved via the MediaWiki API (`action=query`, `generator=random`, `prop=extracts`, `explaintext=1`).
- **License:** Text is available under the **Creative Commons Attribution-ShareAlike 4.0** license (CC BY-SA 4.0). See https://creativecommons.org/licenses/by-sa/4.0/
- **Retrieval date:** 2026-07-19

## Cleaning rules

Sentences (`sentences.txt`):
- Extracts split into sentences on `.`, `!`, `?`, `…` followed by whitespace/newline.
- Kept only sentences whose characters are ALL within the allowed charset:
  the 33 Mkhedruli letters ა–ჰ (U+10D0–U+10F0), space, digits 0–9, and the
  punctuation `. , ! ? ; : … ( ) - „ “ « »`. Any sentence containing a character
  outside this set (Latin letters, apostrophes, other Unicode, etc.) was rejected.
- Length filter: 20–120 characters.
- Must contain at least 3 Georgian words.
- Exact duplicates removed; order shuffled deterministically (seed 42).

Words (`words.txt`):
- Unique tokens of Mkhedruli letters only, length 2–20 characters,
  ranked by frequency, top 8,000.

## Counts

- Sentences: **10,525**
- Words: **8,000**
- Out-of-charset lines: **0** (verified)
