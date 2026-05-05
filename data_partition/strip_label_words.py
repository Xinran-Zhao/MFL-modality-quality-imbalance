"""Strip label-leaking words from findings_text in prepared_data.csv.

Why
---
The 8 binary labels in prepared_data.csv were derived from the `Problems`
column of indiana_reports.csv (originally MeSH-encoded). The `findings`
narrative was written by the same radiologist describing the same
patient, so the disease words that *define* the labels recur naturally
in the input text. For example, 100% of Calcified_Granuloma=1 samples
contain the literal phrase "granuloma" in findings.

This script replaces every occurrence of the label vocabulary (and
obvious clinical synonyms) with the existing `[DEIDENTIFIED]` token —
already used elsewhere in the same reports — so the BERT tokenizer sees
a familiar mask and sentence grammar stays intact.

What is NOT stripped
--------------------
- The `No_Finding` lexicon ("normal", "unremarkable"). These words are
  too pervasive in CXR narrative grammar to mask without destroying
  syntax; No_Finding leakage is also weaker because the label is
  effectively defined by *absence* of the other seven.
- Anything outside `findings_text`. The label columns and the `Problems`
  source column are untouched (the model never reads `Problems` anyway).

Usage
-----
    python -m data_partition.strip_label_words \
        --in  /data/amciilab/xinran/indiana_cxr/prepared/prepared_data.csv \
        --out /data/amciilab/xinran/indiana_cxr/prepared/prepared_data_stripped.csv

After running, train with `--csv-path <stripped>` and compare to the
original sweep on the same partitions / seeds.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd


# Token used to replace stripped words. Already appears in the original
# reports (e.g. for redacted PHI tokens like XXXX), so the tokenizer is
# already familiar with it -- no new vocab item needed.
MASK_TOKEN = "[DEIDENTIFIED]"

# Per-label patterns. Order matters: longer/more specific phrases first
# so we don't double-mask substrings (e.g., "calcified granuloma" before
# "calcified"). Patterns are case-insensitive (compiled with re.I).
PATTERNS = {
    "Calcified_Granuloma": [
        r"\bcalcified\s+granuloma\w*",
        r"\bgranuloma\w*",                # granuloma, granulomas, granulomata,
                                          # granulomatous, granulomatosis
    ],
    "Cardiomegaly": [
        r"\benlarged\s+cardiac\s+silhouette\b",
        r"\bcardiac\s+enlargement\b",
        r"\benlarged\s+heart\b",
        r"\bheart\s+is\s+enlarged\b",
        r"\bcardiomegaly\b",
    ],
    "Atelectasis": [
        r"\batelecta(?:s|t)\w*",          # atelectasis, atelectatic, atelectases
    ],
    "Pleural_Effusion": [
        r"\bpleural\s+effusions?\b",
        r"\beffusions?\b",                # captures "effusion" / "effusions" alone
    ],
    "Opacity": [
        r"\bopacit\w*",                   # opacity, opacities, opacification
    ],
    "Calcinosis": [
        r"\bcalcinos\w*",                 # calcinosis
        r"\bcalcifications?\b",
        r"\bcalcified\b",                 # bare "calcified"; longer phrases above already consumed
    ],
    "Airspace_Disease": [
        r"\bair[\s\-]?space\b",           # airspace / air space / air-space
        r"\bconsolidations?\b",
    ],
}

# Compile once, keeping per-label tags so we can report stats.
COMPILED = [
    (label, re.compile(pat, flags=re.IGNORECASE))
    for label, pats in PATTERNS.items()
    for pat in pats
]


def strip_text(text: str, counter: Counter | None = None) -> str:
    """Replace every label-vocab span in `text` with MASK_TOKEN.

    Patterns are applied in declaration order, so longer/more specific
    phrases (e.g. 'calcified granuloma') consume their input before
    shorter siblings (e.g. 'calcified') see them.
    """
    if not isinstance(text, str) or not text:
        return text
    out = text
    for label, rx in COMPILED:
        if counter is not None:
            n = len(rx.findall(out))
            if n:
                counter[label] += n
        out = rx.sub(MASK_TOKEN, out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, type=Path,
                    help="Path to prepared_data.csv")
    ap.add_argument("--out", dest="out_path", required=True, type=Path,
                    help="Path to write prepared_data_stripped.csv")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.in_path)
    print(f"loaded {len(df)} rows from {args.in_path}")

    counter: Counter = Counter()
    df["findings_text"] = df["findings_text"].apply(lambda t: strip_text(t, counter))

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_path, index=False)
    print(f"wrote stripped CSV -> {args.out_path}")

    print("\n=== spans replaced per label-pattern group ===")
    total = 0
    for label in PATTERNS:
        n = counter.get(label, 0)
        total += n
        print(f"  {label:<22s} {n:>6d}")
    print(f"  {'TOTAL':<22s} {total:>6d}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
