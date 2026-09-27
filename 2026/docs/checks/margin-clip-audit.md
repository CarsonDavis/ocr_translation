# Margin-clip audit — regenerated margin strips

**Date:** 2026-09-21
**Scope:** finalized pages `p000-title` (skipped, no notes), `p000-argument`, `p001`,
`p002`, `p003`, `p004`, `p005`, `p006`, `p008`, `p041`, `p044`, `p159`.

**Question:** the old `pages/strips/<id>/margin-K.jpg` crops were too narrow on the outer
side, so a long citation line could lose its last word (`a Ciceron en` read as
`a Ciceron`). The strips have been regenerated out to the paper edge. For every line of
every `margin_notes` / `foot_notes` entry in `transcription/final/<id>.json`, does the
print continue past where the transcription ends?

This audit judges **completeness at the outer end of the line only**. It does not
re-transcribe and does not judge letter-level readings.

## Findings

| page | note key | line index | transcribed | what the print shows beyond it | verdict |
|---|---|---|---|---|---|
| — | — | — | — | *no line was found to be short* | — |

No clipped line was found on any page. Every note line ends in the print exactly where
the transcription ends, and every note's last line is the last line printed for that note.

## Lines checked

| page | margin notes | foot notes | lines checked |
|---|---|---|---|
| p000-title | — | — | skipped (no notes) |
| p000-argument | 0 | 0 | 0 (margin column verified blank) |
| p001 | 0 | 0 | 0 (margin column verified blank) |
| p002 | 6 (`a`–`f`) | 0 | 32 |
| p003 | 10 (`g`–`q`) | 1 (`r`) | 51 |
| p004 | 8 (`ſ`–`b`) | 0 | 39 |
| p005 | 6 (`a`–`f`) | 0 | 26 |
| p006 | 7 (`g`–`n`) | 0 | 38 |
| p008 | 1 (`a`) | 0 | 3 |
| p041 | 11 (`a`–`l`) | 3 (`m`–`o`) | 50 |
| p044 | 2 (`a`, `b`) | 0 | 6 |
| p159 | 13 (`b`–`o`) | 2 (`p`, `q`) | 46 |
| **total** | **64** | **6** | **291** |

## Why the regenerated crops are sufficient

The new margin strip runs from the outer paper edge to 60px inside the body column
(`MARGIN_INWARD` in `scripts/crop.py`), so the whole margin column plus the start of the
body is in frame on every page:

- **Recto pages** (`p001`, `p003`, `p005`, `p041`, `p159`) carry the notes in the right
  margin; the strip now ends at the paper edge, leaving 200–300px of blank paper past the
  longest line. Line ends cannot be cut.
- **Verso pages** (`p000-argument`, `p002`, `p004`, `p006`, `p008`, `p044`) carry the
  notes in the left margin, where the line ends face the body. Body type is visible at the
  inner edge of every strip, which proves nothing between the note and the body was lost.

## Observations (not clipping)

These are places where a note line is real and complete but does **not** appear inside the
`margin-*.jpg` strips, so they should not be mistaken for phantom or missing lines:

- **p003 `q`[4] `loix.`** — the last line of the bottom margin note is set as a run-over in
  the bottom margin, under the body column. It is visible in `p003/foot.jpg`, at the left,
  ahead of foot note `r`. Complete.
- **p041 `n`[0] and `o`[0]** — the foot notes run the full leaf width; their right-hand
  tails (`c. final de`, `rnitatis.`) fall under the margin column and appear at the left of
  `p041/margin-2.jpg`. `p041/foot.jpg` is blank for this page (its bottom-18% band sits
  below the type), so the foot notes read from the bottom of `body-2.jpg`. Complete.
- **p159 `o`[1]** — `tem pro redemptione, & P. ſuyuant de eccleſ. titu. cdlla. vij.` is a
  full-width run-over line that sits just above the band `p159/foot.jpg` covers. Read from
  `pages/full/p159.jpg`. Complete.
- **p041 `e`[2]–[3]** — the `[...]` in `l. penul[...] deſ-` / `ſ[...] alleguee.` is a pen
  stroke / ink smear across the middle of those lines on this bitonal scan, not a crop.
  Both lines end normally (`deſ-`, `alleguee.`).
- **p041 `g`[1]** — a faint dash-like mark sits after `tatiõ premiere` at the outer edge.
  Nothing follows it; the note reads complete (`En l'annotation premiere`).

## Summary

- **Pages with clipped lines:** none.
- **Total clipped lines:** 0 (of 291 note lines checked across 9 pages that have notes).
