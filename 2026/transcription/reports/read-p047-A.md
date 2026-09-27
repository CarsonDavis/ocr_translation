# Read report — p047, reader A

**Output path:** `/Users/cdavis/github/translator/2026/transcription/reads/A/p047.json`

**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/A/p047.json`
→ `1 ok, 0 failed`, exit code 0, no warnings (no long-s warnings, no NFC warnings).

## Counts

| thing | count |
|---|---|
| body lines | 29 |
| paragraphs | 1 |
| headings | 0 |
| markers `{x}` | 0 |
| margin notes | 0 |
| foot notes | 0 |
| ornaments | 0 |
| `uncertain[]` entries | 15 (3 escalated) |

Running head `PARLEMENT DE THOLOSE.` (spaced caps, closed up), folio `47` (matches the
manifest). `signature: null`, `catchword: null`.

## Page structure

The whole page is a single 29-line paragraph in the large TEXTE type, filling the column
from the running head to the foot. No heading, no decorated initial, no rule, no ornament,
no break of any kind in the column. `continues_prev: true` (opens mid-sentence with
"grãde ſimilitude"), `continues_next: true` (last line ends "enquis d'office", no terminal
punctuation, no catchword).

The outer margin is completely blank down the whole page — margin-1.jpg through
margin-4.jpg show clean unprinted paper beside every body line — and foot.jpg shows nothing
below the last body line. So there are no letter markers and no citations on this page, and
the marker alphabet is not advanced here: whatever letter p046 ended on carries straight
over to p048.

Note that the context page supplied, p044, is **not** the immediately preceding page
(p045 and p046 are missing from `transcription/final/`), so the sentence join at the top of
p047 could not be checked against a transcription.

## `uncertain[]` entries — one line each

1. **`blocks[0].lines[5]` "frõt"** *(escalate)* — printed f-r-õ-t, not "fort"; almost
   certainly a transposition misprint, transcribed as printed.
2. **`blocks[0].lines[17]` "Cuerre"** *(escalate)* — the initial is an open C with no bar or
   stem; misprint for "Guerre", transcribed as printed.
3. **`blocks[0].lines[20]` period after "preuenu"** *(escalate)* — a real but very lightly
   inked round dot on the baseline; kept as a period although the sense argues against it.
4. **`blocks[0].lines[23]` "aſsiſtans"** — long ſ + round s, not ſſ; verified at 6x.
5. **`blocks[0].lines[9]` "celuy martin reſſẽblẽt"** — "martin" printed lowercase (sic); the
   l of "reſſẽblẽt" is almost unprinted but present at 7x.
6. **`blocks[0].lines[18]` "lad"** — no abbreviation point here, unlike "lad." on line 14; sic.
7. **`blocks[0].lines[11]` "cõdẽné"** — two tildes and one acute on the same word,
   distinguished at 5x against "rapporté" and "q̃".
8. **`blocks[0].lines[22]` "plꝰ"** — raised us sign after "pl", transcribed ꝰ (U+A770).
9. **`blocks[0].lines[4]` "p̃uenu" and two colons** — tilde over the p (p + U+0303); both
   marks on the line are colons (two dots each), not periods.
10. **`blocks[0].lines[15]` "Toloſe"** — printed without the h although the running head of
    the same page reads THOLOSE; sic. "laq̃lle" = laquelle (q + U+0303).
11. **`blocks[0]` word division** — eight places where the compositor set two words tight,
    normalized per §1 (ſõt auſ, auec le, en pleine, ſuffiſãment in, qu'il ſeroit,
    enquis d'office, grãde ſimilitude, aſſeurer ſi, attẽdu l'importãce).
12. **`blocks[0]` punctuation spacing** — five places with a justification space before the
    mark and thirteen with no space after it, all normalized per §1.
13. **`blocks[0]` line-end breaks** — hyphens at lines 1, 3, 6, 9, 18, 23; lines 2, 5, 12, 28
    break a word with no hyphen (normal for this print, recorded so it is not read as an omission).
14. **`margin_notes`** — no margin notes and no markers anywhere on the page; margin strips
    and foot block are blank.
15. **`blocks[0]` block structure** — one paragraph, continues_prev and continues_next both
    true, and the p044 context is not the immediately preceding page.

## What the reconciler should know

- **Three likely disagreement points, all of them misprints I kept as printed:** `frõt`
  (line 6), `Cuerre` (line 18), and the faint period after `preuenu` (line 21). A reader
  reading for sense will produce `fort`, `Guerre`, and no period. All three were checked at
  high magnification with a same-page control glyph (the correctly-printed `fort` on line
  10, the correctly-printed `Guerre` on line 4, and the solid periods on lines 10 and 19),
  and the reading stands on the letterforms, not the sense.
- **Double s:** the page has six double-s words. Five are genuine `ſſ` (`aſſeurer`,
  `reſſẽble`, `reſſẽblẽt`, `perſuadaſſent`, `aſſeurée`); exactly one is `ſs` —
  `aſsiſtans` on line 24. That one is the trap.
- **Tilde vs acute:** this fount's tilde is a horizontal, slightly wavy bar and its acute is
  a short stroke slanting up to the right; they are easy to level at reading size. Every
  mark on the page was checked at 5x or better. `cõdẽné` (line 12) carries both kinds.
- **Condition:** the paper is clean and the impression is good. Two spots are lightly
  inked: the `l` of `reſſẽblẽt` (line 10) and the period after `preuenu` (line 21). No
  damage, no show-through worth noting, no gutter loss — the inner edge of every strip
  shows only the facing leaf, which was ignored.
- **Layout:** unusually plain for this book — one uninterrupted 29-line paragraph of the
  large TEXTE type, with no marginal apparatus at all. It is a stretch of the narrative
  Arrest, not an annotation.
