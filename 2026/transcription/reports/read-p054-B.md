# Read report — p054, reader B

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/B/p054.json`

**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/B/p054.json`
→ exit 0 (`1 ok, 0 failed`).
One WARNING, checked against the image and dismissed: `blocks[2].lines[0]: possible normalized
long s (si) in "Si ces teſmoins n'euſſent eſté ſinguliers, chacun de-"`. The word is the
sentence-opening **capital** `Si`; a roman capital S is always the round form, so no long s was
normalized away. Verified at 3.5x on `body-3.jpg`.

## Counts

| | |
|---|---|
| running head | `ARREST DV` (spaced caps) |
| folio | `54` (matches manifest) |
| blocks | 7 |
| paragraphs | 4 |
| headings | 3 (`ANNOTAT, XXXII.` / `TEXTE.` / `ANNOT. XXXIII.`, all spaced caps) |
| body lines (paragraph lines) | 29 (7 + 9 + 10 + 3) |
| markers in body | 3 — `{a}`, `{b}`, `{a2}` |
| margin notes | 3 — keys `a`, `b`, `a2` (16 note lines total: 3 + 4 + 6, plus… see below) |
| foot notes | 0 |
| signature | null |
| catchword | null |
| ornaments | none |
| `uncertain[]` entries | 14 |

Note-line counts: `a` 3 lines, `b` 4 lines, `a2` 6 lines = 13 note lines.

## Page structure (as read)

1. `paragraph`, `continues_prev: true` — 7 lines of large text type, ends the sentence
   `…Iean du Tilh ſon frere.` (no `continues_next`).
2. `heading` — `ANNOTAT, XXXII.` (comma after ANNOTAT, period at the end; both verified at 5x).
3. `paragraph` — 9 lines of the small annotation type; carries markers `{a}` (line 8) and
   `{b}` (line 9).
4. `heading` — `TEXTE.`
5. `paragraph` — 10 lines of large text type, ends `nee de S. Laurens.`
6. `heading` — `ANNOT. XXXIII.`
7. `paragraph`, `continues_next: true` — 3 lines of small type, carries marker `{a2}`,
   ends mid-word `…Martin Guerre arri-` at the foot of the page.

Marker/note cross-check passes: every `{x}` has a note and every note has a marker. The
alphabet **restarts** at the `ANNOT. XXXIII` section, so the third marker is keyed `a2` per
conventions §4.

## `uncertain[]` entries (14) — one line each

1. `blocks[0].lines[5]` — an apostrophe (raised comma) is printed after `mouchoirs` **before**
   the comma: `mouchoirs',`. Transcribed as printed; may be a stray sort.
2. `blocks[4].lines[2]` — the comma ending `lequel,` is small and fused to the foot of the `l`;
   read as a comma at 10x but faint.
3. `blocks[4].lines[3]` — the line-end `Mar` (of *Martin*, continued next line) is set in the
   **smaller annotation type**, not the large text type, and a comma precedes it: `ſoy dire, Mar`.
4. `blocks[4].lines[4]` — `qu il` printed without an apostrophe; sic.
5. `blocks[4].lines[7]` — a round baseline **period** is printed mid-sentence after `l'vne`:
   `perdu l'vne. d'vn coup`; transcribed as printed, sic.
6. `blocks[4].lines[8]` — the word-break sign after `iour` (`iour-`/`nee`) is a small **rounded
   blob** at mid-height, not the clean dash used at `troi-`; transcribed as `-` per §1.
7. `blocks[6].lines[1]` — **`cuy` for `ouy`**: at 9x the first glyph is an open `c` with serif
   terminals, not a closed `o`. Transcribed as printed, sic (wrong sort). Same rounded
   line-break blob after `preu`.
8. `blocks[6].lines[2]` — marker keyed `a2` (alphabet restart); same rounded blob after `arri`,
   read as a hyphen.
9. `margin_notes[0].lines[1]` — the print sets `ex.` then a small gap then the line-end hyphen;
   written `graue. de ex. -` after §1 spacing normalization.
10. `margin_notes[0].lines[2]` — `prelato` is printed with a plain `e`, **not** the `æ` ligature
    (`prælato`); sic.
11. `margin_notes[1].lines[2]` — a **hyphen**, not a period, is printed between `quinto` and
    `D.`, with a visible gap before the `D`: `quinto- D. de`; sic.
12. `margin_notes[1]` — note `b` is printed beside the `TEXTE` heading, not beside a body line,
    so `beside_line` is null.
13. `margin_notes[2]` — note `a2` starts in the gap between `nee de S. Laurens.` and the
    `ANNOT. XXXIII` heading, i.e. **ahead of** its marker line, so `beside_line` is null. Its
    key letter is a heavily inked italic `a`.
14. `margin_notes[2].lines[3]` — **`rutela` for `tutela`**: the first glyph is a short italic `r`
    with a shoulder, plainly different from the tall crossed `t` two letters later in the same
    word. Transcribed as printed, sic.

## Things the reconciler should know

- **Layout.** Verso: margin column on the left, body on the right. The crop keeps a strip of the
  facing recto along the right (gutter) edge — ignored, including the word `Bier…` visible at the
  right of `foot.jpg`.
- **No foot block, no signature, no catchword.** `foot.jpg` shows only the last three body lines
  of the `ANNOT. XXXIII` paragraph and then blank paper to the trimmed edge. The page ends
  mid-word (`arri-`), so `continues_next` is true on the last paragraph.
- **Two type sizes in the body.** The `TEXTE` paragraphs are large; the `ANNOTAT`/`ANNOT`
  paragraphs are the small annotation type. One large-type line (`…ſoy dire, Mar`) ends with
  three letters set in the *small* type — worth a second opinion, as it could equally be read as
  small capitals. I read it as ordinary `M` + lowercase `ar` in a smaller font.
- **The line-break sign is inconsistent.** In the small annotation type it is a clean horizontal
  dash (`de-`, `preu-` in ANNOTAT XXXII, `lite-`, `de-`, `af-` in the margin). In three places
  (`iour`, `preu`, `arri`) it is a small rounded blob at mid-height that could be mistaken for a
  period. All three break a word across the line, so all three are transcribed as `-`.
- **Two likely wrong sorts** on this page, both flagged `sic`: `cuy` for `ouy`
  (`blocks[6].lines[1]`) and `rutela` for `tutela` (`margin_notes[2].lines[3]`). The intended
  Latin is the Decretals title *de tutela*; `c. licet ex quadam` and `c. iam literis` belong with
  it, so the `r` really is a misprint, not my misreading — but reader A should confirm the glyph.
- **`ſſ` audit.** Every double-s on the page was checked at ≥3x and every one is long s + long s
  (`recognoiſſoit`, `n'euſſent`, `confeſſion`, `paſſa`, `cõfeſſe`, `neceſſai`). No `ſs`
  combinations were found; `ſuffiſante`/`ſuffiſant` are `ff` ligature + single long s.
- **Word division.** Several places are set very tight (`de boulet`, `S. Laurens`, `à la charge`,
  `Guerre eſtoit`) and one very loose (`ſingulier s`, `qu il`). Normalized to single spaces per
  §1; `qu il` keeps the missing apostrophe as printed.
- **Paper/ink.** Clean copy, no damage or show-through that obscures text. A few small ink specks
  (notably a raised tick between `eſté` and `fort` on `blocks[6].lines[0]`, and one between
  `dire` and `Mar`) were judged to be specks, not sorts.
