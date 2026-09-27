# read-p065-A

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/A/p065.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/A/p065.json` → `1 ok, 0 failed`, exit 0, no warnings.

## Counts

| | |
|---|---|
| body lines | 30 (14 + 13 + 3) |
| paragraphs | 3 |
| headings | 2 (`ANNOTAT. XLIII.`, `TEXTE.`, both spaced caps) |
| markers in body | 3 (`{a}`, `{b}`, `{c}`) |
| margin notes | 3 (keys a, b, c) |
| foot notes | 0 |
| running head | `PARLEMENT DE THOLOSE.` |
| folio | `65` (matches manifest) |
| signature | `E` |
| catchword | none |
| ornaments | none |
| uncertain entries | 11 |

Page structure: continuation of the previous page's TEXTE paragraph (14 lines, `continues_prev: true`,
ends the paragraph at "par entre eux plus ſemblables."), then the heading `ANNOTAT. XLIII.`, the
annotation in small type (13 lines, self-contained on this page), then the heading `TEXTE.`, then
3 lines of large-type text broken off at "audit du" (`continues_next: true`). Margin column is
empty for the whole first paragraph; the three notes sit beside annotation lines 4, 6 and 10.

The signature `E` matches the gathering pattern of the finished pages (A on 1/3/5/7, B on
17/19/21/23, C on 33/35/37/39, D on 49/51/53/55 → E begins at 65).

## uncertain[] entries (11)

1. `blocks[0].lines[2]` — **sic `congoiſſance`** for *cognoiſſance*; g and o transposed. True `ſſ` here.
2. `blocks[0].lines[4]` — **`impoſsi-` is `ſ` + round `s`**, not `ſſ`; confirmed twice (body-2 and margin-1).
3. `blocks[0].lines[9]` — **`blãce` carries a tilde, not an acute**; compared with the acute of `rapporté`/`reprouué` in the same type size.
4. `blocks[2].lines[2]` — **turned sort: the m of `mouſches` is set on its side**, hence the wide gap; transcribed as an upright m.
5. `blocks[2].lines[6]` — the print really sets **`dit. il,`** with a period between the words.
6. `blocks[2].lines[8]` — **the line-end break sign after `di` is a round baseline dot**, not the slanted hyphen used elsewhere on the page; written `-` per convention, but `di.` is a live alternative. *(Most likely A/B divergence point.)*
7. `blocks[2].lines[11]` — **`dit'il`** is printed with an apostrophe, not a hyphen.
8. `blocks[4].lines[2]` — **sic `pre-` / `enu`** for *preuenu*; the u is missing, and there is no second hyphen.
9. `margin_notes[0].lines[1]` — the numeral is italic **`ij.`** (i + j, one dot), easily misread as `y.`; = Cicero, *Academica* II.
10. `margin_notes[2].lines[2]` — **`Menechmus.`** final stop: baseline dot with a faint mark above it; a worn colon cannot be ruled out.
11. `margin_notes[0]` (page-level) — **a stray vertical ink stroke** stands in the margin beside the annotation's first line, well above note *a*; no letterform, not transcribed.

## Notes for the reconciler

- **Condition.** Good. A brown damp/foxing stain runs across the top right of the leaf, through the
  right end of the running head and the first two body lines; it does not obscure any letter. Ink
  is even throughout; no tears, no gutter loss. Some show-through from the verso is visible in the
  blank lower margin but nothing overlaps the type.
- **Three real misprints on one page** (`congoiſſance`, the turned m of `mouſches`, and `pre-enu`),
  all transcribed as printed. Reader B should be checked against these specifically — the eye
  supplies the expected word at all three places.
- **The `di-` / `di.` break at annotation line 9** is the one spot I could not settle from the image
  alone; the mark is a period-shaped dot at baseline height, clearly unlike the three hyphens on the
  same strip. I followed the convention rule ("transcribe any line-end word-break sign as a single
  `-`") because the word is plainly *diſoit*, but a reader who transcribes literally will write `di.`
- **`ſſ` vs `ſs`** on this page: `congoiſſance` and `Meſſenio` are true `ſſ`; `impoſsi-` and `auſsi`
  are `ſ` + round `s`. All four zoomed to 300–450%.
- **Markers.** The alphabet restarts at `a` with the new annotation (ANNOTAT. XLIII); the first
  paragraph, which belongs to the previous TEXTE section, carries no markers at all. Every `{x}`
  has a note and every note has a marker.
- No foot citation block on this page.
