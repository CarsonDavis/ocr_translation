# Read report — p053, reader B

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/B/p053.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/B/p053.json` → `1 ok, 0 failed`, exit 0, no warnings.

## Counts

| item | count |
|---|---|
| body lines (all paragraph blocks) | 32 |
| paragraphs | 4 (13 + 1 + 5 + 13 lines) |
| headings | 1 (`TEXTE.`, spaced caps) |
| markers in body | 9 (`b c d e f g h i k`) |
| margin notes | 9 (keys `b c d e f g h i k`, all matched) |
| foot notes | 0 |
| ornaments | 0 |
| signature | `D iij` |
| catchword | none |
| running head / folio | `PARLEMENT DE THOLOSE.` / `53` |

## Page structure

Running head (spaced caps) with folio `53` at the outer right. Then the tail of an
annotation paragraph continuing from the previous page (13 lines, ends `… celeſte beauté {g}:`).
Then a one-line italic Latin hexameter set apart and indented
(`Exẽplúmq; Dei quiſque eſt, in imagine parua {h}`) — transcribed as a `paragraph` block,
not a heading, per §6. Then the annotation prose resumes flush left for 5 lines and closes
`… falſifié le ſeau du prince {k}.`. Then the display line `TEXTE.` and a new section set in
a noticeably larger type: 13 lines, indented first line, breaking off at `A l'au-`
(`continues_next: true`). Signature `D iij` centred below the last line; the rest of the
page is blank (no block recorded, per §6).

Margin column: nine italic notes running continuously `b`…`k` (no `j`), ending well above
the mid-page; `margin-3.jpg` and `margin-4.jpg` carry no margin text at all. No foot
citation block.

## uncertain[] entries — 9

1. `blocks[0].lines[7]` — **`gtandement`** for *grandement*: wrong sort (t for r), confirmed at high zoom. `sic`.
2. `blocks[0].lines[10]` — **`Lictance`** for *Lactance*: dotted i, wrong sort. Margin note `f` spells the same name `Lactance`. `sic`.
3. `blocks[0].lines[11]` — the print sets a comma immediately after `&` (`pourtraict &,ſimulachre`). Kept; §1 normalization renders it `& , ſimulachre`. **escalate**
4. `blocks[2].lines[1]` — **`impreſſion`** is badly under-inked. Only one clearly inked long-s stem, an x-height stroke, and a half-inked `o` survive. Best reading `impreſſion`; `impreſsion` / `impreſion` cannot be excluded. **escalate**
5. `margin_notes[0].lines[0]` — note `b`, mark after `ſtigmata` is a mid-height horizontal dash (hyphen shape) though no word continues; could be a period set high.
6. `margin_notes[0].lines[1]` — note `b`, mark after `fabricẽ` is a baseline dot (period) although the word continues `ſib` on the next line. Transcribed as printed.
7. `margin_notes[2].lines[1]` — note `d`, `ex eo.` — the `e` is faint and could be `c` (`ex co.`).
8. `margin_notes[2].lines[0]` — note `d`, the capital read as `P.`; conceivably the `ꝑ` abbreviation, but no stroke crosses the descender.
9. `margin_notes[5].lines[0]` — note `g`, `Geneſee. j.` — the final two letters both show the italic e-eye, so `Geneſee`, not `Geneſe`/`Geneſes`. Possible wrong sort.

## Notes for the reconciler

- **Faint inking in one spot only.** The right-hand half of `blocks[2].lines[1]`
  (`par impreſſion, &`) is the one genuinely damaged place on an otherwise clean, sharp
  page. Everything else is legible at 3–8x.
- **Two wrong sorts** (`gtandement`, `Lictance`) are certain, not reading errors — both
  were checked glyph by glyph at 3x+ per the r/t and a/i warning. Both `Guerre` instances
  were checked for the `Cuerre` misprint and are correct `G`.
- **`ſſ` vs `ſs`:** every double-s on this page (`chauſſoit` ×2, `chauſſe`, `impreſſion`)
  was zoomed. The three `chauſſ-` words show two full-height stems joined at the top =
  `ſſ`. Only `impreſſion` is doubtful, and only because of the inking.
- **Note keys `c` and `e`** both begin `l… ſi quis in metallum` — they are distinguished by
  their key letters, which were separately zoomed: note 2's key is a bar-less italic `c`,
  note 4's key is an italic `e` with a clear eye. Note `i` repeats the same citation a
  third time (`metallum. alle / gue.`).
- **`l` without a following period** in notes `c` (`l pen.`) and `d` (`l locum.`), versus
  `l.` with the period in notes `b`, `e`, `i`. Verified at 12x; this is the print, not a
  dropped character.
- **Paragraph continuity judgement:** `blocks[2]` (`Ce ſeroit vn eſpece…`) is flush left,
  not indented, so it is marked `continues_prev: true` — it resumes the annotation prose
  that `blocks[0]` broke off with a colon to introduce the Latin verse. `blocks[1]` (the
  verse) is marked `false`/`false`. If the project prefers the verse to break the chain,
  this is the one structural call on the page worth a second look.
- The `TEXTE.` section is set in a larger type than the annotation above it; that is a type
  change only, recorded by nothing in the schema, so it is flagged here.
- The crops keep a strip of the facing verso along the left (gutter) edge; it was ignored.
