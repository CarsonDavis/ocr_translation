# Read report — p055, reader B

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/B/p055.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/B/p055.json` → `1 ok, 0 failed`, exit 0, no warnings.

## Counts

| item | count |
|---|---|
| body lines | 26 (5 + 7 + 6 + 7 + 1) |
| paragraphs | 5 |
| headings | 4 (`TEXTE.`, `ANNOTAT. XXXIIII.`, `TEXTE.`, `ANNOTAT. XXXV.`) |
| markers `{x}` in body | 0 |
| margin notes | 1 (key `a`) |
| foot notes | 0 |
| ornaments | 0 |
| `uncertain[]` entries | 4 |

Page furniture: running head `PARLEMENT DE THOLOSE.` (spaced caps), folio `55`, signature `D iiij`, catchword `null`.

## Layout

Recto, folio 55. Top paragraph continues from p054 (first line opens mid-word, `ua en tel equipage` — almost certainly the tail of `arri-` / `arriua`), so `continues_prev: true`. Then `TEXTE.` + a 7-line large-type text block; `ANNOTAT. XXXIIII.` + a 6-line small-type annotation; `TEXTE.` + a 7-line large-type text block; `ANNOTAT. XXXV.` + a single line that runs onto the next page (`continues_next: true`, line ends with `&`). Signature `D iiij` centred under that last line. Nothing else in the foot; no foot citation block, no catchword.

Only one marginal note on the page, in the outer (right) margin, keyed `a`, set beside the line `contenteray pour le preſent, renuoyer le lecteur à ce`. Margin strips 1 and 4 are empty. The narrow column of type along the left (gutter) edge of the read image and of margin-1/-2/-3 belongs to the facing verso (p054) and was ignored.

## uncertain[] entries (4)

1. `margin_notes[0]` — **escalated.** The note keyed `a` (`Plutarque / au liure v. de / placit. Philoſ.`) has **no marker printed in the body.** I zoomed the whole of `qu'en a eſcrit Plutarque.` and the two lines above it; there is nothing after the final period and no raised letter anywhere in the annotation. The reconciler should decide whether to leave the note unmarked.
2. `blocks[0].lines[3]` — `d'eſt eſpoinçonnez`: **sic**, the print sets `d'eſt` (d, apostrophe, e, ſt-ligature) followed by a full word space where the sense wants `d'eſtre`. Checked at 800%; no `re`, no hyphen.
3. `blocks[6].lines[3]` — two issues on one line: a small comma-shaped mark sits just below the baseline between `&` and `Gaſcon` (broken low comma or ink speck — **not** transcribed); and `entendible` is set as two words, `enten dible`, with a full word space, transcribed as printed.
4. `blocks[6].lines[5]` — the apostrophe in `n'en` is printed as a small caret/circumflex-shaped raised mark sitting over the `n`, unlike the comma-shaped apostrophes elsewhere on the page. Read as an apostrophe.

## Checks the brief called out explicitly

- **`ſſ` vs `ſs`.** Both double-s words on the page were zoomed to 3x+ and they differ: `eſdictes enqueſtes, confirment auſsi.` (blocks[2], last line) is **`ſs`** — long s then a clearly round short s. `Ceſte preuue auſſi n'eſtoit pas` (blocks[4], first line) is **`ſſ`** — a true long-s pair. This is the most likely place for a diff with reader A.
- **Punctuation.** Every clause-boundary mark was zoomed. Colons confirmed at `S. Quentin:`, `tude:`, `concluante:`, and `du pays :&` (printed with a space before the colon and the `&` jammed against it; normalized per §1 to `du pays: & neantmoins`). `Quentin:les` and `enqueſtes,confirment` etc. are set tight in the print and normalized to one space after the mark.
- **Wrong sorts.** I specifically re-checked `de boulet` — at 700% the final glyph has a crossbar extending to the *left* of the stem and an ascender, i.e. a genuine `t`, **not** the `bouler` misprint it looks like at reading size. Same test on `n'eſtoit` (blocks[4], line 1) → `t`. The one real anomaly I found is `d'eſt` (entry 2 above).

## Notes for the reconciler

- Paper is clean, ink is even; no damage, no show-through worth flagging. Small ink specks under `ua` (line 1) and under `&` on `çois, & Gaſcon` are the only stray marks.
- Word-division normalizations applied per §1: `SãxiGuer` → `Sãxi Guer`, `re,fils` → `re, fils`, `té,cõme` → `té, cõme`, `teſmoins,ouys` → `teſmoins, ouys`, `enqueſtes,confirment` → `enqueſtes, confirment`, `commenous` → `comme nous`, `bien,qu'` → `bien, qu'`, `çois,&` → `çois, &`, `Gaſcon,peu` → `Gaſcon, peu`, `dible,ſi` → `dible, ſi`, `au liure v.de` → `au liure v. de`.
- Line-end word breaks without a hyphen are frequent in the large-type blocks (`Guer`/`re`, `rappor`/`té`, `ſimili`/`tude`, `Frã`/`çois`) and are transcribed as printed, per §1.
