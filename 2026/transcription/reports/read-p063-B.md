# Read report — p063, reader B

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/B/p063.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/B/p063.json` → `1 ok, 0 failed`, exit 0, no warnings.

## Counts

| | |
|---|---|
| body lines | 29 (6 in the annotation paragraph + 23 in the TEXTE paragraph) |
| paragraphs | 2 |
| headings | 1 (`TEXTE.`, spaced capitals) |
| markers in the body | 2 (`{a}`, `{b}`) |
| margin notes | 3 (keys `a`, `b`, and one printed `o`) |
| foot notes | 0 |
| ornaments | 0 |
| `uncertain[]` entries | 10 |

Page furniture: running head `PARLEMENT DE THOLOSE.` (spaced caps, closed up), folio `63`
(matches the manifest), no signature, no catchword, no foot citation block.

## Layout

Recto. The page opens mid-sentence with the tail of an annotation paragraph (6 lines, small
type, `continues_prev: true`), then the heading `TEXTE.`, then a TEXTE paragraph in the large
face (23 lines) that runs off the foot of the page (`continues_next: true`) — it ends
"vn tel coffre quand ie parti: ce que fut". The margin column holds three citation notes, all
beside the annotation paragraph; the TEXTE paragraph carries no markers and no notes.
`foot.jpg` and `body-8.jpg` are blank paper.

## `uncertain[]` entries (10)

1. **`blocks[2].lines[22]` — "vn tel coffre…" (escalate)** — the n of `vn` prints as two bare
   stems: no shoulder arch, no bottom bow. Read as `vn`; a literal reading of the shape would
   give `vu`. See below.
2. **`margin_notes[2]` — key (escalate)** — the third note's key letter is a closed round `o`
   where the run a, b calls for `c`; and no marker for it exists anywhere in the body.
3. **`blocks[0].lines[4]`** — abbreviation marks on that line: `q̃` (straight bar = que),
   and the slanted strokes over `Frãçois` and `Ieã` read as nasal tildes.
4. **`margin_notes[0].lines[3]`** — `au c. præterea.` is set with a true `æ` ligature plus the
   italic curled-top `t`; at reading size it looks like "praterea".
5. **`margin_notes[2].lines[1]`** — the capital `P` in `la. ij. P. ſi du-` is the printer's
   substitute sort for `§`, transcribed as the P that is printed.
6. **`blocks[1]` — `TEXTE.`** — the final E is a visibly smaller sort after a wider gap; still
   a capital E, so the heading is `TEXTE.`.
7. **`blocks[2].lines[0]`** — the paragraph opens with a large swash Q two lines deep; an
   oversize type capital, not a woodcut/decorated initial, so no `ornaments` entry.
8. **`blocks[2].lines[3]`** — line ends `ren` / next line `controit` with no word-break sign
   printed (same at `tinſ` / `mes`); normal for this print, recorded so it is not read as a
   lost hyphen. `ſalüoit` carries a diaeresis.
9. **`margin_notes[2]` — placement** — how the three `beside_line` values were measured.
10. **`blocks[0]` — page furniture and general** — folio, head, absence of signature/catchword/
    foot block, damp staining, and the full list of places where §1 spacing was normalized.

## What the reconciler should know

- **The one real crux is `vn` in the last body line.** At 12x, and again with contrast boosted,
  the word is `v` followed by two bare stems with a clean white gap between them from top to
  bottom. Compared on the same page: every other `vn` (`eſtions à vn`, `quelq'vn`, `vn tel, &
  tel propos`) shows a solid thick shoulder arch, and the `u` of `quand` five words later on
  this same line shows a solid bottom bow. Neither is present here. I transcribed `vn` because
  the sense requires it and because the thin apex of an n is the first part to fail, but a
  reader taking the shape at face value would write `vu`, and the other reader may well have.
  A human eye on the full-resolution scan would settle it.
- **The third margin note is doubly irregular.** Its key letter is unambiguously a closed `o`
  (the bowl closes on the right; the italic `c` of `c. per tuas` two lines above is open), and
  the body offers it no marker at all — the annotation paragraph ends `de quelq'vn.` with a
  plain period, checked at 6x. It is recorded with `"key": "o"` as printed. If the project
  prefers the alphabet over the shape, `c` is the intended key, and the missing marker is a
  compositor's omission (or the note belongs to text that ran over from p062).
- **No preceding page was supplied** ("Preceding pages: none"), so `continues_prev` on the
  first paragraph is inferred from the page itself — the first line begins `vn faict fort
  ambigu` in mid-sentence, mid-clause — and the marker run could not be checked against p062.
  Note that `transcription/final/` does contain pages up to p058 but no p059–p062.
- **Double-s.** Every double-s on the page was checked at ≥4x and all are `ſſ` (long s + long
  s, as a ligature): `aſſeurent`, `cognoiſſance`, `paſſées`, `deſſus`, `chauſſes`, `laiſſay`,
  `ſuffiſante` (second s long, word-final s round). No `ſs` occurs on this page.
- **Condition.** Light brown damp staining across the head and the top right of the body
  column, plus a few foxing specks. None of it touches or obscures type; ink is even and
  strong throughout, and the page is one of the cleaner ones.
- **Spacing.** Heavily normalized per §1 — the compositor sets `Pierre,ou`, `filiatiõb.ou`,
  `Quatriemement,preſque`, `eſtionsà`, `le s chauſſes`, `blesde taffetas`, `c.per`,
  `c.præterea`, `la.ij.P.`, `quẽ ad.teſt.` tight, and sets a space *before* the comma in
  `ambigu ,` and before the colon in `cognuz :`, `particulierement :`, `parti :` and the
  period in `quelq'vn .`.
