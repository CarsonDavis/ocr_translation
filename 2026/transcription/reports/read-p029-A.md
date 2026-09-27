# read-p029-A

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/A/p029.json`
(reader A, model opus; validator exits 0)

## Counts

| item | count |
|---|---|
| body lines (all paragraph blocks) | 29 |
| paragraphs | 3 |
| headings | 2 (`TEXTE.`, `ANNOTAT. XIIII.`) |
| markers in body | 1 (`{a}`) |
| margin notes | 1 (key `a`) |
| foot notes | 0 |
| `uncertain[]` entries | 6 |

Page furniture: running head `PARLEMENT DE THOLOSE.` (spaced caps), folio `29`,
signature `null`, catchword `null`, no ornaments.

## Layout

Recto. Three-line tail of the paragraph carried over from p028 (`continues_prev: true`),
then the spaced-caps heading `TEXTE.`, then a 20-line paragraph in the large text type
ending cleanly at `ce iour inouye eſpece de crime.`, then the spaced-caps heading
`ANNOTAT. XIIII.`, then the first 6 lines of the annotation in the small type; the
annotation runs on to p030 (`continues_next: true`). No foot citation block, no
signature, no catchword — the lower third of the page is blank.

The only marginal note is `Eccleſiaſtique / c. x.`, in the outer (right) margin beside
the fourth annotation line. The column of text along the LEFT (gutter) edge of the crops
is the facing page and was ignored.

## uncertain[] entries

1. `blocks[4].lines[3]` — the marker letter before `lequel` is small italic and badly
   blotted; read `{a}` (first marker of the new ANNOTAT. XIIII section).
2. `margin_notes[0]` — the margin note carries no printed key letter; key `a` is taken
   from the body marker. Second line read `c. x.` (= chapter 10); the italic x is small
   and the final point faint.
3. `blocks[4].lines[1]` — after `Sirach` a raised comma/apostrophe-shaped mark precedes
   the baseline comma; transcribed as an apostrophe, but it may be a risen or broken sort.
4. `blocks[2].lines[17]` — the mark over the e of `auroyẽt` is a short slanted bar; read
   as a tilde, though it could be taken for an acute (the tildes in `ayãt`, `mõde`,
   `deuãt` on this page are flatter).
5. `blocks[2].lines[6]` — printed tight as `qu'ileſtoit`; word division normalized to
   `qu'il eſtoit` per conventions §1. The apostrophe after `qu` is a raised mark.
6. `blocks[4].lines[1]` — validator long-s warning checked against the image and
   rejected: `IL N'EST` is set in spaced SMALL CAPITALS, so the S is a capital/round S,
   not long s. Do not "correct" it to `eſt`.

## Notes for the reconciler

- Punctuation checked at 3x–9x on every clause boundary. Confirmed colons (not commas)
  at `de mort:`, `en argent:`, `d'Artigat:`, `le laiſſer:`, `veriſimilitude:`,
  `d'autruy:`; confirmed full stops at `iniuſtement.`, `& defendeur.`, `de crime.`
- Every double s on the page was zoomed: `poſſedé`, `laiſſer`, `aſſouui`, `raſſaſié`,
  `aſſem`, `richeſſes` are all `ſſ` (long s + long s, ligatured). No `ſs` occurs on this
  page. `toutesfois` has a round s (before f), `deffence` is `ff`.
- `calomnieuſement` opens with `c`, not `e` — the ink bleed makes the c look closed, but
  it has no crossbar (compare the `e` later in the same word).
- Lines with a word broken at the line end WITHOUT a hyphen (normal for this print, not
  flagged): `pour luy vo` / `ler`, `& in` / `uenté`, `car com` / `me`, `pour aſſem` /
  `bler`.
- The print sets a space before the comma in `richeſſes , &` on the last line; normalized
  per conventions §1.
- Paper condition is good; ink in the small annotation type is lighter and slightly
  spread at the right edge of the text block (last word of the final line, `rien`, sits
  right at the trim), but it is legible. The marker `{a}` is the only badly inked sort.
