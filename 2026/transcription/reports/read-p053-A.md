# read p053 — reader A

## Output

`transcription/reads/A/p053.json`
Validator: `uv run --with jsonschema python scripts/validate_page.py transcription/reads/A/p053.json`
→ `1 ok, 0 failed`, exit 0, no warnings.

## Counts

| item | count |
|---|---|
| body lines (all paragraph blocks) | 32 |
| paragraph blocks | 4 (13 + 1 + 5 + 13 lines) |
| headings | 1 (`TEXTE.`, spaced capitals) |
| markers in the body | 9 — `b c d e f g h i k` |
| margin notes | 9 — `b c d e f g h i k` (24 note lines) |
| foot notes | 0 |
| ornaments | 0 |
| `uncertain[]` entries | 13 (1 escalated) |

Page furniture: running head `PARLEMENT DE THOLOSE.` (spaced capitals, closed up),
folio `53` (matches the manifest), signature `D iij`, no catchword, no foot citation block.

The signature fits the run already established in `transcription/final/`:
p049 `D`, p053 `D iij`, p055 `D iiij`.

## Layout

Top of the page is the tail of an annotation in the small roman fount (13 lines, continuing
from p052), then an italic Latin verse line set off with space above and below, then five
more lines of the same annotation ending `falſifié le ſeau du prince {k}.` Then the heading
`TEXTE.` in spaced capitals, then 13 lines of the large text fount beginning
`Quatriemement, le cordonnier qui` and breaking off at `A l'au-`.

Block decisions:
- blocks[0] `continues_prev: true` (the page opens mid-word/mid-sentence: `ces & marques …`).
- The verse is its own `paragraph` block per §6, not a heading. blocks[0] `continues_next`
  is left **false** and blocks[2] `continues_prev` **false**, following the dominant pattern
  in `transcription/final/` for prose → verse → prose (p009, p019, p021, p030). blocks[0]
  ends on a colon that introduces the verse, so a reconciler could defensibly flip both to
  true; it changes no text.
- blocks[4] `continues_next: true` (page ends on the hyphen of `A l'au-`).

Marker alphabet: the page runs `b`…`k` with no `j` and no restart, so p052 should end on `a`.

## `uncertain[]` — one line each

1. `blocks[0].lines[7]` — `fuſt-il`: the break sign is a low baseline wedge, not the usual
   mid-height stroke; read as a hyphen (cf. `faluſt-il`, p030). Same form in `diſent-ils`.
2. `blocks[0].lines[8]` — `gtandement` **sic** (clear t, not r) for *grandement*.
3. `blocks[0].lines[11]` — `Lictance` **sic**: unambiguously dotted i; margin note `f`
   spells the same author `Lactance`.
4. `blocks[0].lines[12]` — the print sets `&,ſimulachre`, a comma tight against the
   ampersand (sic); the §1 spacing rule mechanically yields `& , ſimulachre`, which is what
   is written (cf. `P. proinde & .` in p159).
5. `blocks[1].lines[0]` — the `-que` sign after `Exẽplúmq` is an ordinary **semicolon**, not
   the 3-shaped `ꝫ` of §2; transcribed `;`.
6. `blocks[2].lines[1]` — `impreſſion` badly under-inked from the double s to the o; `ſſ`
   read at 10x with levels stretched, but `ſſ` vs `ſs` is not fully certain.
7. `blocks[4].lines[1]` — `qu'ice-`: a round raised dot, not the comma-shaped apostrophe of
   `qu'à` two lines below, and the i has no separate dot; read as the apostrophe.
8. `margin_notes[0].lines[0]` — a mid-height stroke follows the complete word `ſtigmata` at
   the line end; transcribed `-` though nothing is broken.
9. `margin_notes[0].lines[1]` — **escalated**: an italic `l` is printed *raised* right after
   the `C`; §2 gives superscripts as plain letters, so `Cl.` is written. The citation is
   Codex XI *de fabricensibus*, where a plain `C.` is expected, so the raised sort may be a
   stray or a compositor's error.
10. `margin_notes[0].lines[2]` — `ſib` read as long ſ by the word (`fabricẽ-ſib.`); in this
    italic fount `ſ` and `f` are near-identical.
11. `margin_notes[3].lines[1]` — `metallnm.` **sic** (turned u); note `i` on the same page
    sets the same word correctly as `metallum`.
12. `margin_notes[5].lines[0]` — `Geneſee. j.`: two round sorts after the long ſ, both read
    as e; the second could be a c (`Geneſec. j.` = Genèse chap. j.). Counters filled.
13. `margin_notes[6].lines[1]` — `l'Aſtio.` ends the line with a round baseline **point**
    although the word runs on into `nomie.`; not a hyphen (note `d` ends a line with an
    unmistakable dash, so the two shapes are distinguishable here).

## For the reconciler

- **Ink**: the page is clean and well inked except for one bad patch — the second half of
  `impreſſion` on `blocks[2].lines[1]`, where the type barely took. Nothing else on the page
  is faint, and there is no damage, stain or show-through worth recording.
- **Misprints kept as printed**, none of them repairs: `autre parties` (blocks[0].lines[0]),
  `gtandement`, `Lictance`, `&,ſimulachre`, `metallnm.` and possibly `Geneſee.`. This page
  is unusually rich in compositor slips; a reader who "fixed" any of them is wrong.
- `ſſ` was checked at ≥8x on every double s: `chauſſoit` (×2), `chauſſe`, `impreſſion` are
  all the two-ascender `ſſ`; `toutesfois`, `deſquels`, `trois teſ-` are round s as printed.
- The only structural judgment call is the `continues_next` / `continues_prev` pair around
  the verse block (see **Layout**).
- Note `h` starts beside the blank leading above the verse line; its `beside_line` is set to
  the verse line, which is where its marker sits.
