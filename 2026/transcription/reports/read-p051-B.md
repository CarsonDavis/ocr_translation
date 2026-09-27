# Read report — p051, reader B

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/B/p051.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/B/p051.json` → `1 ok, 0 failed`, exit 0, no warnings.

## Counts

| | |
|---|---|
| body lines | 32 (9 + 8 + 15) |
| paragraphs | 3 |
| headings | 3 (`ANNOTAT. XXIX.`, `TEXTE.`, `ANNOTAT. XXX.`) |
| markers in body | 4 (`{a}`, `{b}`, `{s}`, `{a2}`) |
| margin notes | 4 (keys `a`, `b`, `c`, `a2`) |
| foot notes | 0 |
| uncertain entries | 7 |

Page furniture: running head `PARLEMENT DE TOLOSE.`, folio `51`, signature `D ij`, no catchword, no ornaments.

## Layout

Recto. Running head + folio, then `ANNOTAT. XXIX.` (9-line paragraph, ends mid-page),
then the display line `TEXTE.` with the large-type text block (8 lines), then
`ANNOTAT. XXX.` whose paragraph runs to the foot and continues onto the next page
(`continues_next: true`; it breaks after `Alexander Aphrodiſee`). Nothing continues from
the previous page — the page opens on a heading. Margin column carries four notes:
three (a, b, c) beside the ANNOT. XXIX paragraph, one (a2) beside the ANNOT. XXX
paragraph; the alphabet restarts at `a` at the ANNOT. XXX boundary, hence key `a2`.
Foot strip holds only the signature `D ij` — no foot citation block.

`spaced_caps` set on all three headings and on the ANNOT. XXX paragraph, which contains
the letterspaced `L A-` / `C R I M A` (transcribed closed up as `LA-` and `CRIMA`).

## uncertain[] entries (7)

1. `blocks[1].lines[8]` — **escalated.** Third body marker is printed as a small **round
   superscript s**, not the tall long-s marker the book uses for key `ſ`, and not a `c`.
   By position it should be `{c}` (the margin note keyed `c`, 1 Timothy, sits beside it).
   Transcribed as printed (`{s}`); looks like a wrong sort.
2. `margin_notes[2]` — **escalated.** Same problem from the note's side: the note's key
   letter is a clear italic `c` but no `{c}` exists in the body.
3. `margin_notes[3].lines[3]` — two battered italic sorts between `dai` and `s` are
   unreadable at native resolution; written `dai[??]s c. ij.`. Line 3 ends `Iu-`, so the
   word is almost certainly `Iudaiques`; the pair is probably `qu` (with a faint `e`) or
   the abbreviation `qꝫ` / `q̃`. A stray dot sits between the pair and the `s`, and a
   wedge-shaped mark sits below the space before them.
4. `blocks[5].lines[10]` — the `e` of `yeux` is barely inked (reads almost as `yux`);
   reading secure from the surrounding letters and sense.
5. `margin_notes[1].lines[1]` — a very faint speck follows the `v` of `c. v`; possibly a
   period, possibly a paper fibre. Left off.
6. `running_head` — the final mark is a dot on the baseline with no tail, read as a
   period; a separate ink speck sits high above it. Also note the printer's `TOLOSE`
   (not `THOLOSE`), transcribed as printed.
7. `blocks[1].lines[6]` — `conſeque` is what is printed (c-o-n-ſ-e-q-u-e, checked at 8x)
   where the sense wants `conſerue`; sic.

## Notes for the reconciler

- **`ſſ` vs `ſs`** was checked glyph by glyph at 7–16x. Long s + **round** s: `auſsi`
  (line 4 of ANNOT. XXIX), `exceſsiue` (ANNOT. XXX line 2). True `ſſ` ligature:
  `congnoiſſe`, `triſteſſe` (twice), `engoiſſe`, `preſſe`. `quelquesfois` and
  `toutesfois` use a round `s` before `f`, as the print requires.
- Punctuation was read from glyph shape, not sense. Confirmed colons (two dots):
  `chair:`, `la cour:`, `raiſon:`, `trouue:`, `dehors:`, `d'or {a2}:`, `CRIMA:`,
  `Larme:`, `mer:`. Confirmed commas (tail below baseline): `grande,`, `aduient,`,
  `malancholie,`, `l'eſprit,`. Confirmed periods: `ſang.`, `amerement.`, `Dont` (after
  `{b}.`), `prouoquees.`.
- Tildes kept, not expanded: `incontinẽt`, `d'ẽnuy`, `tellemẽt`, `rõpement`.
- Line 7 of the ANNOT. XXX paragraph ends `affli` and line 8 begins `ge` — a word broken
  **without** a hyphen, which the conventions say is normal here, so no entry was made.
- `xij` in `liure xij. des` is set with the italic `ij` whose dots read almost as a
  dieresis (`xÿ`); read as `xij` (Josephus, *Antiquities* book XII, the Ptolemy
  Philadelphus episode), consistent with `c. ij.` later in the same note.
- Margin hyphens (`Ti-`, `Iu-`) are printed as a small raised stroke; transcribed as a
  plain `-` per §1.
- No damage, no gutter loss. Ink is even except for the faint `e` in `yeux` and the
  battered sorts in the last margin note. The facing-page show-through along the left
  (gutter) edge of the crops was ignored.
