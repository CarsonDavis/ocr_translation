# read-p030-A

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/A/p030.json`
(reader A, model opus). Validator: `uv run --with jsonschema python scripts/validate_page.py
transcription/reads/A/p030.json` → **exit 0** ("1 ok, 0 failed"), 2 warnings, both checked
against the image and correct as transcribed (see "Validator warnings" below).

## Counts

| item | count |
|---|---|
| body lines (paragraph blocks) | 33 |
| display/heading lines | 2 (`TEXTE.`, `ANNOT XV.`) |
| total printed lines in the body column | 35 |
| blocks | 7 |
| paragraphs | 5 (incl. the Latin distich, which is a paragraph per §6) |
| headings | 2 |
| markers in the body | 9 — `{b} {c} {d} {e} {f} {g} {h} {i}` then `{a}` (alphabet restarts at ANNOT XV) |
| margin notes | 9 — keys `b c d e f g h i a`, every one matched to a marker |
| foot notes | 0 |
| ornaments | 0 |
| `uncertain[]` entries | 18 |

`running_head` `"ARREST DV"`, `folio` `"30"` (matches the manifest), `signature` null,
`catchword` null. `foot.jpg` shows nothing below the last body line.

## Page structure

1. paragraph, `continues_prev: true` (17 lines) — Pygmalion / Polymnestor / Eriphyle,
   ending `oit bien en diſant {f},`
2. paragraph — the Virgil distich, 2 lines (`Quid non mortalia pectora cogis, / Auri ſacra fames?`)
3. paragraph, 7 lines — `C'eſt pourquoy l'empereur M. Antonin …`
4. heading `TEXTE.` (spaced caps)
5. paragraph, 4 lines — the Toile text, large type, ending `nee ladite de ROLS à le pourſuyure.`
6. heading `ANNOT XV.` (spaced caps)
7. paragraph, 3 lines, `continues_next: true` — start of Annotation XV

## `uncertain[]` — one line each

1. `blocks[0].lines[0]` — **`faluſt-il`**: the sign between the two words is a single round dot
   set at mid x-height, higher than every period on the page but not the slanted double stroke
   used at line ends; read as a hyphen, could be read as a period. **escalate**
2. `blocks[0].lines[1]` — `des hommes` printed tight as `deshommes`; normalized per §1.
3. `blocks[0].lines[4]` — `ſa ſeut,` sic (final letter is a t, not r; sense wants `ſeur`).
4. `blocks[0].lines[13]` — `qu's'eſtoit` sic (qu + ' + round s + ' + eſtoit).
5. `blocks[0].lines[14]` — `poinr` sic (final r, not t); also a space printed before the comma
   in `Thebes ,de`, normalized.
6. `blocks[0].lines[11]` — apostrophe gaps (`qu' Adraſtus`, `d' eſtre`) closed up per house practice.
7. `blocks[0].lines[15]` — the raised italic `e` after `predit` read as marker `{e}`, not as the
   last letter of `predite`.
8. `blocks[2].lines[1]` (and `.lines[6]`) — **`vertuſ`**: word-final long ſ in `vertus`, twice;
   unusual, confirmed at 9x in both places. **escalate**
9. `blocks[2].lines[5]` — a stray baseline dot before the comma in `Saluſte,` (broken sort);
   not transcribed.
10. `blocks[4].lines[0]` — `Martin` set with a wide justification gap (`Mart in`); one word.
11. `blocks[4].lines[3]` — `ROLS` is small capitals (as on p044).
12. `blocks[5]` — heading is `ANNOT XV.` with **no period after ANNOT** (checked at 12x).
13. `margin_notes[0]` — whether the `c.` ending `Rethorique c.` belongs to note b or is the key
    of a separate note; kept in note b. **escalate**
14. `margin_notes[0].lines[4]` — the ij/ÿ numeral form, transcribed `ij` (also in notes d, f, i).
15. `margin_notes[1].key` — the key glyph is blotted (c/e); read `c` from sequence, placement
    and sense (Aeneid I = Pygmalion/Sichæus).
16. `margin_notes[3].lines[4]` — `Aeneides,` final punctuation faint/broken, comma vs period.
17. `margin_notes[7].lines[1]` — `vt Iude`: capital I vs lowercase l indistinguishable in italic.
18. `margin_notes[2].lines[1]` — tight word-settings in the margin (`auxParalelles.`, `panor.en la`,
    `enla vie`, `moth.c.ix,`, `Parag.j.ſur`) normalized per §1.

## Validator warnings (both checked, no change)

- `blocks[0].lines[15]` "Sur" and `blocks[6].lines[0]` "Si" — both are **capital S** at the start
  of a sentence (`Sur quoy`, `Si les loix`); capitals in this fount are never long s. Verified on
  `body-3.jpg` and `body-7.jpg`.

## For the reconciler

- **Margin note b / note c split** is the one structural judgement on this page: the margin sets
  `b Ciceron au / b. liure de ſa / Rethorique c. / panor. en la x- / xxvij. diſtin- / ction.` as a
  continuous block. I read the `c.` as text of note b (second citation), because the note that is
  actually keyed `c` (`Vergile au / j. des Aenei- / des.`) starts three margin lines lower, exactly
  beside the body line carrying `{c}`, and Aeneid I is the reference the marker needs.
- **Two `ſſ` vs `ſs` checks** made explicitly: `auſſi` (line 7) is a true ſſ ligature (both strokes
  full height, 7x); there is no `ſs` word on this page.
- **`vertuſ` twice** and **`faluſt-il`** are the two readings most likely to differ from reader B.
- Note d spans body lines 8–13 in the margin and note i runs to the foot of the annotation; margin
  notes drift **upward** relative to their markers on this page (the margin leading, ~22px, is
  tighter than the body leading, ~39px), so `beside_line` is placement only and does not always
  equal the marker's line.
- Condition: the leaf is clean; a diagonal crease crosses the top-left corner of the margin column
  (above note b, no text lost) and there is mild show-through from the recto in the lower half.
  Faintest ink on the page is the punctuation after `Aeneides` in note e and the key letter of
  note c.
- The gutter strip of the facing page (right edge of every crop, and the right third of `foot.jpg`)
  was ignored throughout.
