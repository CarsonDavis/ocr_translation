# read-p057-A

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/A/p057.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/A/p057.json` → `1 ok, 0 failed` (exit 0, no warnings).

## Counts

| | |
|---|---|
| body lines (printed, excl. heading) | 37 (3 + 34) |
| paragraphs | 2 |
| headings | 1 (`ANNOT. XXXVI.`, spaced caps) |
| markers in body | 9 (`a b c d e f g h i`) |
| margin notes | 9 |
| foot notes | 0 |
| running head | `PARLEMENT DE THOLOSE.` (spaced caps, final point) |
| folio | `57` (matches manifest) |
| signature / catchword / ornaments | none |

## Structure

Page opens with the tail of the preceding paragraph (3 lines, `continues_prev: true`,
ending `mateur du nom de Dieu.`), then the display line `ANNOT. XXXVI.`, then a single
34-line paragraph that runs off the foot of the page mid-word/sentence
(`… ou mau`), so `continues_next: true`. `foot.jpg` holds no foot citation block, no
signature and no catchword — the page simply ends at the last body line.

## `uncertain[]` entries (12)

1. `margin_notes[0].lines[0]` "Iean d'Ana" — letters after `d'` blotted; read so the name
   runs `Iean d'Ananie` (Ioannes de Anania) across the line break. **escalate**
2. `margin_notes[0].lines[1]` "nie au fina." — `fina.` is what is printed; probably
   abbreviating *au final*; not expanded.
3. `margin_notes[1].lines[1]` "c de religioſ" — key placement ambiguity, see below.
   **escalate**
4. `margin_notes[3].lines[0]` "S. Iean. c. x." — the letter before `x.` is very faint;
   read `c` (chapitre) by parallel with `c. xxiiij.`; could be `e`.
5. `margin_notes[4].lines[1]` "c xiiij." — printed *xiiij* (14) where the sense wants
   Leviticus 24; as printed, sic.
6. `margin_notes[5].lines[1]` "guinis. cx ij." — printed `cx ij.`, presumably `c. xxij.`
   (Decretum C. xxij. q. v.); as printed, sic.
7. `margin_notes[6].lines[6]` "iiij-collat." — the mark between `iiij` and `collat.` is a
   short stroke at mid height; transcribed as a hyphen, may be a high-set period.
8. `margin_notes[7].lines[2]` "maieſt." — blotted; last two letters not cleanly separable.
9. `margin_notes[8]` (key `i`) — no key letter legibly printed before `Les interpre-`;
   only a faint dot at the key column. Keyed `i` because `{i}` is the only marker left
   without a note and the page's run is a–i. **escalate**
10. `blocks[2].lines[18]` — `Sauuenr` printed for *Sauueur* (wrong sort, n for u); sic,
    checked at high magnification.
11. `blocks[2].lines[21]` — the mark over the e of `anciéne` slants like an acute, unlike
    the flat tildes of `ledicẽce`, `condẽnent`, `pẽſe` elsewhere on this page; read `é`,
    but `anciẽne` is possible.
12. `blocks[2].lines[28]` — `facilemeot` printed for *facilement* (wrong sort, o for n):
    the glyph is a closed round `o`, plainly unlike the `n` of `vn` on the same line; sic.

## For the reconciler

* **The one real structural decision on this page.** Three consecutive margin lines begin
  at the key column with a single letter + space: `b Aut, alearũ`, `c de religioſ`,
  `c leuitique.`. Only one of the two `c`s can be the key (duplicate keys are invalid and
  there are exactly 9 markers). I took the second as a continuation of note b
  (`Aut. alearum, c. de religios.`) and the third as the key, because the two lines under
  it, `leuitique.` + `c. xxiiij.`, are one citation — Leviticus 24, the stoning of the
  blasphemer — which is what marker `{c}` (`les prouoquant d'opprobres & iniures`) wants,
  and because note `e` cites the same book (`Leuitique. / c xiiij.`). Continuation lines on
  this page demonstrably do start with a lowercase `c` + space (`c xiiij.` under note e),
  so the shape alone does not settle it. If B reads it the other way, note b is one line
  and note c is three.
* Margin leading is tighter than body leading (~67 px vs ~78 px at native scale), so notes
  drift upward relative to their markers as the column descends: note `a` sits exactly
  beside its marker line, but `f`–`i` sit three to five lines above theirs. `beside_line`
  values record where each note's first line actually prints, not where its marker is.
* Distinguishing `b` from `h` in the margin keys needs magnification: the `b` bowl closes
  on the stem at the baseline, the `h` leg does not. Both keys occur on this page and look
  alike at reading size.
* Two wrong sorts confirmed at 8–9× (`Sauuenr`, `facilemeot`). `qu'atribuer` (one t) and
  `S. Tomas` (no h) are as printed and need no entry per §1.
* Double-s checked at ≥3× throughout: `laiſſant`, `gliſſement`, `paſſion` are all `ſſ`;
  `toutesfois` is round s before f as expected.
* Ink is even and the impression clean; the only genuinely damaged/blotted readings are in
  note `a` line 1 and note `h` line 3. No tears, no show-through worth noting. The facing
  page's gutter strip on the left of the crops was ignored.
