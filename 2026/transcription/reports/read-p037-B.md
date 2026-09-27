# Read report — p037, reader B

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/B/p037.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/B/p037.json` → exit 0
(one non-blocking WARNING, "possible normalized long s (si)", checked against the image and correct: the
word is `Si` with a roman capital S opening the annotation.)

## Counts

| item | count |
|---|---|
| body lines (all blocks) | 34 |
| — TEXTE paragraph | 10 |
| — ANNOTAT. XX. paragraph | 24 |
| paragraphs | 2 |
| headings | 2 (`TEXTE.`, `ANNOTAT. XX.`, both spaced caps) |
| markers in body | 8 (`a b c d e f g h`) |
| margin notes | 8 (keys `a`–`h`, 31 note lines) |
| foot notes | 0 |
| ornaments | 0 |
| `uncertain[]` entries | 26 (2 flagged `escalate`) |

Page furniture: running head `PARLEMENT DE THOLOSE.`, folio `37` (matches the manifest),
signature `C iij`, no catchword, no foot citation block.

## Layout

Recto. Running head + folio, then the display line `TEXTE.`, then ten lines of the arrêt
text in large roman (continuing a sentence from p036 → `continues_prev: true`, ending
cleanly at `aux fins abſolutoires.`). Then the display line `ANNOTAT. XX.` and twenty-four
lines of Coras's annotation in the smaller roman; that paragraph breaks off mid-word at
`craint ſon au` → `continues_next: true`. The outer margin carries eight keyed citations,
`a` through `h`, in italic; several are long (note `d` runs six lines, notes `e` and `g`
six each), so the notes are set tighter than the body lines and drift ahead of their
markers — notes `c`, `d` and `e` begin one to five body lines *above* the line holding
their marker. `beside_line` records the line each note's first line is actually printed
beside. Marker/note cross-check: every `{x}` in the body has a note with key `x` and vice
versa, no gaps.

## `uncertain[]` entries (26)

1. `blocks[1].lines[0]` — sic `ſes femme` (plural determiner, singular noun); `fuſſent` verified ſſ at 7x.
2. `blocks[1].lines[1]` — a faint comma-shaped mark at cap height before `accarez`; kept as `'` but much lighter than the solid apostrophes on the same line, could be offset ink or a stray sort.
3. `blocks[1].lines[4]` — mark between `puiſſance` and `dudit` read as a PERIOD (small compact dot on the baseline, no tail, unlike the comma after `bien` three lines below); space before it normalized.
4. `blocks[1].lines[7]` — `debien` set tight, normalized; the mark after `bien` is a comma (clear tail below baseline).
5. `blocks[1].lines[2]` — `de bien` set tight; line ends `recog` with no hyphen.
6. `blocks[3].lines[2]` — `puiſſance` and `poſſeſſeur` both verified ſſ at 12x.
7. `blocks[3].lines[3]` — `b.parce` set tight, normalized; the stop is a period.
8. `blocks[3].lines[5]` — `poſſeſſion` verified ſſ at 12x; `la poſſeſſion` set tight.
9. `blocks[3].lines[10]` — `parẽs`: the mark over the e is a thick horizontal wavy bar (tilde), not the thin acute of `reintegré`; `auſſi` verified ſſ, not `auſsi`.
10. `blocks[3].lines[16]` — `difficulté ,que` printed with a space before the comma, normalized; line ends `ſoup` unhyphenated.
11. `blocks[3].lines[19]` — mark after `femme` verified as a colon; line-end break sign after `cõpa` is a raised blob, transcribed `-`.
12. `blocks[3].lines[22]` — a distinctly inked raised comma-shaped sort between `exemple` and the comma; transcribed as an apostrophe, purpose unclear.
13. `blocks[3].lines[23]` — page ends mid-word (`au`, no hyphen, no catchword); continues on p038.
14. `blocks[3].lines[18]` — `puiſſe` verified ſſ.
15. **ESCALATE** `margin_notes[0].lines[0]` — the p-with-stroke in `l. e. C. de ꝓ-`: at 18x the stroke reads as a bar through the descender (= `ꝑ`), but the continuation `hib.` requires *pro*, so transcribed `ꝓ`.
16. `margin_notes[0].lines[1]` — `hib.` not `bib.`: open arch, unlike the closed bowls of the neighbouring b's; citation resolves to C. *de prohibita sequestratione pecuniae*.
17. `margin_notes[2].lines[0]` — final word `lis` vs `lia`; the glyph matches the italic round s of `domus`, and `lis` + `pend.` (lite pendente) is coherent. Wide gap normalized.
18. `margin_notes[3].lines[0]` — `l` with no period here (contrast `l.licet`, `l.ſi`); `ſi` has no crossbar, so long s not f.
19. `margin_notes[3].lines[1]` — `confi` with an fi ligature (clear crossbar at 12x), continuing `tetur` = *confitetur*; line unhyphenated.
20. `margin_notes[3].lines[3]` — `poſſeſ-` verified ſſ + final long ſ.
21. `margin_notes[4].lines[0]` — `aquiſſi-` verified ſſ (both tall italic long s with descenders), not ſs.
22. `margin_notes[6].lines[2]` — the isolated tall sort after `app.` read as `l`; could be a thin-bowled `b`.
23. `margin_notes[7].lines[0]` — `ijs` printed with two dots over the i/j pair; no period printed after the `P`.
24. `margin_notes[7].lines[1]` — the slanted mark above `ro` is the descender of the long ſ of `ſi` on the line above, NOT an accent; transcribed `ro`.
25. **ESCALATE** `margin_notes[7].lines[2]` — over-inked blot ~3 sorts wide before the break sign; read `ex-` because the next line is `hib.` (D. *de liberis exhibendis*), but it could be transcribed `exx` or `exe`.
26. `blocks[3].lines[0]` — the validator's long-s warning: `Si` opens the annotation with a roman capital S, so the short s is correct.

(Entries 1–25 plus the long-s note = 26 objects in `uncertain[]`; the two `escalate: true`
flags are items 15 and 25.)

## Notes for the reconciler

- **The delivered `pages/strips/p037/margin-*.jpg` were originally cropped too far outward**
  and cut off the note key letters at the left edge; I re-cut the margin column myself from
  `raw/img059.jpg` (the raw scan for this page — note `raw/img037.jpg` is a *different*
  leaf) to read the keys. The strips were later regenerated with wider inward padding and
  now show keys `a`–`h` clearly; they confirm every reading above.
- `raw/img059.jpg` (2941×4711) is slightly higher resolution than `pages/full/p037.jpg`
  (2780×3966) and was used for all the close checks.
- **Ink quality.** The body is clean and well inked. The margin italic is lighter and the
  bottom of the margin column (end of note `h`) is over-inked: the abbreviation before
  `hib.` is a solid blot. Note `a`'s abbreviation stroke is likewise hard to resolve. These
  are the two escalations.
- **Two stray-looking raised commas** — one before `accarez` (TEXTE line 2), one after
  `exemple` (annotation line 23). The second is clearly type; the first is faint enough
  that reader A may well have omitted it. Both are recorded in `uncertain[]`.
- **Every double-s on the page was checked at ≥3x** per the prompt's error class (a):
  `fuſſent`, `ſ'aſſeurant`, `puiſſance` (×2), `poſſeſſeur`, `poſſeſſion`, `cognoiſſance`,
  `auſſi`, `puiſſe`, and in the margin `poſſeſ-`/`ſionum` and `aquiſſi-`. **All are ſſ on
  this page; none is the ſs form.** `ſuffiſante` is ff + single ſ.
- **Sentence punctuation** was checked glyph-by-glyph (error class b). The only one worth
  flagging is `puiſſance. dudit` (TEXTE line 5), a mid-sentence period; the colons after
  `accarez`, `noiſtront`, `biens {a}`, `prohibee`, `pres`, `l'immeuble {g}` and `femme` all
  show two dots at zoom.
- **Tildes:** `parẽs` (annotation line 11) and `cõpa-` (line 20). Both bars are thick and
  horizontal, unlike the thin acutes on `reintegré`, `poureté`, `és`, `difficulté`.
- The annotation number is `ANNOTAT. XX.` — the marker alphabet restarts at `a` for this
  section, so this page's markers run a–h independently of p036.
