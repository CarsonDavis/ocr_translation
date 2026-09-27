# read-p048-B

Page `p048` (manifest: page 48, image 70, verso, printed folio 58, source cudl).
Reader B, model opus.

## Output

- `transcription/reads/B/p048.json`
- Validator: `uv run --with jsonschema python scripts/validate_page.py transcription/reads/B/p048.json`
  exits **0** ("1 ok, 0 failed") with one warning, checked against the image and recorded in
  `uncertain[]`: the long-s warning on `Sur` in `blocks[2].lines[9]` is a roman **capital** S
  (capitals never take long s in this fount), not a normalized `ſ`.

## Counts

| item | count |
|---|---|
| body lines | 32 (13 + 19) |
| paragraphs | 2 |
| headings | 1 (`ANNOTAT. XXVI.`, spaced capitals) |
| markers in body | 2 (`{a}`, `{b}`) |
| margin notes | 2 (keys `a`, `b`) |
| foot notes | 0 |
| ornaments | 0 |
| signature / catchword | none / none |
| `uncertain[]` entries | 18 |

## Layout

- Running head `ARREST DV` (spaced capitals), folio `58` at the outer (left) edge — verso,
  so the margin column is on the left and the facing-page strip is along the right/gutter edge.
- `blocks[0]`: large-type TEXTE paragraph, 13 lines, continuing the sentence from p047
  (`continues_prev: true`), ending `l'autre.` (`continues_next: false`).
- `blocks[1]`: heading `ANNOTAT. XXVI.` in spaced capitals.
- `blocks[2]`: the annotation, 19 lines of smaller type, a fresh paragraph, ending the page
  with a full stop after marker `{b}` (`continues_next: false`). Flagged `spaced_caps: true`
  because of `TV DOIS` in line 12.
- Marker alphabet restarts at `a` with this annotation: `{a}` in line 17, `{b}` in line 19;
  both have notes. Note `a` stands beside the line holding marker `a`; note `b` stands beside
  the following body line, one line above its own marker.
- `foot.jpg` is blank below the last body line and `body-8.jpg` is entirely blank: no foot
  citation block, no signature, no catchword. `margin-1.jpg` and `margin-2.jpg` are blank —
  the only margin matter is the two notes at the very bottom of the column.
- Paper and ink are good; no damage, no show-through worth noting beyond normal bleed.

## `uncertain[]` entries (18)

1. `blocks[0].lines[10]` — a round baseline dot is printed after `n'o` at the line end although
   the word runs on as `n'o|ſans`; read as a period (sic), not a hyphen and not a comma.
2. `blocks[0].lines[8]` — `lereſte` set as one word; normalized to `le reſte` per §1.
3. `blocks[0].lines[2]` — `iuge.Mais` / `quoy?ces` set with no space after the point and the
   question mark; normalized per §1. The mark after `iuge` is a period.
4. `blocks[0].lines[3]` — justification space before the comma (`faites , les`); also at
   `blocks[2].lines[2]` and `blocks[2].lines[6]`; normalized away.
5. `blocks[2].lines[0]` — a small tilde-shaped tick sits above the final `t` of `ſingulierement`
   at the right margin; not transcribed (the word needs no nasal); probably a stray inked sort.
6. `blocks[2].lines[17]` — last word `mot`: the `o` is under-inked on its right side and can read
   as `c`; the same stray tick sits over the final `t`.
7. `blocks[2].lines[18]` — the print sets `d'a uantage` with a space; closed up per §1.
8. `blocks[2].lines[18]` — punctuation after marker `b`: a baseline round dot with a much smaller
   wedge-shaped tick above it; read as a period (the page's colons are two equal round dots set
   further apart); a badly inked colon cannot be completely excluded.
9. `blocks[2].lines[12]` — `TV DOIS` spaced capitals closed up per §3; the mark after `DOIS` is a
   comma (descending tail).
10. `blocks[2].lines[13]` — `q̃lle` = `quelle`: q + U+0303, not expanded; line ends `qua` with no
    hyphen.
11. `blocks[2].lines[4]` — the nasal marks on this page (`affectiõ`, `plemẽt`, `pourpẽſees`,
    `chãcelé`) are straight bars rather than wavy tildes; transcribed as precomposed tilde
    vowels per §2. `credit:ou` set tight; normalized.
12. `margin_notes[0].lines[0]` — the citation `l. iij. P. ij. D.`: glyph-by-glyph reading at 9x/12x
    (`l` · `iij` · `P` · `ij` · `D`, each followed by a point). `P.` is an italic capital P where a
    paragraph sign would be expected; transcribed as printed, not expanded. Points set tight;
    spacing normalized per §1.
13. `margin_notes[1].lines[0]` — a small hook-shaped mark stands above the final `o` of `anno`;
    the word runs on to `tation` (`l'annotation`), so no nasal is possible. **Not** transcribed as
    a tilde; reader A may read `õ`.
14. `margin_notes[1].lines[1]` — `lXXiij`: the two X glyphs are cap-height (taller than `a o n` of
    `tation` on the same line), so transcribed as capitals per §3, with a lowercase initial `l` and
    the italic `ij` pair at the end (= lxxiij, 73). Reader A is likely to write `lxxiij`; the
    difference is letter-case only.
15. `margin_notes[0]` — note placement, and the confirmation that there is no foot block, no
    signature and no catchword.
16. `blocks[2].lines[16]` — marker `a` printed hard against `ſemblables` with no space; spacing
    normalized per §4.
17. `blocks[2].lines[1]` — line ends `de pe` with no hyphen (→ `perils`); likewise
    `c'eſt`/`oit` and `qua`/`lité`. Also records the 6x checks on `ils ſont:` (colon) and `ſ'ils`
    (apostrophe after the long s).
18. `blocks[2].lines[9]` — the validator's long-s warning on `Sur`, checked: roman capital S.

## For the reconciler

- **Word-division normalizations are the likeliest A/B divergences on this page.** I applied §1 in
  both directions: two words set tight were separated (`lereſte` → `le reſte`) and one word set
  with a space inside was closed up (`d'a uantage` → `d'auantage`). Reader A may well have kept
  the printed spacing in the second case. `doutoyent` in `blocks[0].lines[10]` looks spaced at
  reading size but the u–t gap is ordinary letter spacing at 5x, so it is one word with no
  judgement involved.
- **Letter-case in `margin_notes[1].lines[1]`** (`lXXiij` vs `lxxiij`) is a pure small-capital
  judgement; the evidence is the glyph height against `tation` on the same line.
- **`margin_notes[1].lines[0]`**: whether the mark over `anno` is a tilde. Sense rules it out
  (`l'anno|tation`), so I left it off, but a reader following the ink alone would write `annõ`.
- **`blocks[0].lines[10]`** (`n'o.`) is the oddest thing on the page: a period printed inside a word
  broken across lines. It is a dot on the baseline, not a hyphen; kept as printed, sic.
- **`blocks[2].lines[18]`**: the period vs colon after marker `b` is the only punctuation call I
  would call genuinely close.
- The two stray ticks over the final `t` of `ſingulierement` and `mot`, both at the right margin
  of the annotation block, look like the same defect; worth a glance from a human if the two
  readers disagree about them.
- `l. iij. P. ij. D. de teſt.` — the `P.` is unusual for a Digest citation (one would expect `§`);
  it is what is on the page.
- Note for the record: the printed folio is `58` on the 48th page of the edition, which matches
  the manifest's expected folio, so no discrepancy was raised.
