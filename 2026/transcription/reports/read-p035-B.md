# read-p035-B

**Output path:** `/Users/cdavis/github/translator/2026/transcription/reads/B/p035.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/B/p035.json` → `1 ok, 0 failed`, exit 0, no warnings.

## Counts

| thing | count |
|---|---|
| body lines (paragraph lines, headings excluded) | 32 |
| paragraph blocks | 4 |
| heading blocks | 2 (`TEXTE.`, `ANNOT. XIX.`, both spaced caps) |
| markers in the body | 5 (`{b}`, `{a}`, `{b2}`, `{c}`, `{d}`) |
| margin notes | 5 (keys `b`, `a`, `b2`, `c`, `d`) |
| foot notes | 0 |
| uncertain[] entries | 11 |

Page furniture: running head `PARLEMENT DE THOLOSE.`, folio `35`, signature `C ij`, no catchword, no ornaments.

Block breakdown: paragraph (2 lines, `continues_prev`) → Latin quatrain from Horace (4 lines, italic, treated as a paragraph per §6) → heading `TEXTE.` → TEXTE paragraph (8 lines, large type) → heading `ANNOT. XIX.` → annotation paragraph (18 lines, `continues_next`). The page ends part-way down; the empty space below the last line gets no block.

## uncertain[] entries, one line each

1. `margin_notes[2].key` — **escalated.** The page carries two notes lettered `b`: the Horace note at the head (tail of the previous annotation's alphabet) and the second note of ANNOT. XIX's restarted `a b c d` run. Keyed the second `b2` per §4's a2/a3 rule so keys stay unique (a duplicate key is a hard validator failure); the print shows a plain `b` in both places. Reader A may have keyed this differently — the reconciler should settle it.
2. `margin_notes[0].lines[0]` "Horace au" — a heavy mark sits over the final `u`; read as plain `au`, could be a damaged sort or a tilde (`aũ`).
3. `margin_notes[0].lines[2]` "Carmes, Ode." — a stroke over the final `e` of `Ode`; read as plain `e`, possibly an acute (`Odé`) or ink offset.
4. `margin_notes[1].lines[2]` "quis. iij. q. viij" — the roman numerals are tiny and partly blotted at native resolution; `iij` could be `ij`, `viij` could be `vij`.
5. `margin_notes[1].lines[0]` "l. fin. C. de" — printed `l. fin .C. de` with the point set before the `C`; spacing normalized per §1.
6. `margin_notes[3].lines[0]` "c. ſi. & illec" — last word read `illec`; the canonical citation would be `illic` and the penultimate letter is ambiguous.
7. `blocks[5].lines[6]` — `Empereuts` sic: the letter before the final `s` is clearly a `t`, a misprint for `Empereurs`.
8. `blocks[5].lines[14]` — a faint speck over the first `e` of `intention`; read as plain `e` (unlike the solid tilde of `cõſort` on line 2 of the page), tilde not excluded.
9. `blocks[5].lines[4]` — the mark after marker `a` is a compact square dot on the baseline, read as a period; it matches the marks after `b2` and `d`.
10. `blocks[0].lines[0]` — marker `b` is set tight against `Horace` and inside the closing parenthesis (`Horaceb)`); spacing normalized per §4.
11. `blocks[3].lines[2]` — `l'accuſation du crime prodigieux & horri` ends without a hyphen (`horri` / `ble`), as printed.

## Notes for the reconciler

- **Paper and ink are clean.** No damage, no gutter loss, no faint passages. The only reading difficulties are the margin column's very small italic numerals (note `a`, line 3) and scattered foxing specks that mimic accents.
- **Marker alphabet straddles a section break on this page.** `{b}` (Horace) closes the previous annotation's run; ANNOT. XIX restarts at `{a}` and runs `a b c d`. That is the source of the `b`/`b2` collision above.
- **`ſſ` vs `ſs` checked at 3x or better on every double s.** Findings: `fauſſement`, `aſſez`, `paſſion` are all `ſſ` (long s + long s); `auſsi` (TEXTE, line 5) is the only `ſs` on the page (long s + round s). Reader A should be diffed carefully here.
- **Sentence punctuation checked glyph by glyph.** `impoſé {a}.`, `garantir {b2}.`, `paſſion {d}.` are all periods (square dot on the baseline); `crimes {c}:` is a colon; `endurer, ſil` and `griefuës,` are commas (visible tail below the baseline). `parlé. &` (TEXTE line 4) is a period, not a comma.
- **Margin notes drift upward relative to the body**, by up to one line as the column descends: notes `a` and `b2` start beside their own marker line, but note `c` starts beside `dignes de peines…` (one line above its marker) and note `d` beside `& deliberation…` (one line above its marker). `beside_line` records where each note's first line is actually printed, not where its marker is.
- **Two line-end word breaks have no hyphen**: `horri` / `ble` (TEXTE) — normal for this print, no entry needed beyond the note above. All other line-end breaks use the double-stroke hyphen, transcribed `-`.
- **`ſils` (twice, annotation lines 4 and 13) and `ceſt` (line 1) are printed without an apostrophe** — sic, left as printed.
- The page has no foot citation block and no catchword; `C ij` sits alone under the last line.
