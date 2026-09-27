# Read report — p033, reader B

Output: `transcription/reads/B/p033.json`
Validator: `uv run --with jsonschema python scripts/validate_page.py transcription/reads/B/p033.json` → `1 ok, 0 failed`, exit 0, no warnings.

## Counts

- Body (paragraph) lines: 31 — block 0 = 11, block 2 = 19, block 4 = 1
- Paragraphs: 3
- Headings: 2 (`ANNOTAT. XVII.`, `TEXTE.`)
- Markers in the body: 5 — `{c} {d} {e} {f} {g}`
- Margin notes: 7 — keys `a b c d e f g`
- Foot notes: 0 (no small-type foot block; the margin column simply runs down past the last body line)
- Ornaments: 0
- `uncertain[]` entries: 11

## Page furniture

- `running_head`: `PARLEMENT DE THOLOSE.` (spaced capitals, closed up)
- `folio`: `33` (matches the manifest)
- `signature`: `C` — a single blotted letter centred below the last body line
- `catchword`: none

## Layout

1. Large-type TEXTE paragraph continuing from the previous page (`continues_prev: true`), 11 lines, ending `temps repeu de belles paroles.`
2. Heading `ANNOTAT. XVII.`
3. Annotation paragraph, 19 lines of small type, self-contained on this page (does not continue onto p034). It contains the spaced-capital quotation `IE VEVX.` in line 8, so the block carries `spaced_caps: true`.
4. Heading `TEXTE.`
5. Large-type TEXTE paragraph, a single line `En fin fut contraint, le mectre en in-` (`continues_next: true`; the word `in-` breaks over the page).

Margin notes a–g run down the outer column. Notes a–e are set against annotation lines 1, 5, 8, 11 and 12; then the margin is empty for four lines before note f (beside line 17) and note g (beside line 19, whose last two lines drop below the body, level with the `TEXTE.` heading and the last line).

## `uncertain[]` entries (11)

1. `margin_notes[0]` — **note key `a` has no marker in the body.** The italic `a` on that row sits in the margin column, left-aligned with the note's other lines; there is no marker letter anywhere in annotation line 1. **escalate: true**
2. `margin_notes[1]` — **note key `b` has no marker in the body**, same situation beside annotation line 5. **escalate: true**
3. `margin_notes[0].lines[2]` `Cluentio. l. de` — the glyph after `Cluentio.` is a short flagged stroke, not the tall looped italic `l` used in notes b and c; it could be the numeral `1`. Read as `l.` on the sense (`l. debitores. C. de pignor.`).
4. `margin_notes[0].lines[0]` `Ciceron en` — a heavy ink mark sits over the final `n` of `en`; probably show-through or a broken sort, but `eñ` cannot be ruled out.
5. `margin_notes[2].lines[2]` `de ſeruit. vrb:` — the mark after `vrb` is two dots, the upper thin and irregular; read as a colon, could be a period plus a speck.
6. `margin_notes[3].lines[1]` `c xix.` — no period is visible after this `c` (contrast note g, where `c.` is clearly pointed), and none after `eccleſiaſtic` on the line above.
7. `margin_notes[4].lines[0]` `Tite iij. C.` — the final letter is a tall `C`; could be a large lowercase `c`, and the order `Tite iij. C.` is unusual for this book's citations.
8. `margin_notes[6].lines[2]` `c. xvij. Leuiti-` — the word-break sign after `Leuiti` is faint and partly lost in the paper; supplied as a hyphen because `que` opens the next line.
9. `blocks[1].text` `ANNOTAT. XVII.` — the two final `I`s of the numeral carry dots in the spaced-capital fount; transcribed as `XVII`.
10. `blocks[2].lines[2]` `reuiroit` — clearly printed r-e-u-i-r-o-i-t; sic, the sense would want `retiroit`.
11. `signature` `C` — the letter at the foot is blotted; read as `C`, which agrees with the gathering scheme visible in the finished pages (A = ff. 1–16, B = ff. 17–32, so C begins at f. 33).

## For the reconciler

- **The missing `a` and `b` markers are the one real problem on this page.** I checked annotation lines 1–5 at 2–4x across their full width and found no small italic letter in the body column. The `a` and `b` that a quick read might take for markers are the notes' own key letters: both are left-aligned with the lines under them in the margin column, exactly like the unambiguous `c`, `d`, `e`, `f`, `g` note keys further down. If reader A recorded `{a}` or `{b}` in the body, that spot needs a third look.
- The body strips keep a narrow slice of the margin column along their right edge, which is where those `a`/`b` key letters appear. Do not read them as body text.
- Double-s checked at 3x or more everywhere it occurs: `careſſé`, `reſſentans`, `menaſſer`, `offenſe`, `affaires`, `ſatisfaire` — all as transcribed, with `ſſ` (long s + long s) in the three double-s words, no `ſs`.
- `toutesfois` (block 0 line 1) has a round `s` before the `f`, as the conventions predict.
- Clause punctuation checked glyph by glyph: `nepueu:` and `abſence:` are colons; `ſouuent.`, `places.`, `iniuſte {c}.`, `mes {e}.`, `ſeul {g}.` are periods; `menaſſer {d},` and `frere {f},` are commas (both have a tail below the baseline).
- Spacing normalizations applied where the compositor set tight: `nepueu: iuſqu'à`, `deux, & trois`, `ſouuent. Ainſi`, `ciuilité: mais`, `ſi nous receuons`, `contraint, le mectre`, and in the margin `l. ſi`, `D. de pi`, `gno. actio.`, `l. quidam`, `Cluentio. l. de`.
- The page is clean: no damage, no faint ink beyond the margin-note details noted above, and the gutter slice of the facing page (left edge, this being a recto) was ignored.
