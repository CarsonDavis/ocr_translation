# read-p046-B

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/B/p046.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/B/p046.json` → `1 ok, 0 failed`, exit 0, no warnings.

## Counts

| | |
|---|---|
| body lines | 30 (9 + 9 + 12) |
| paragraphs | 3 |
| headings | 2 (`ANNOTAT. XXV.`, `TEXTE.`, both spaced caps) |
| markers in body | 3 (`{a}`, `{b}`, `{c}`) |
| margin notes | 3 (keys a, b, c; 13 printed lines total) |
| foot notes | 0 |
| signature | none |
| catchword | none |
| ornaments | none |
| `uncertain[]` entries | 15, of which 3 are `escalate: true` |

## Page structure

Verso. Running head `ARREST DV` (spaced caps), folio `46` at the left, matching the manifest.

1. Paragraph (9 lines, large TEXTE type), `continues_prev: true` — continues the sentence running over from p045 (`… verificatiõ & recognoiſſãce dudit priſon | nier …`), ends `ſement, & honorablement veſcu.`
2. Heading `ANNOTAT. XXV.`
3. Paragraph (9 lines, small annotation type) carrying all three markers: `{a}` after `bien viuãt`, `{b}` after `prochain`, `{c}` after `reputé` on the last line.
4. Heading `TEXTE.`
5. Paragraph (12 lines, large type), `continues_next: true` — the page ends mid-sentence at `… qu'il y a ſi`.

`foot.jpg` shows only the last two body lines and then bare paper: no foot citation block, no signature, no catchword. `margin-1.jpg`, `margin-3.jpg` and `margin-4.jpg` are blank; the whole margin block sits in `margin-2.jpg`.

## `uncertain[]` entries (one line each)

1. **ESCALATE** `blocks[2].lines[5]` — `l'augmente`: the glyph before the apostrophe is an **l**, not a long ſ, although the sense wants `ſ'augmente`; verified at 13x–16x against the ſ of `ſon` and the l of `la` on the same printed line.
2. **ESCALATE** `blocks[4].lines[3]` — `Guerre. pour`: the mark reads as a period (round, on the baseline) against the sense, compared with the comma after `hanté` and the period after `enfance.`
3. **ESCALATE** `margin_notes[1]` — the key letters of the 2nd and 3rd margin entries are x-height round glyphs identical to the `c.` of `c. dudũ`, not an italic `b`; either keys b/c printed with a period, or the abbreviation `c.` with the keys unprinted.
4. `margin_notes[1]` — grouping of the 13 margin lines into 8 / 3 / 2; the break after `cauſ.` is certain, the one before `mandata de` is not (8 / 5 is an alternative).
5. `margin_notes[1]` — placement: the notes drift far below their markers; note b's first line stands beside the blank band at the `TEXTE.` heading, so its `beside_line` is null.
6. `blocks[2].lines[4]` — line-end `con-`: worn short stroke read as a hyphen because it sits at mid-x-height, not on the baseline; also `meſchancé` sic.
7. `blocks[0].lines[6]` — word division normalized per §1 (`toutle`→`tout le`, `deRols`→`de Rols`, `Rolsauoir`, `ily`, `désle`, `Berçeau:le`, `b:la`, `deremilit.`).
8. `blocks[4].lines[10]` — `desteſ` at the line end is `des` (round s) + `teſ` (long ſ), normalized to `des teſ`; no hyphen printed; `dés` sic for `dès`.
9. `blocks[0].lines[0]` — every double-s on the page checked at 4x–8x and is `ſſ` (`recognoiſſãce`, `paſſée`, `cognoiſſent`, `aſſeurent`, `recongnoiſſent`); no `ſs` occurs; all tildes listed and verified.
10. `blocks[0].lines[5]` — `Rols` is ordinary upper-and-lower case here, **not** the small capitals `ROLS` used on p044.
11. `blocks[2].lines[2]` — colons after `fraude`, `prochain {b}` and `Berçeau` verified as two dots at 9x; justification space before the comma in `nature , &` normalized away.
12. `blocks[2].lines[1]` — list of the places where the print sets punctuation tight or with a space before it, all normalized per §1.
13. `blocks[4].lines[6]` — no punctuation printed after `auoit` at the line end.
14. `margin_notes[0].lines[3]` — the opening `q` (and the `j.` on the line above) belong to the citation, not to the key apparatus.
15. `blocks[1]` — spaced capitals closed up with `spaced_caps: true` for the running head and both headings; `TEXTE.` carries a printed period.

## Notes for the reconciler

- **Paper and ink are good on this page.** No damage, no show-through worth recording, no gutter loss. The only genuinely hard readings are the three escalated ones, and all three are typographic rather than physical: a suspect sort (`l` for `ſ`), a period where the sense wants a comma, and the margin key letters.
- The **margin key question is the one that needs a human.** Entry a is keyed with an unmistakable italic `a` and no period. Entries 2 and 3 begin with a round x-height glyph plus a period that is, glyph for glyph, the same sort as the `c.` in `c. dudũ`. I measured it: both leading glyphs are ~28 px tall on `margin-2.jpg`, whereas the italic `b` of `barbaris` three lines below is ~42 px, so neither can be a `b`. I adopted keys a/b/c (three markers, three citation groups, and the validator's marker↔note rule) and treated the leading glyph+period as the key, but reading (ii) — `c.` = *capitulum*, keys unprinted, markers b and c noteless — fits note c (`c. mandata de præſum`, exactly parallel to `c. dudũ. de præſu.`) better than it fits note b (`c.` before `l. non omnes` is not idiomatic). If the reconciler prefers reading (ii), notes b and c gain a leading `c. ` and both keys become null.
- **`l'augmente` (blocks[2].lines[5]).** I am confident about the glyph and unconfident about the intent. The controls are on the same printed line, so the comparison is as clean as this page allows: the ſ of `ſon` hooks right at the top and ends in a left-only spur; the l of `la` has a top-left serif and a broad two-sided foot; the disputed glyph has the top-left serif and the broad two-sided foot. If reader A read `ſ'augmente`, that is a reading from sense, not from the glyph.
- Three body lines break a word with **no hyphen** (`perſonna|ge`, `en|uers`, `des teſ|moins`), which is normal for this print and is recorded without an `uncertain[]` entry except where word division also had to be normalized.
- `meſchancé` (for *meschanceté*) and `dés` (for *dès*) are printed as transcribed; not emended.
