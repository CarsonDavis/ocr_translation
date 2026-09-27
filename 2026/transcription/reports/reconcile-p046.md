# Reconcile p046

Output: `transcription/final/p046.json` — `normalize_spacing.py`: 0 lines changed; `validate_page.py`: 1 ok, 0 failed (exit 0).

## Differences decided: 5 (A 3 / B 1 / neither 0; plus 2 agreed-but-escalated lines confirmed as read)

| where | A | B | chose | reason |
|---|---|---|---|---|
| blocks[2].lines[5] | `ſ'augmente` | `l'augmente` | A | glyph: the stem's top curls RIGHT like the ſ of `ſon` on the same line and has no left wedge serif, which every `l` on the line carries; the two-sided foot is not decisive |
| margin_notes[1] (structure) | one note, key null, `c. l. non omnes` … | notes b (3 lines) and c (2 lines), leading `c.` taken as keys | A | the two opening glyphs are x-height crescents + period, identical to the `c.` of `c. dudũ`; the italic b of `barbaris` is full-ascender; the page's one printed key `a` has no period and the column has no hanging key position — so they are the abbreviation `c.`, no key printed |
| margin_notes[1].lines[2] | `D. de remilit.` | `D. de re milit.` | B | word division normalized (`de re militari`); not a contraction |
| blocks[4].lines[3] `Guerre.` | period | period | A (=B) | escalated by both; at 5x a round baseline dot with no tail, unlike the comma after `hanté` below; period as printed, sic for sense |
| blocks[2].lines[4] `con-` | `-` | `-` | A (=B) | escalated by A; the sign is a raised mid-x-height dot, not a baseline period — this print's word-break sign (as p042 `nou-`); `-` per §1 |

## Shared mistakes fixed: 0

Agreed lines not re-read. `proſo.` (note a) left closed up as both readers agree.

## Escalations: none

The unkeyed margin block is kept as ONE note with `key: null` per the coordinator's rule; the
`uncertain[]` entry records that body markers `{b}` (blocks[2].lines[5]) and `{c}`
(blocks[2].lines[8]) have no keyed note and that this block serves `{b}` and/or `{c}` (a Digest
citation `l. non omnes, P. à barbaris, D. de re milit.` followed by a decretal `c. mandata de
præſum.`, possibly two citations run together). The validator accepts the noteless markers on the
strength of that entry.

## Odd about the page

- Marker alphabet restarts at `a` (ANNOTAT. XXV.); only one key letter (`a`) is printed in the
  whole margin column although the body has markers a, b, c.
- The unkeyed block's first line is level with the `TEXTE.` heading, ~half a line above
  `Et quant au Preuenu, il y a enuiron` (the beside_line given), two lines below marker c.
- Sic items: `Guerre.` (period for comma), `meſchancé` (for meſchanceté), `dés` (acute for grave),
  `confrontez.` (period in a list), no punctuation after `auoit` before `D'autres`.
- Last line `moins iuſqu'à lx. & d'auãtage qu'il y a ſi` ends on the complete word `ſi`, no break
  sign: runs on to p047 mid-sentence, not mid-word.
