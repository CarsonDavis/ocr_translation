# Reconcile p030

Output: `transcription/final/p030.json` — validator exit 0 (1 ok, 0 failed; two long-s warnings on the sentence-initial capitals `Sur` and `Si`, false positives); normalize_spacing changed 0 lines.

## Differences decided: 7 (A 3 / B 3 / neither 1)

1. `blocks[0].lines[0]` `faluſt-il` (A) vs `faluſt il` (B) → **A**. At 8x the dot between `faluſt` and `il` sits at mid x-height, well above the solid baseline period after `{b}` on the same line: a raised word-joining dot, the hyphen of the inversion, not a period and not a speck. Flagged in `uncertain[]` since it is a round dot, not a stroke.
2. `blocks[0].lines[8]` `ſ'emparer` (A) vs `s'emparer` (B) → **B**. At 4x the letter before the apostrophe is a round x-height s.
3. `blocks[2].lines[1]` `vertuſ,` (A) vs `vertu,` (B) → **A**. At 8x the ill-inked sort after `vertu` is a full-height long ſ (stem from below the baseline to ascender height, hook curling right at the top): a letter, not foul type. Word-final long s, transcribed as printed.
4. `blocks[2].lines[6]` `vertuſ {i}.` (A) vs `vertu {i}.` (B) → **A**. Only the upper hook of the same sort printed, in the same position in the same word; read as ſ. Escalated.
5. `blocks[4].lines[3]` `ROLS` (A) vs `Rols` (B) → **B**. At 4x the o and s are x-height lowercase, the same size as the e of `de`, and the l has a full ascender: not small capitals.
6. `margin_notes[7].lines[5]` (note i) `Saluſte.` (A) vs `Salluſte.` (B) → **B**. Two l's clearly printed at 4x.
7. `beside_line` → **neither**: set to the body line carrying each note's marker (notes a, g, h, i had drifted in one or both reads).

Confirmed from the image / by ruling (agreed, flagged by the coordinator): `ANNOT XV.` without a point after ANNOT (clean space at 4x; both readers); `Mart in` joined per the word-division rule; note b keeps `c. panor. en la xxxvij. diſtinction` as its second citation, since note c proper (`Vergile au j. des Aeneides`) is printed with its own key exactly beside the `{c}` line five lines lower (standing ruling); the blotted key of note c read `c` by both readers from the alphabet, placement and citation, kept with an `uncertain[]` entry.

Shared mistakes fixed: 0.

## Escalations

- `blocks[2].lines[6]` `toute vertuſ {i}.` — body-5.jpg x~360-400 y~340-390: only the upper hook of the sort after `vertu` printed; read as the word-final ſ seen in full on `blocks[2].lines[1]` (body-4.jpg x~900-1000 y~450-520), but it could be dismissed as foul type.

## Odd about the page

- Markers b-i continue from p029's `{a}` under ANNOT. XIV; the alphabet restarts at `a` under ANNOT XV. at the foot (note a in margin-3 beside the last lines), so the page holds notes b-i then a.
- Word-final long s in `vertuſ` twice; `s'emparer` with a round s mid-word.
- The Virgil distich (`Quid non mortalia pectora cogis, / Auri ſacra fames?`) is a paragraph block; the large-type TEXTE extract is a paragraph block.
- Sic readings kept: `ſeut` (for ſeur), `poinr`, `qu's'eſtoit`, `on` for `ont` in `les loix on trouué`.
- The marker for note e is a raised italic e after `predit`; without it note e would be orphaned.
- No signature, no catchword, no foot citation block.
