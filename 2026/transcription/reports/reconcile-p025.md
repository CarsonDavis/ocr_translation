# Reconcile p025

Output: `transcription/final/p025.json` — validator exit 0 (1 ok, 0 failed); normalize_spacing changed 0 lines.

## Differences decided: 6 (A 4 / B 1 / neither 1)

1. `blocks[3].lines[4]` `melefices` (A) vs `meleſices` (B) → **A**. At 3x the letter after `mele` carries a full crossbar and forms the fi ligature, unlike the nubbed ſ of `diſons` on the line above. Sic for `malefices`, transcribed as printed.
2. `blocks[3].lines[18]` `ſa femme` (A) vs `la femme` (B) → **A**. At 4x the letter before `a` has the hooked long-s top identical to the ſ of `ſoit` on the same line; an l would show a flat serifed top. The line is otherwise sic (`que ſoit ſa femme eſt,`).
3. `margin_notes[2].lines[1]` (note c) `penule.` (A) vs `penult.` (B) → **A**. At 4x the final sort is round and x-height, matching the e of the same word; `penult.` on the next line ends in the stemmed, curl-topped italic t. Possibly a wrong or worn sort; escalated.
4. `margin_notes[7].lines[3]` (note h) `xxxiiij.` (A) vs `xxxiij.` (B) → **A**. At 6x four minims follow `xxx` (i, i, and a joined ij pair); this matches note e's `xxxiiij` for the same Causa 34. Escalated as a minim count.
5. `margin_notes[7].lines[4]` (note h) `q. j.` (A) vs `q. j,` (B) → **B**. At 4x the final mark has a descending tail: a comma.
6. `beside_line` → **neither**: both readers drifted for notes e-k; set to the body line carrying each marker.

Confirmed from the image (agreed, flagged by the coordinator): `Parigra.` (`Pa` / `rigra.`, sic), note e `xxxiiij` (three minims plus j) and `q j.` with no point, `belleſœur` set closed up (kept as the period compound), note k's key is the italic k that resembles `lz` (marker is a roman K; alphabet ...i, k).

Shared mistakes fixed: 0.

## Escalations

- `margin_notes[2].lines[1]` `rigra. penule.` — margin-2.jpg x~250-420 y~90-130: round final sort where sense wants t.
- `margin_notes[7].lines[3]` `Infect. xxxiiij.` — margin-2.jpg x~300-470 y~1215-1265: minim count 34 vs 33.

## Odd about the page

- Marker alphabet restarts at `a` with ANNOTAT. X (p024 ended at `{r}`); markers a-k all have notes and no orphans.
- Sic readings kept: `n'y point` (no `a`), `epouſe`, `melefices`, `que ſoit ſa femme eſt,`, `Parigra.`, `aq. paui.` (for `pluu.`), `leguee`, `chap. 1.` with an arabic 1.
- `plꝰ` uses the raised -us sign (U+A770) in the body.
- Two large-type TEXTE extracts are paragraph blocks (blocks[1], blocks[5]); no signature, no catchword, no foot citation block.
