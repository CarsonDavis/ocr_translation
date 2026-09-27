# Reconcile p023

Output: `transcription/final/p023.json` — validator exit 0 (1 ok, 0 failed); normalize_spacing changed 0 lines.

## Differences decided: 4 (A 2 / B 2 / neither 0)

1. `blocks[0].lines[1]` `ſen` (A) vs `ſ'en` (B) → **B**. At 3x a distinct raised tick stands between ſ and e at apostrophe height, separate from the ſ nub; the ſ of `hiſtoire` on the line above carries no such mark, so an apostrophe is printed.
2. `margin_notes[1].lines[1]` `liure` (A) vs `liurc` (B) → **B**. The final letter is an open c without an eye, unlike the closed e of `Valere` directly above: a wrong sort, transcribed as printed.
3. `margin_notes[3].beside_line` → **A** (the line carrying `{d}`); B pointed at the line above. Not a text difference; the diff did not list it.
4. `margin_notes[4].lines[3]` `empe.` (coordinator query `empe.`/`empt.`) → kept `empe.` (both readers). At 6x the final sort is round with no stem, so it is not a t; it could be e or an open c. Sense wants `empt.`; transcribed as printed and escalated.

Also confirmed from the image (both readers agreed, flagged uncertain): `Valere an` (n), `piecẽs` (clear tilde), `gué.` (light acute stroke), the tall hooked `ſ` between `ix.` and `xvi` in note b (not the round c of note a).

Shared mistakes fixed: 0.

## Escalations

- `margin_notes[1].lines[1]` `liurc ix. ſ. xvi` — margin-1.jpg y~1075-1135: the letter between `ix.` and `xvi` is a hooked tall glyph with a tail (italic ſ or ç); sense wants `c.`.
- `margin_notes[4].lines[3]` `actio. empe.` — margin-3.jpg y~985-1020: round final sort, e or c; sense wants `empt.`.
- `blocks[4].lines[2]` `piecẽs` — body-6.jpg x~1230 y~360: a tilde where sense wants plain `pieces`.

## Odd about the page

- Note `a` has no `{a}` marker on this page; it answers the orphan marker `{a2}` at the end of p022 (whose final records no note for it). Kept with key `a` as printed and an `uncertain[]` entry, per the standing ruling.
- The article `le` is set twice across `blocks[0].lines[6-7]` (`fit rendre le` / `le bien`), sic.
- `generenx` (turned u) and `chacum` are sic, both readers agree.
- Signature `B iiij`; no catchword; no foot citation block; body-9 and margin-4 strips are blank.
