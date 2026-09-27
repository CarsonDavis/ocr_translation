# Reconcile p024

Output: `transcription/final/p024.json` — validator exit 0 (1 ok, 0 failed); normalize_spacing changed 0 lines.

## Differences decided: 6 (A 3 / B 2 / neither 1)

Body: 100% agreement, nothing re-read.

1. Structural: A's trailing `blank` block for the empty lower third of the page → **B** (no block), per conventions §6 and the standing ruling.
2. `margin_notes[5].lines[3]` (note l) `na. l rẽ alienã` (A) vs `na. l. rẽ alienã` (B) → **A**. At 6x no point is printed after the `l`; the dot after `na` is a round period sitting against the foot of the italic l (the apparent tail is the l's foot), so `na.` stands.
3. `margin_notes[10].lines[0]` (note q) `præalle-` (A) vs `praalle-` (B) → **A**. At 4x the glyph after `pr` carries the attached e loop of the æ ligature, and a separate `a` follows it (seven glyphs).
4. `margin_notes[11].lines[0]` (note r) `quæcunqꝫ` (A) vs `quacunqꝫ` (B) → **A**. The vowel after `qu` is the æ ligature; the small 3-shaped `-que` sign hanging below the baseline after `q` is confirmed and rendered `ꝫ` (U+A76B).
5. `margin_notes[11].lines[4]` (note r) `ſig.` (A) vs `ſi.` (B) → **B**. At 6x the blotted group is a long ſ joined to a small closed loop, with no g bowl or descender: the ſi ligature. Escalated because the sort is blotted.
6. `beside_line` → **neither**: A supplied none and B only for note f; filled from the body line carrying each marker (note f keeps B's line).

`bona fidei` (notes m, o, r): both readers already split the closed-up `bonafidei` per word division; agreed, kept.

Shared mistakes fixed: 0.

## Escalations

- `margin_notes[0]` note f — no `{f}` marker on this page (p023's last marker is `{e}`). On `blocks[0].lines[1]` a blank of two to four letter-widths stands between `vertu` and the period where a marker sort may have failed to print; nothing visible, nothing transcribed. Kept with key f. Pointer: body-1.jpg line 2, right of `vertu`.
- `margin_notes[11].lines[4]` `D. de verb. ſi.` — margin-3.jpg x~540-650 y~905-975: blotted ligature, `ſig.` possible.

## Odd about the page

- Sic readings both readers confirmed at high magnification: `demeureronr` (line 14), `lcy` for `loy` (line 33), `perſcription` (line 36, set `pe rſcription`), `ſusap.` for `ſucap.` (note l), `haredẽ` without æ (note k).
- `continues_next` false is confirmed: p025 opens with the heading `TEXTE.`.
- No signature, no catchword, no foot citation block.
