# Reconcile p036

Output: `transcription/final/p036.json` — validator exit 0 (1 ok, 0 failed); normalize_spacing changed 0 lines.

## Differences decided: 5 (A 0 / B 3 / neither 2)

1. `blocks[0].lines[28]` `leur eſt. comme` (A) vs `leur eſt comme` (B) → **B**. At 6x the mark between `eſt` and `comme` is a faint ragged thread rising from a small foot blob to mid x-height; the fount's period (`fait.` line 21) is a solid square dot and its comma (`permis,` same line) has a tail below the baseline. Foul matter / riding space, not a cast point. Uncertain entry with pointer (`body-5.jpg` x~255-275 y~425-460).
2. `margin_notes[4].lines[1]` `vij.` (A) vs `vij,` (B) → **B**. Clear tail below the baseline at 6x.
3. `margin_notes[6].lines[6]` `ij.` (A) vs `iij.` (B) → **B**. Three stems and three dots, same width as `iij. A la pre-` above, wider than `ij. Saphonie`.
4. `margin_notes[1].key` null (A, B) → **neither**: `f`. `Leuit. c. xix.` is printed with no key letter, and body marker `{f}` has no note; promoted per the standing ruling, stated in `uncertain[]`.
5. `blocks[0].lines[38]` `cou roux` (A, B) → **neither**: `couroux`. Justification gap inside a single word, joined per §1 (as both readers did for `preſt e` → `preſte` on line 32, and as p030 did for `Mart in`). Single r kept (sic).

Kept as agreed: `main 'vne` (a solid raised sort in apostrophe position, likely the apostrophe of `l'vne` with the l unprinted), `nons`, `q'vne`, `ſ'emflãboit`, `preſte`.

Shared mistakes fixed: 1 (`cou roux` normalization).

## Escalations

none. (The `eſt comme` mark is noted with a pointer, `escalate: false`.)

## Odd about the page

- Note `e` (`l. famoſi. D. ad l. Iul. maieſta.`) has no `{e}` marker on this page or on p035 (whose restarted run ends at `d2`); the printer omitted it. Kept per §4.
- The `Leuit. c. xix.` citation has no printed key; assigned to `{f}`.
- `beside_line` for notes g and l taken from B (the g–l citations are one tight block; placement is approximate to a line).
- No signature, catchword, or foot block.
