# Reconcile p035

Output: `transcription/final/p035.json` — validator exit 0 (1 ok, 0 failed); normalize_spacing changed 0 lines.

## Differences decided: 6 (A 3 / B 2 / neither 1)

1. `blocks[5].lines[17]` `meuztre` (A) vs `meurtre` (B) → **A**. At 5x the fourth sort is a z (flat top bar, diagonal, flat foot bar), unlike the r of `tre` / `par` on the same line. Wrong sort, kept as printed (sic).
2. `margin_notes[0].lines[0]` `Horace aũ` (A) vs `Horace au` (B) → **A**. At 8x a raised wave-shaped stroke sits centred over the u, same weight and height as the tilde of `calũnia` below; a separate speck lies to its right. Printed ũ (wrong sort for `au`), kept with an uncertain entry.
3. `margin_notes[1].lines[0]` `l. fiñ. C. de` (A) vs `l. fin. C. de` (B) → **B**. At 7x the mark over the n is a dot/tick like the i-dot, not a bar; `de` carries similar specks. `.C.` normalized per §1.
4. `margin_notes[1].lines[2]` `quis. ij. q. viij` (A) vs `quis. iij. q. viij` (B) → **B**. Three stems, as wide as the `iij` of `q. iij.` two lines down and twice the `ij` of `Paulum. ij.`.
5. `margin_notes[3].lines[0]` `c. fi. & illec` (A) vs `c. ſi. & illec` (B) → **A**. Crossbar on both sides of the stem (italic f as in `fallaciter`), unlike the left-nub ſ of `accuſa`. `illec` (closed-bowl e) confirmed as printed.
6. keys of the restarted run (structural) → **neither**: both readers keyed `c`, `d`; per §4 and the p031 precedent a same-page restart is `a, b2, c2, d2`, so `{c}`→`{c2}`, `{d}`→`{d2}` in body and margin. B's `beside_line` values for these two notes kept (they sit beside `dignes de peines` and `& deliberation`).

Shared mistakes fixed: 0.

## Escalations

none.

## Odd about the page

- Alphabet restarts at `ANNOT. XIX.`: first-run `b` (Horace, continuing p034's `a`), then `a b2 c2 d2`.
- Sic readings kept: `Empereuts`, `meuztre`, `aũ`; `griefuës` with diaeresis; `horri` / `ble` breaks without hyphen.
- Signature reads `C ij` on folio 35 (the letter printed is a C).
- No catchword, no foot citation block.
