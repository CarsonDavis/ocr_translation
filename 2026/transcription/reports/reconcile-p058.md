# Reconcile p058

Output: `transcription/final/p058.json` — validator exit 0.

## Differences decided: 5 (A 4 / B 1 / neither 0)

1. blocks[1].text — B `TBXTE.`. body-1.jpg heading at 4x: the second sort has two closed bowls and a rounded right side, a capital B; the E's of ARREST and of the lower TEXTE are plain three-armed E's. Wrong sort, transcribed as printed with a sic note (B had escalated; resolved).
2. margin_notes[0].lines[0] (unkeyed note) — A `Panor' au c. ij.`. The raised mark after Panor is an abbreviation stroke closing the word; `au` is a distinct word, so one space per §1 word division although the compositor set them tight.
3. margin_notes[4].lines[3] (note d) — A `ſcribã. de p̃te`. margin-2.jpg at 6x: the p carries a flat bar across the top of its stem projecting left, absent from every other italic p on the page (compare `paruu` directly below); an abbreviation stroke, p + combining tilde per §2.
4. margin_notes[7].lines[0] (note g) — A `l. ſicut P. ſu`. margin-2.jpg at 7x: a small faint dot sits on the baseline directly after the l, exactly where the other notes print the period of `l.`; read as a lightly inked period. Faint; noted in uncertain (not escalated).
5. margin_notes[5].lines[1] (note e) — A `D. de cond. & .` (both readers identical; counted because the coordinator asked for the print's `&.`). The print sets a solid period tight against the ampersand; the validator rejects `&.` under §1 spacing, so `& .` is kept and the uncertain entry records that the print has `&.`.

## Shared mistakes fixed: 0 (the `&.` form could not be recorded in the line; see 5)

## Escalations: none

## Marker k

Marker `{k}` (blocks[0].lines[0], carried over from p057) has no keyed note. The only candidate is the unkeyed two-line note `Panor' au c. ij.` / `de maled.`, printed with no key letter (margin-1.jpg, 4x: the line begins directly with the italic capital P at the note column's edge) two lines below the marker. The note keeps `key: null` as printed; the pairing is recorded in uncertain[] on both the marker line and the note.

## Coordinator watch items (all checked at 4–7x)

- `ſeigueur` (blocks[4].lines[21]): u, not n — wrong sort, kept, sic.
- `poſti|dens` (note g lines 3–4): the second tall letter is a crossed italic t — wrong sort for ſ, kept, sic.
- `impudẽce` (blocks[2].lines[1]): a flat wavy bar with hooked ends, not the steep acute of `ſongé`; tilde.
- `pauure; encor` (blocks[4].lines[20]): round upper dot, narrow vertical wedge below the baseline; semicolon kept, colon noted as the alternative.
- note e `&.`: see difference 5.

## Odd about the page

- `TBXTE.` heading with a B for E.
- Three wrong sorts (`TBXTE`, `ſeigueur`, `poſtidens`); `l'eſtré` with an acute twice, plain `l'eſtre` once; an italic capital P standing for the paragraph sign in notes a, d and g.
- Margin set solid in a smaller face, so notes drift several lines from their markers (beside_line values follow reader A's measurement).
