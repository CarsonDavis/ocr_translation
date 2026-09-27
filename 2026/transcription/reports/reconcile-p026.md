# Reconcile p026

Output: `transcription/final/p026.json` — validator exit 0 (1 ok, 0 failed); normalize_spacing changed 0 lines.

## Differences decided: 8 (A 4 / B 3 / neither 1)

1. `blocks[1].lines[21]` `ſeront` (A) vs `ſeront,` (B) → **A**. At 6x the mark after `ſeront` is a faint grey speck below the baseline, far lighter than the solid printed commas on the page (cf. `h,` on line 13); not a printed sort.
2. `blocks[1].lines[24]` `deſendus` (A) vs `defendus` (B) → **A**. At 6x the letter after `de` has a nub on the left only and no crossbar extending right, unlike the f of `enfans` (line 13) whose bar crosses the stem: long ſ, sic for `defendus`.
3. `margin_notes[0].lines[4]` (note a) `q̃` (A) vs `ꝗ` (B) → **B**. At 4x the q carries a horizontal stroke through its descender, below the bowl, not a tilde above: `ꝗ` (U+A757), per the standing ruling.
4. `margin_notes[0].lines[7]` (note a) `affin` (A) vs `aſſin` (B) → **A**. At 4x both letters after `a` carry full crossbars (italic ff ligature).
5. `margin_notes[3].lines[1]` (note d) `defũcto.` (A) vs `defucto.` (B) → **A**. A tilde is visible over the u at 4x.
6. `margin_notes[4].lines[0]` (note e) `Paragra au` (A) vs `Paragra. au` (B) → **B**. At 8x a solid round dot sits on the baseline after `Paragra`, with only a faint wisp trailing from it: a printed period, not a comma.
7. `margin_notes[5].lines[1]` (note f) `matr.` (A) vs `marr.` (B) → **B**. At 4x the third and fourth letters are both italic r; `marr.` as printed, sic for `matr.`.
8. `beside_line` → **neither**: both readers drifted for several notes; set to the body line carrying each note's marker (note d to line 13, where the alphabet wants its marker; note i kept at line 32 as both readers had it).

Confirmed from the image (agreed, flagged by the coordinator): the line-13 marker is the italic h sort (left ascender, open right arch, identical to the marker h on line 30 and the key of note h) where the alphabet wants d; kept as `{h}` twice with an escalated `uncertain[]` entry linking note d. `ignoran ce` joined per the word-division rule. Note i has no marker (both readers); kept with an escalated entry.

Shared mistakes fixed: 0.

## Escalations

- `blocks[1].lines[13]` `pour la legitimité des enfans {h}, encore qu'il y euſt,` — body-3.jpg x~900-1000 y~400-460: marker printed as h where the alphabet and note d want d; `{h}` therefore occurs twice on the page and margin note d has no body marker (also flagged at `margin_notes[3]`).
- `margin_notes[8]` note i (`l. ij. Preal- / leguee.`) — margin-3.jpg y~640-690: no marker i in the body (lines 32-36).

## Odd about the page

- Marker alphabet a-l continues from the ANNOTAT. XI head (p025 ended at k under ANNOTAT. X; restart at a here): a, b, c, h(=d), e, f, g, h, k, l in the body; notes a-l in the margin, i without a marker.
- Sic readings kept: `certian`, `ſe` for `ſi`, `deſendus`, `preuueut`, `incotinant`, `de lay`, `marr.`, `ceſte. ci`.
- The marker for note l is printed as an italic capital L; transcribed `{l}` to match the key.
- Note k is a single line that breaks off after `C. de`.
- `ꝗ` (U+A757) used in note a line 5.
- No signature, no catchword, no foot citation block.
