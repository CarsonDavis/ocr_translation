# Reconcile p048

Output: `transcription/final/p048.json` — validator exit 0 (1 ok, 0 failed; one WARNING on `Sur`, a roman capital S, explained in `uncertain[]`); normalize_spacing changed 0 lines.

## Differences decided: 3 (A 1 / B 2 / neither 0)

1. `blocks[0].lines[10]` `n'o-` (A) vs `n'o.` (B) → **B**. At 5x the mark is a solid square dot on the baseline, the same sort as the periods of `iuge.` and `l'autre.`; the hyphen of `ſimili-` two lines above sits at mid x-height. A point sort standing where the break hyphen was wanted (the word runs on `n'o|ſans`); transcribed as printed, sic.
2. `blocks[2].lines[17]` `mct` (A) vs `mot` (B) → **A**. At 8x the middle sort is an open c with top and bottom right terminals, like the c of `ces`; the o's of `dirons` on the same line are closed and evenly inked. Wrong sort for `mot`, kept as printed (sic).
3. `margin_notes[1].lines[0]` `En l'anno-` (A) vs `En l'anno` (B) → **B**. The mark above the final o is a small hook-shaped stroke rising above x-height, not the level mid-height dash this italic uses for breaks; a stray/accent-like mark, not transcribed, noted.

Decided as asked: `le reſte` and `d'auantage` (both readers already normalized word division per §1) kept; note b line 1 `lXXiij` kept as the sorts print (lowercase l, two capital X, `iij`); after `{b}` a large baseline dot with a small wedge fleck above it, the page's colons having two equal dots — period kept, colon noted as not excluded.

Shared mistakes fixed: 0.

## Escalations

- `margin_notes[0].lines[0]` `l. iij. P. ij. D.`: the numeral between `P.` and `D.` prints as one stem with two dots and a j-curl (ij pair with the first stem unprinted, or a bare j) — `pages/strips/p048/margin-3.jpg` x~450-520, y~1035-1070. Kept `ij`, escalation from reader A retained.

## Odd about the page

- Folio printed `58` on true page 48 (matches the manifest).
- `ANNOTAT. XXVI.` restarts the alphabet at `a`; `TV DOIS` set in spaced capitals, block flagged.
- `Tilb` for Tilh; unhyphenated breaks `c'eſt|oit`, `pe|rils`, `qua|lité`.
- No foot block, signature or catchword.
