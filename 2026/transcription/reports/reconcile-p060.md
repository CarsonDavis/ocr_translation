# reconcile-p060

**Output:** `/Users/cdavis/github/translator/2026/transcription/final/p060.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/final/p060.json` → `1 ok, 0 failed` (exit 0).
**Diff check:** re-ran `scripts/diff_reads.py p060` to the scratchpad before deciding; output identical to `transcription/diff/p060.md` (B's read on disk is the one the diff was built from).

## Counts

- Differences decided: 6 (4 body, 2 note): **A 2 / B 4 / neither 0**.
  - A: `ont dõnce deux …` (body 12), `l. parentes.` (note a, line 0).
  - B: `… deſdites ſœurs.` (body 13), `prouuer vne ehoſe …` (body 22), `… comme'quand` (body 29), `alleguét. Ac-` (note c, line 1).
- Shared mistakes fixed: 0. `beside_line` values kept from A (b level with `l'aage…`, c with `le parent…`, checked on margin-3.jpg); B's were one line off.

## Escalations

1. `margin_notes[0].lines[0]` "l. parentes.": the mark between `l` and `parentes` is a small round dot printed at ascender height, not on the baseline. Dot-shaped, so not the apostrophe B read; taken as the point of the abbreviation `l.` printed high (turned or high-riding sort). Notes b and c print the same abbreviation as bare `l etiam`, so a stray sort with intended `l parentes.` is possible. Pointer: `pages/strips/p060/margin-2.jpg`, first line of note a, x≈400, y≈1085.

No other escalations. B's escalations on `des peres`, `parẽs`, `comme'quand`, `alleguét.` and `aud. c. literas` were checked and resolved (see the `uncertain[]` notes); A's on `vnc ehoſe` is resolved against A.

## Notes on the page

- Wrong sorts confirmed and kept with sic notes: `Preniierement`, `Cuerre`, `dõnce`, `ehoſe` (body); `alleguét.` (margin).
- `vnc ehoſe`: at 10x both the last letter of `vne` and the first of `ehoſe` carry the closed eye and crossbar of an e. The c/e transposition A reported is not in the print; only `ehoſe` is wrong.
- `comme'quand`: a comma-shaped sort at apostrophe height stands between the two words with no word space, the same form as the apostrophes of `qu'en` and `d'vn` on the same strip. Printed, so transcribed as printed with a sic note; the sense wants plain `comme quand`.
- `alleguét.`: the final sort is the italic fount's curled-top t (as in `parentes`, `tut.`) with no descender; the italic z of this fount (cf. `ſoubz`, p057) has a flat top and a tail below the baseline.
- `aud. c. literas` (note e) is the faintest line on the page but reads a-u-d at 7x; kept, not escalated.
- The print sets several colons and commas tight (`Gaſcogne:leſquelles`, `perſonne:comme`, `desperes`); all normalized per conventions §1, as both readers did.
