# Read report — p034, reader B (opus)

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/B/p034.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/B/p034.json` → `1 ok, 0 failed`, exit 0, no warnings.

## Counts

| item | count |
|---|---|
| body lines | 31 (16 in the large-type paragraph, 15 in the annotation paragraph) |
| paragraphs | 2 |
| headings | 1 (`ANNOTAT. XIII.`, spaced caps) |
| markers in the body | 1 (`{a}`) |
| margin notes | 1 (key `a`) |
| foot notes | 0 |
| uncertain[] entries | 12 (none escalated) |

Running head `ARREST DV` (spaced caps), folio `34`, no signature, no catchword, no ornaments,
no foot citation block.

## Page structure

Verso. Folio `34` at the top left, running head `A R R E S T   D V` centred. The large text
type runs 16 lines and continues both the sentence and the paragraph from p033 (the page opens
mid-word: `ſtance` = `[in]ſtance`); it closes with a full stop at `les coups.` and does **not**
run on, because the `ANNOTAT. XIII.` head follows. Below the head, the annotation is set in the
smaller type, indented on its first line, 15 lines, and runs on to p035 (`auoit` with no stop).
Roughly the bottom sixth of the page is blank below the last line — no block recorded, per §6.

One marginal note only, in the italic margin font, keyed `a`, beginning beside the eleventh line
of the annotation paragraph. The letter alphabet restarts at `a` here, at the `ANNOTAT. XIII.`
section boundary, so no `a2` keying is involved.

## uncertain[] entries (12) — one line each

1. `blocks[0].lines[1]` — `couuremenr` sic: at 8x the final sort is an **r** (stem + top-right arm, no crossbar, no curved foot), not a `t`; same fault as `demeureronr` on p024. Same line: `quantaux` set closed up, split to `quant aux` per §1.
2. `blocks[0].lines[4]` — `enhaine` set closed up, split to `en haine`; the mark after `entendre` is a round baseline dot (period), not a comma.
3. `blocks[0].lines[5]` — the mark over the e of `auroiẽt` is a thick near-horizontal bar, clearly unlike the thin slanted acute on `recherché` in the same line: read as a tilde (likewise `moyẽs`, `auroyẽt`).
4. `blocks[0].lines[6]` — `poſsibles`: long ſ + short round s at 6x, **not** `ſſ`.
5. `blocks[0].lines[14]` — `lepouuant` set closed up, split to `le pouuant` per §1.
6. `blocks[0].lines[15]` — the o of `receuoir` is only partly inked (faint broken ring); reading not in doubt.
7. `blocks[2].lines[1]` — `preſente a la mort`: no grave printed on the `a` (contrast the clear `à la mort` on line 13 of the same block).
8. `blocks[2].lines[2]` — line breaks after `The` with no hyphen (`The` / `ſalie`); normal for this print, noted for the reconciler only.
9. `blocks[2].lines[5]` — the first o of `pouuoit` is partly inked, like `receuoir` above.
10. `blocks[2].lines[11]` — `auſsi`: long ſ + short round s at 7x, **not** `ſſ`; the mark after `Danaus` has a tail below the baseline (comma).
11. `margin_notes[0]` — the note's key letter is a heavily inked italic `a` (nearly a blob) at the head of line 1; identified from the body marker and the section restart. Line 2 is set `la      tragedie` with a wide gap, normalized to one space.
12. `margin_notes[0].lines[3]` — the second letter of `uenal` is blotted (filled e plus a stray mark); read as e on the sense (`Iu-` / `uenal` = Iuuenal). Numeral transcribed `vj` as printed.

## Things the reconciler should know

- **Double-s checked explicitly at 6–8x on every occurrence.** `ſſ` confirmed in `eſſay`,
  `aſſailli`, `aſſom-`, `deſſus`. `ſ` + round `s` confirmed in `poſsibles` and `auſsi`. These
  two are the likeliest place for reader A to differ.
- **`couuremenr` (blocks[0].lines[1]) is the one reading most likely to be contested.** The
  glyph was compared side by side with a certain `t` (`quant`, same line; `ment`, line 4) and a
  certain `r` at high magnification; it is an `r`. It is a sic, not a transcription slip.
- **Three word-division normalizations** were applied per §1 where the compositor set the words
  closed up: `quant aux`, `en haine`, `le pouuant`. Also `de ſon`, `fit iadis`, `en terre`,
  `ce faire` etc., all set tight. If reader A kept any of these closed up, the convention
  governs, not the print.
- **Tilde vs acute**: this page has both on the letter e within a few lines of each other
  (`auroiẽt` / `recherché`; `moyẽs`; `auroyẽt`; also `tãt`, `deuãt`, `ſãs`, `cõme`). The tilde is
  a thick, blunt, near-horizontal bar; the acute is a thin stroke slanting up to the right. Both
  forms are present and distinguishable at 6x.
- **Ink**: generally clean and dark. Two vowels are lightly inked (`receuoir`, `pouuoit`); the
  margin note's key letter and the `e` of `uenal` are over-inked. No damage, no staining, no
  show-through affecting a reading. The right-hand strip of the facing recto is present along
  the gutter in all crops and was ignored.
- **Punctuation checked glyph by glyph** at clause boundaries: the period after `entendre`, the
  period after `perdre`, the semicolon after `comptes`, the colons after `bien`, `amis`, `faire`
  and `eſpoux`, and the commas after `mari`, `Admetus`, `Danaus` and `filles` were each zoomed
  and identified by shape, not by sense.
