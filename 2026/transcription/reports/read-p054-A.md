# Read report — p054, reader A

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/A/p054.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/A/p054.json` → exit 0 ("1 ok, 0 failed"), one warning (see below).

## Counts

| item | count |
|---|---|
| body lines | 29 (7 + 9 + 10 + 3) |
| paragraph blocks | 4 |
| heading blocks | 3 (`ANNOTAT, XXXII.`, `TEXTE.`, `ANNOT. XXXIII.`) |
| markers in the body | 3 — `{a}`, `{b}`, `{a2}` |
| margin notes | 3 — keys `a`, `b`, `a2` |
| foot notes | 0 |
| signature / catchword / ornaments | none |
| `uncertain[]` entries | 19 (5 flagged `escalate`) |

Running head `ARREST DV` (spaced caps), folio `54` (old-style 5, matches the manifest).

## Page structure

1. paragraph, `continues_prev: true` — 7 lines, the tail of the TEXTE carried over from p053 (`tre appelé Valentin Rougié …` / `… Iean du Tilh ſon frere.`).
2. heading `ANNOTAT, XXXII.` — spaced capitals.
3. paragraph — 9 lines of small type, markers `{a}` (line 8) and `{b}` (line 9, last line).
4. heading `TEXTE.` — spaced capitals.
5. paragraph — 10 lines of large type (`Sixiément, deux autres teſmoins depo…` / `… nee de S. Laurens.`).
6. heading `ANNOT. XXXIII.` — spaced capitals.
7. paragraph, `continues_next: true` — 3 lines, marker `{a2}` in the last line; the page breaks mid-word (`arri-`).

Every marker has a note and every note has a marker. The alphabet restarts at `a` under ANNOT. XXXIII, so the third marker and its note are keyed `a2` per §4.

## `uncertain[]` entries (19)

1. `blocks[1]` `ANNOTAT, XXXII.` — sic: a **comma** after ANNOTAT, not the period the other annotation heads carry (tail drops below the baseline at 6x).
2. `blocks[0].lines[5]` — a faint raised speck above the gap between `mouchoirs` and its comma; read as a stray mark, not transcribed (could be taken for an apostrophe).
3. `blocks[2].lines[0]` — `ſingulier s` set with a justification gap; joined per §1.
4. `blocks[3]` `TEXTE.` — the point sits at mid x-height, not on the baseline; read as a period.
5. `blocks[4].lines[3]` — (a) a **comma** stands between `dire` and `Mar` although the sense runs on; (b) `Mar` is set in a smaller type size with a full-size capital M (word runs on as `Mar|tin`); not small capitals, so no `spaced_caps`.
6. `blocks[4].lines[4]` — sic: `qu il` printed with a word space and **no apostrophe**.
7. `blocks[4].lines[7]` **(escalate)** — a small square point on the baseline after `l'vne`; fainter/smaller than the page's periods, a speck cannot be excluded. Transcribed as a period.
8. `blocks[4].lines[8]` **(escalate)** — sic: the line ends with a **point sort in the word-break hyphen's position** (`iour.` / `nee`). At 14x it is a wedge dot above the baseline, the same sort as the point after `TEXTE`; the page's break hyphens are level dashes. Same phenomenon as p048 `n'o.`.
9. `blocks[6].lines[0]` — sic: `Encor` set as two words, `En cor`, with a full word space.
10. `blocks[6].lines[1]` **(escalate)** — sic: **`cuy` for `ouy`**. At 12x the sort is open on the right with two inward terminals, exactly like the `c` of `ce` on the line, while the `o` of `auoir` three sorts earlier is a closed ring. Wrong sort, kept as printed.
11. `blocks[6].lines[2]` — (a) marker keyed `a2` (alphabet restart); (b) the first `r` of `arri` is over-inked and can read as a `t` at 13x; compared with the `rr` of `Guerre` on the same line it is an `r`.
12. `margin_notes[0].lines[1]` — after `ex.` a separate short level dash stands at the line end (break sign for `ex-|ceſ.`); spaced per §1 as `graue. de ex. -`.
13. `margin_notes[0].lines[2]` — `prelato` is printed with a plain italic `e`, **not** an `æ` ligature (9x).
14. `margin_notes[1]` — the key letter `b` is followed by a round point before `Accurſe`; the point is treated as part of the key and excluded from `lines`.
15. `margin_notes[1].lines[2]` — the mark after `quinto` is a level mid-height **dash** (hyphen), not the round point the note uses for `l.` and `D.`; a word space follows before `D.`. Transcribed `quinto- D. de`.
16. `margin_notes[2]` **(escalate)** — the note's key letter is a heavily blotted italic sort; read as `a` (→ `a2`) because the body marker it answers is `a` and the alphabet restarts at ANNOT. XXXIII.
17. `margin_notes[2].lines[1]` — `e x` set with a gap; joined to `ex` per §1.
18. `margin_notes[2].lines[3]` **(escalate)** — (a) sic: **`rutela` for `tutela`** (at 18x a stem with a top-right arm and no crossbar, unlike the `t` two sorts later); (b) with line 2 the reference reads `de-|ſti.` = `deſti.`, apparently a contraction or compositor's error for `de teſti.`. Both transcribed as printed.
19. `margin_notes[2]` — this note has **no `beside_line`**: its first line is printed beside the blank band above the `ANNOT. XXXIII` heading, two lines above the first body line of the paragraph that holds its marker.

## Validator warning (checked, not silenced)

`blocks[2].lines[0]: possible normalized long s (si) in "Si ces teſmoins n'euſſent eſté ſinguliers, chacun de-"` — false positive. The word is the sentence-opening **capital** `Si`; capitals have no long-s form in this fount, and the zoom confirms a roman capital S. All the interior s-sounds on the page are long s (`teſmoins`, `n'euſſent`, `eſté`, `ſinguliers`).

## Notes for the reconciler

- **Paper and ink are clean**; no damage, no tears, no heavy show-through. A few light specks and show-through marks (noted above at `mouchoirs` and over `reprochez`) are the only interference.
- **This page's characteristic fault is the point sort.** Three separate places set a round/wedge point where something else belongs: at a line-end word break (`iour.`), possibly mid-line (`l'vne.`), and the comma-for-period in the heading `ANNOTAT, XXXII.`. Expect reader B to differ on all three; they are the highest-value diffs on the page.
- **Two wrong sorts are asserted:** `cuy` for `ouy` (body, `blocks[6].lines[1]`) and `rutela` for `tutela` (margin note `a2`). Both were checked against a closed `o` / a true `t` on the same line at 12–18x.
- **The margin is uncrowded**; all three notes sit in the left column, and there is no foot citation block, no signature and no catchword. The bottom third of the page is blank below the last body line.
- **`Mar` at the end of `blocks[4].lines[3]`** is a genuinely smaller type size, not small capitals — worth a second eye, since it affects whether the block should carry `spaced_caps`. Reader A says no.
