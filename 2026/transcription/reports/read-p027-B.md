# Read report — p027, reader B

**Output path:** `/Users/cdavis/github/translator/2026/transcription/reads/B/p027.json`
(validates clean: `1 ok, 0 failed`, exit 0, no warnings)

## Counts

| | |
|---|---|
| body lines | 36 (3 + 5 + 28) |
| paragraphs | 3 |
| headings | 2 (`TEXTE.`, `ANNOTAT. XII.`, both spaced caps) |
| markers in body | 10 — `m a b c d e f g h i` |
| margin notes | 10 — same keys, all matched |
| foot notes | 0 |
| ornaments | none |
| signature / catchword | none / none |
| running head / folio | `PARLEMENT DE THOLOSE.` / `27` |
| `uncertain[]` entries | 13 (3 flagged `escalate`) |

## Layout

Running head + folio, then a 3-line paragraph finishing the previous annotation
(`continues_prev: true`), ending `ainſi qu'Accurſe meſme enſeigne. {m}`. Then the heading
`TEXTE.`, a 5-line paragraph in the large display roman (the *Arrest* text itself), the
heading `ANNOTAT. XII.`, and a single 28-line annotation paragraph that runs off the foot
of the page (`continues_next: true`). The last line breaks a word **without a hyphen**:
`…que la pei` → `ne` on p028. The margin column is crowded but unbroken, in the small
italic, with the notes set at a much tighter leading than the body, so margin rows do not
align one-to-one with body lines; `beside_line` is therefore only filled in for note `m`.
The alphabet on this page runs `a`–`i` after the isolated `m` that belongs to the previous
annotation's run, so the reconciler should expect p028 to open at `k`.

## `uncertain[]` entries (13)

1. `blocks[0].lines[1]` — `prohition` as printed (sic, for *prohibition*); no `b` at 3x.
2. `blocks[4].lines[9]` — line ends `on n'en,` with a real comma (tail below baseline), though the sense wants no mark.
3. `blocks[4].lines[10]` — `anciẽs`: mark over the `e` is the thick wavy tilde bar, not the thin slanted acute of `parlé` (compared side by side at 9x).
4. `blocks[4].lines[12]` — **escalate**: `quelqueſfois`, two tall letters, vs the round `s` + `f` of `quelquesfois` on the lines immediately above and below. Read `ſ` + `f`; `ff` is possible.
5. `blocks[4].lines[14]` — `plns` as printed (sic, turned letter for *plus*); clear `n` at 5x.
6. `blocks[4].lines[15]` — `par hors` as printed (sic, for *pas hors*); `r` confirmed at 4.5x.
7. `blocks[4].lines[22]` — `ordie` as printed (sic, for *ourdie*); no `u` after the `o`.
8. `blocks[4].lines[24]` — `ſe gaigne`: the `e`'s crossbar did not print, so the glyph reads like a `c`; taken as `e` from the word.
9. `blocks[4].lines[27]` — **escalate**: `iuterualle`; the second letter shows no top arch and a bottom join, so read as `u` (turned `n`) rather than `interualle`.
10. `margin_notes[2].lines[0]` — `vniqne` as printed (sic, turned letter for *vnique*).
11. `margin_notes[2].lines[2]` — **escalate**: `nom & ciba.`; the abbreviation after the ampersand reads letter by letter as `c-i-b-a`, but the citation (C. de mutatione nominis) gives no support, so the sense does not confirm the reading.
12. `margin_notes[8].lines[5]` — `l quarela. C`; printed `l quarela .C` with the period set *before* the C, spacing normalized per §1. `quarela` sic for *querela*.
13. `margin_notes[8].lines[2]` — `Sylanianum`, a single `l` after the `y` at 9x (not `Syllanianum`).

## Notes for the reconciler

- **Paper and ink are good**; no damage, no gutter loss, no faint patches. The only
  illegibility is the ordinary one of this fount: crossbars and arches that failed to ink
  (items 8 and 9 above).
- **`ſſ` vs `ſs` was checked explicitly.** `puniſſable` (line 9 of the annotation) and
  `auſſi` (last line) are both the joined double long-s ligature — no round `s` in either.
  The only unresolved double-tall-letter is item 4.
- **Tilde vs acute was checked explicitly** by cropping `parlé` (thin slanted acute) against
  `aidãt`/`biẽs`/`qu'ẽ`/`anciẽs` (thick wavy bar) at 9x. All four of the latter are tildes.
- **Marker punctuation order varies on this page**: `enſeigne. {m}`, `perſonnes. {a}`,
  `cõmode. {e}` put the point *before* the marker letter, while `d'autruy {b}.`,
  `faux {c}.`, `uile {d}.`, `temps {i}.` put it after. Both are as printed.
  `mari {f}` and `aide {g} &` carry no point at all.
- **Two markers sit tight against the next word in the print**: `aide g&` was read as
  `{g} &` and spaced per §1.
- The italic `t` with the curled top appears in margin note `b` (`C. de muta.`); transcribed
  as an ordinary `t` per §2.
- The narrow strip of the facing page along the left (gutter) edge of the crops was ignored.
