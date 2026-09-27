# Read report — p027, reader A

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/A/p027.json`
(validator: `1 ok, 0 failed`, exit 0, no warnings)

## Counts

| item | count |
|---|---|
| body lines (printed lines inside paragraph blocks) | 36 |
| paragraphs | 3 (3 lines / 5 lines / 28 lines) |
| headings | 2 (`TEXTE.`, `ANNOTAT. XII.`, both spaced caps) |
| markers in the body | 10 (`m a b c d e f g h i`) |
| margin notes | 10 (37 note lines) |
| foot notes | 0 |
| `uncertain[]` entries | 16 |

Running head `PARLEMENT DE THOLOSE.` (spaced caps), folio `27`, signature `null`,
catchword `null`, no ornaments.

## Page structure

1. paragraph, `continues_prev: true` — 3 lines, tail of the previous annotation,
   ends `ainſi qu'Accurſe meſme enſeigne. {m}`
2. heading `TEXTE.`
3. paragraph — the 5-line TEXTE in large type (`En fin, aduertie icelle de Rols…`)
4. heading `ANNOTAT. XII.`
5. paragraph, `continues_next: true` — 28 lines, ends mid-word `…que la pei`
   (no hyphen; continues on p028)

Marker alphabet: the page opens with `{m}` (carrying on the previous page's run),
then restarts at `{a}` under `ANNOTAT. XII.` and runs `a`–`i` continuously. Every
marker has a note and every note has a marker.

## `uncertain[]` entries (16)

1. `blocks[0].lines[1]` — `prohition` sic (no `b`), clear at 8x.
2. `blocks[2].lines[0]` — `En fin` set with a real word gap; may be intended as `Enfin`.
3. `blocks[2].lines[1]` — nasal marks in `affrontemẽt` / `eſtrãge` printed as thick
   slanted strokes; read as tildes, not acutes.
4. `blocks[4].lines[1]` — mark before `{a}` read as a period (round dot, no descender).
5. `blocks[4].lines[10]` — `anciẽs`: nasal mark set slanted, unlike the horizontal
   tildes elsewhere on the page.
6. `blocks[4].lines[11]` — stray raised tick between `punie,` and `autresfois`, not
   transcribed.
7. `blocks[4].lines[12]` — `quelqueffois` sic: two separate barred tall letters,
   unlike the ſſ ligature in `puniſſable` and unlike `quelquesfois` in ll. 11/13.
8. `blocks[4].lines[14]` — `plns` sic (turned u) for `plus`.
9. `blocks[4].lines[15]` — `par` sic for `pas`.
10. `blocks[4].lines[22]` — `ordie` sic for `ourdie`.
11. `blocks[4].lines[24]` — `ſc gaigne`: the second letter shows no crossbar; almost
    certainly `ſe` with a broken e-bar.
12. `blocks[4].lines[27]` — no punctuation printed between `quelconque` and `Vray`.
13. `margin_notes[2].lines[0]` — `vniqne` sic (turned u).
14. `margin_notes[7].lines[2]` — `Carbo. ediſti`: the letter before `ti` descends below
    the baseline (italic long s), but sense and note `f` (`Carbo edict.`) want `edicti`.
    **Escalated.**
15. `margin_notes[8].lines[2]` — `Syllanianum`: two ascenders after `Sy` at 14x.
16. `margin_notes[8].lines[3]` — `l, adulterij.`: the mark after `l` has a descender,
    so read as a comma where the other notes take a period.

## Notes for the reconciler

- **No foot block, no signature, no catchword.** `foot.jpg` shows only the last three
  body lines, the tail of note `i`, and blank paper below; `body-8.jpg` and
  `margin-4.jpg` are blank.
- **The print is unusually error-prone on this page.** Four independent turned/wrong
  sorts (`prohition`, `plns`, `par` for `pas`, `vniqne`) plus `ordie` and the broken
  `ſc`. None of these are reading doubts — the glyphs are clear — so they are
  transcribed as printed and flagged `sic`.
- **The two systematically hard calls** are (a) the slanted nasal marks (items 3 and 5):
  this page's compositor sets the tilde as a slanted stroke in the large TEXTE type and
  once in the body type (`anciẽs`), close enough to the acute in `vſé`/`gayeté` that a
  reader working fast will write `é`/`á`; (b) the tall-letter pairs — `puniſſable` and
  `auſſi` are genuine joined `ſſ` ligatures, but `quelqueffois` (l. 12) is two separate
  barred letters and does **not** match them.
- `beside_line` was deliberately **omitted** from every note. The margin is set on a
  tighter leading than the body (roughly 62px vs 80px at native resolution), so the two
  columns drift against each other down the page and I could not fix the alignment to
  better than ±1 body line from the strips. Wrong placement data seemed worse than none.
- Marker punctuation order is worth a diff check: the print sets the period **before**
  the marker in `enſeigne.m`, `perſonnes.a`, `autruyb.`, `faux c.`, `uile d.`,
  `cõmode.e`, `temps i.` but sets `mari f` and `annees h` and `aide g` with no
  punctuation at all. I transcribed each exactly in the printed order under the §4
  spacing rule.
- Ink is even and the leaf is undamaged; the only faint area is the lower margin column
  (notes `h`–`i`), which needed 8–14x to read.
