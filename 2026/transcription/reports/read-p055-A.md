# read-p055-A

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/A/p055.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/A/p055.json` → `1 ok, 0 failed` (exit 0, no warnings).

## Counts

| | |
|---|---|
| body lines (paragraph lines) | 26 |
| body lines incl. the 4 display/heading lines | 30 |
| paragraphs | 5 |
| headings | 4 (`TEXTE.`, `ANNOTAT. XXXIIII.`, `TEXTE.`, `ANNOTAT. XXXV.`) |
| markers `{x}` in the body | 0 |
| margin notes | 1 (key `a`) |
| foot notes | 0 |
| ornaments | 0 |
| `uncertain[]` entries | 5 |

Running head `PARLEMENT DE THOLOSE.` (spaced caps, closed up), folio `55` (matches the manifest),
signature `D iiij`, no catchword.

## Page structure

1. paragraph, `continues_prev: true`, 5 lines — end of the annotation running over from p054.
2. heading `TEXTE.` (spaced caps)
3. paragraph, 7 lines, large text type — the *seconde raison* (the Rieux enquiry).
4. heading `ANNOTAT. XXXIIII.` (spaced caps)
5. paragraph, 6 lines, small annotation type — ends `qu'en a eſcrit Plutarque.`
6. heading `TEXTE.` (spaced caps)
7. paragraph, 7 lines, large text type — the *troiſieme* raison (the Basque language).
8. heading `ANNOTAT. XXXV.` (spaced caps)
9. paragraph, 1 line, `continues_next: true` — `Bien que la langue des Baſcouz ſoit fort obſcure &`,
   the annotation breaks off at the foot and runs on to p056.

Below that line: the signature `D iiij`, then blank paper to the trimmed foot.

## `uncertain[]` entries (5)

1. **`margin_notes[0]` (escalate: true)** — the margin note with key `a`
   (`Plutarque / au liure v. de / placit. Philoſ.`) has **no marker `{a}` anywhere in the body**.
2. **`blocks[0].lines[2]`** — `de bouler deuant S. Quentin:` — wrong sort, `bouler` for `boulet`;
   the final glyph is unmistakably `r`. Transcribed as printed, `sic`.
3. **`blocks[0].lines[3]`** — `d'eſt eſpoinçonnez` — `sic`; sense wants `d'eſtre`, the print sets
   `d'eſt` (checked at 8x: d + apostrophe + e + `ſt` ligature, then a clean word space).
4. **`blocks[6].lines[3]`** — `peu enten dible` — `entendible` set with a full word space, `sic`.
5. **`blocks[0].lines[0]`** — the page opens mid-word with `ua` (completing a word broken at the
   foot of p054, presumably `arri-`/`ua`); no preceding-page transcription existed to confirm.

## For the reconciler

- **The missing marker is the one real open question.** The note keyed `a` is the first of a
  new alphabet run (`a` restarting at this section boundary); the Plutarch citation belongs to the
  end of `ANNOTAT. XXXIIII.`, i.e. to `qu'en a eſcrit Plutarque.` (set as `beside_line`). The note's
  first line is printed roughly midway between `…le lecteur à ce` and `qu'en a eſcrit Plutarque.`,
  fractionally closer to the latter; either could reasonably be chosen as `beside_line`.
  Its placement in the outer margin is unambiguous, the marker in the body simply is not there.
- **Double-s was checked at ≥3x on every occurrence.** The page mixes both forms and the difference
  is real, not a reading artefact:
  - `auſſi` in `blocks[4].lines[0]` (`Ceſte preuue auſſi n'eſtoit pas`) is a true **`ſſ`** ligature.
  - `auſsi` in `blocks[2].lines[6]` (`confirment auſsi.`) is **`ſs`** — long s followed by a clearly
    round, short s. Do not let these two be levelled to one spelling.
- **Punctuation checked glyph by glyph at clause boundaries.** All four colons on the page are real
  two-dot colons: `S. Quentin:`, `concluante:`, `tude:`, and `pays :&` (printed with a space before
  the colon and none after; normalized to `pays: & neantmoins` per §1).
- Normalizations applied per §1 that a reader could differ on: `La ſecõ de raiſon` is set with a
  visible gap but is the single word `ſecõde`, so it is closed up; `enten dible` has a gap of the
  same width as the surrounding word spaces, so it is **kept** open and flagged.
- Ligatures decomposed as required: `ct` in `eſdictes`, `lecteur`; `ff` in `different`;
  `ſt` in `atteſtatoire`, `eſt`, `eſtoit`, `enqueſtes`, `remonſtré`, `d'eſt`.
- Tildes kept, not expanded: `ſecõde`, `Sãxi`, `cõme`, `aduiẽt`, `Frã` (line-end, no hyphen).
- Line-end word breaks without a hyphen are frequent in the large-type blocks and are normal here:
  `Guer`/`re`, `rappor`/`té`, `ſimili`/`tude`, `qu'`/`on`, `Frã`/`çois`. None is hyphenated in the print.
- `ſçauoir` carries a cedilla; `ſcait` (twice) does not. Both verified at high magnification.
- Condition: paper is clean and the impression strong; no damage, no faint ink, nothing lost to the
  gutter. Scattered brown specks and some show-through from the verso sit below the baseline in
  several lines (e.g. under `ſouuentesfois`, under `& Gaſcon`); they are not punctuation and were not
  transcribed. The bottom outer corner is dog-eared in the scan but carries no text.
- The crops keep a strip of the facing verso along the left (gutter) edge; it was ignored throughout.
