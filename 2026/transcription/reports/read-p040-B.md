# Read report — p040, reader B

**Output path:** `/Users/cdavis/github/translator/2026/transcription/reads/B/p040.json`

**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/B/p040.json`
→ `1 ok, 0 failed`, exit 0, no warnings.

## Counts

| Item | Count |
|---|---|
| Body lines (all paragraph blocks) | 26 (3 + 21 + 2) |
| Paragraphs | 3 |
| Headings | 2 (`TEXTE.`, `ANNOTAT. V.`) |
| Markers in body | 1 (`{e}`) |
| Margin notes | 1 (key `e`) |
| Foot notes | 0 |
| `uncertain[]` entries | 2 |

## Page structure

- Verso. Running head `ARREST DV` (spaced caps), folio `40` at the left, matching the manifest.
- `blocks[0]` — 3 lines of small-type annotation text, `continues_prev: true`. Tail of the
  previous annotation (ANNOTAT. IIII), carrying the single marker `{e}`.
- `blocks[1]` — heading `TEXTE.` (spaced caps).
- `blocks[2]` — the 21-line large-type TEXTE quotation, a self-contained paragraph
  (indented start `La matiere…`, ends `oncques entendre.`). No markers, no margin notes
  beside it.
- `blocks[3]` — heading `ANNOTAT. V.` (spaced caps).
- `blocks[4]` — 2 lines of small type opening ANNOTAT. V, `continues_next: true`
  (runs on to p041, which begins `lut demãder…`).
- No signature, no catchword, no foot citation block, no ornaments. The lower third of the
  page below the two annotation lines is blank (`body-8.jpg` is entirely empty).
- `margin-2.jpg`, `margin-3.jpg`, `margin-4.jpg` are entirely blank; the page carries only
  one margin note, at the very top of the outer (left) margin.

## `uncertain[]` entries

1. **`blocks[4].lines[1]`** (escalated) — `…de Rols, qui ne vou`. The print sets `nevou`
   tight and there is **no** line-end hyphen. Transcribed `ne vou` per §1 word-division
   normalization (cf. the doc's own `conceut in-`). Flagged because `docs/conventions.md`
   §1 uses this exact line as its hyphen example, written `"qui nevou-"` / `"loit"` — which
   conflicts with the page on both counts: no hyphen is printed, and p041 opens with `lut`,
   so the word is `voulut`, not `vouloit`. The reconciler should fix the house rule (and
   possibly the conventions example).
2. **`margin_notes[0].key`** — the margin key letter is inked/blotted and at first sight
   resembles `ç`; a small mark sits just below it (ink speck or set-off). Read as `e` on the
   evidence of the body marker, which is an unambiguous italic `e`, and of alphabet
   continuity: p041 restarts the run at `a` with ANNOTAT. V, so `e` is the last marker of
   the ANNOTAT. IIII run.

## Explicit checks the prompt asked for

- **`ſſ` vs `ſs`** — the page has exactly **one** double-s: `auſsi` in `blocks[0].lines[1]`.
  Checked at 3.5× and again at 7×: long ſ followed by a **short round s**. Transcribed
  `auſsi`, not `auſſi`. Every other `ſ` on the page is a single long s
  (`enſemble`, `ſans`, `faiſoit`, `noſtre`, `ruſtre`, `miſe`, `ſur`, `enſuit`, `preſẽtera`,
  `perſõne`, `eſtre`, `beſoin`, `eſt`, `teſmoins`, `ſoy`, `diſant`, `ſeront`, `ſerõt`,
  `reſul`, `reſ-`, `meſmes`, `reſponſes`, `adiouſte`, `ſes`, `conſeilloyent`, `ſeparation`,
  `ſeul`, `hõneſteté`).
- **Punctuation at clause boundaries** — each was zoomed and judged by shape, not sense:
  - `enfance.` — round dot on the baseline → period.
  - `eſt . &` — round dot (print sets a space before it; normalized to `eſt. &`).
  - `Tilh. &`, `Tilh. hors`, `proces.`, `ble.`, `entendre.`, `Martin.`, `Declamatiõs.` — periods.
  - `mariage:à` — **two dots, no tail → colon**, not semicolon. Normalized to `mariage: à`.
  - Commas at `neãtmoins,`, `ouye,`, `Guerre,`, `faicts,`, `meſmes,`, `mis,`, `adiouſte,`,
    `liez,`, `Rols,`, `proceder,`, `tion,` all show the tail below the baseline.

## Other notes for the reconciler

- **Tildes.** Seven tilde-bearing sorts, all the same angular tilde and all verified at 6–7×:
  `cõme` (twice), `Ordõnãce` (**both** marks are tildes — at reading size the `ã` can look
  like an acute), `neãtmoins`, `q̃` (q + U+0303, = *que*), `preſẽtera`, `perſõne`, `ſerõt`,
  `tãs`, `põd`, `grãd`, `hõneſteté`, and `Declamatiõs.` in the margin note. Only `hõneſteté`
  carries a genuine acute (final `é`), and `à` in `mariage: à quoy` a genuine grave.
- **Word breaks without a hyphen** (normal for this print, per §1, no entry needed):
  `reſul` / `tãs`, `ſans pou` / `uoir`, and `qui ne vou` / (p041) `lut`.
- **Hyphenated line ends kept:** `ia-`, `manie-`, `du-`, `declara-`, `reſ-`, `enſem-`.
- **Spacing normalized** in `enfance.enſemble`, `eſt . &`, `Tilh.hors`, `faiſoit(cõme`,
  `mariage:à`, and `ente ndre` → `entendre`.
- **Condition:** the page is clean — no damage, no show-through worth noting beyond faint
  set-off from the facing page in the blank areas. The marginal note is lighter than the
  body but fully legible. The gutter strip of the facing page (p041: `lu`, `d`, `le`, …,
  and `frig. &` / `de frig` at the foot right) appears along the **right** edge of the
  read image and of `foot.jpg`; it was ignored throughout.
- **Markers cross-check:** the single `{e}` in `blocks[0].lines[1]` has its `margin_notes`
  entry with key `e`; there is no note without a marker and no marker without a note.
