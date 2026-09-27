# Read report — p040, reader A

**Output path:** `/Users/cdavis/github/translator/2026/transcription/reads/A/p040.json`

**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/A/p040.json`
→ `1 ok, 0 failed`, exit 0. No warnings (no suspicious normalized words reported).

## Counts

| item | count |
|---|---|
| body lines | 26 |
| paragraphs (paragraph blocks) | 3 |
| headings | 2 (`TEXTE.`, `ANNOTAT. V.`) |
| markers in body | 1 (`{e}`) |
| margin notes | 1 (key `e`) |
| foot notes | 0 |
| ornaments | 0 |
| uncertain entries | 2 |

Block sequence: paragraph (3 lines, `continues_prev`) → heading `TEXTE.` → paragraph
(21 lines, large type) → heading `ANNOTAT. V.` → paragraph (2 lines, `continues_next`).

Page furniture: `running_head` `ARREST DV` (spaced caps), `folio` `40` (matches the
manifest's expected folio), `signature` null, `catchword` null.

## `uncertain[]` entries

1. **`blocks[4].lines[1]`** — `grãd preuue de l'hõneſteté de ladite de Rols, qui ne vou`
   (**escalate: true**). The last line of the page breaks *vouloit* **without a printed
   hyphen** (verified at high zoom right out to the edge of the type page — the space after
   `vou` is blank). The compositor also set `ne` and `vou` very tight. I transcribed them as
   two words per the word-division rule in conventions §1 (`conceut in-`), but conventions §1
   also gives the illustrative hyphen example `"qui nevou-"`, which conflicts on both points.
   The reconciler should settle (a) `ne vou` vs `nevou` and (b) the absence of the hyphen.
2. **`blocks[2].lines[1]`** — `re de proceder, ſ'en enſuit Ordõnãce de`. The mark over the
   `a` of `Ordõnãce` is lightly inked and prints as a slanted stroke that could be mistaken
   for an acute. Read as a tilde (giving *Ordonnance*), consistent with the clearly wavy
   tilde over the `o` of the same word.

## Notes for the reconciler

- **Layout.** Verso; single body column with the margin column on the **left**. The crops
  keep a strip of the facing recto along the right (gutter) edge — that column of italic
  fragments (`lu / d / le / des / ble / dre …` and `m Pl… / frig.& / de frig.`) belongs to
  p041 and has been ignored.
- **The one margin note** sits beside body line 2 of the opening paragraph:
  key `e`, lines `Seneque au` / `prologue des` / `Declamatiõs.` The key letter is an italic
  `e` (crossbar clearly visible at 8×; it is not a `c` — compared directly against the roman
  `c` of `cõme` on the same line). Marker and note agree; nothing is orphaned.
- **Marker alphabet.** No preceding page was available as context, so the `e` could not be
  checked for alphabet continuity with p039. It is legible on its own, so this is not
  flagged as doubtful, but a later continuity pass may want to confirm it.
- **`ſſ` vs `ſs`.** The page has exactly one double-s: `auſsi` in body line 2. Zoomed to
  ~8×: the second letter is unmistakably a short round `s`, so `ſs`, not `ſſ`.
- **Punctuation checked glyph by glyph at 6–10×**, not from sense:
  - `enfance.` — round dot on the baseline (period), set tight against `enſemble`; normalized
    to `enfance. enſemble`.
  - `d'ongle {e}.` — period after the marker.
  - `ſi beſoin eſt . &` — the print sets a space **before** the period; normalized to
    `eſt. &`.
  - `du Tilh.hors mis,` — period (single baseline dot) after `Tilh`, then comma after `mis`.
  - `ouye,reſ-` and `Rols,qui` — commas (tail below the baseline), not periods.
  - `mariage:à quoy` — **colon** (two dots, one at x-height, one on the baseline), not a
    semicolon and not a period.
  - `ble. dont` — period.
- **Tildes.** `cõme` (×2), `Ordõnãce`, `neãtmoins`, `preſẽtera`, `perſõne`, `ſerõt`, `tãs`,
  `põd`, `grãd`, `hõneſteté`, and `Declamatiõs.` in the margin. One tilde over a consonant:
  `q̃` (= *que*) in `neãtmoins, q̃ ladite`, transcribed as `q` + U+0303 per §2.
- **Line breaks without a hyphen** (normal for this print, per §1, so not flagged):
  `reſul` / `tãs` (line 10→11 of the `TEXTE` paragraph) and the page-final `vou` (see
  uncertain entry 1). All other line-end breaks do carry a printed hyphen:
  `ia-`, `manie-`, `du-`, `declara-`, `reſ-`, `enſem-`.
- **Spacing normalizations applied** (print sets them tight): `faiſoit(cõme` → `faiſoit (cõme`;
  `meſmes,&` → `meſmes, &`; `liez,&` → `liez, &`; `Tilh ,ſoy` → `Tilh, ſoy`;
  `ente ndre.` → `entendre.` (the compositor left a gap inside the word).
- **No foot citation block, no signature, no catchword.** `foot.jpg` shows the last two body
  lines and then blank paper down to the trimmed edge; `body-8.jpg` is empty below the text.
  The `ANNOTAT. V.` annotation runs over onto p041, hence `continues_next: true`.
- **Condition.** Paper is clean and the impression is even; light show-through from the
  facing page is visible in the white space between `TEXTE.` and the first body line but
  does not obscure anything. No damage, no faint or dropped type.
