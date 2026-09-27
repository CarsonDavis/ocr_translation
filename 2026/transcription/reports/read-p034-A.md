# Read report — p034, reader A

**Output path:** `/Users/cdavis/github/translator/2026/transcription/reads/A/p034.json`

**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/A/p034.json`
→ `1 ok, 0 failed`, exit 0, no warnings.

## Counts

| Item | Count |
|---|---|
| Body lines (total) | 31 |
| — paragraph 1 (large type, continues_prev) | 16 |
| — paragraph 2 (ANNOTAT. XIII, small type, continues_next) | 15 |
| Paragraphs | 2 |
| Headings | 1 (`ANNOTAT. XIII.`, spaced caps) |
| Markers in body | 1 (`{a}`) |
| Margin notes | 1 (key `a`) |
| Foot notes | 0 |
| Ornaments | 0 |
| `uncertain[]` entries | 8 |

Page furniture: `running_head` = `ARREST DV` (spaced caps), `folio` = `34`,
`signature` = null, `catchword` = null. No foot citation block; `foot.jpg` shows only the
last two body lines, the tail of the margin note, and blank paper below.

## Structure

1. `blocks[0]` — paragraph, 16 lines, `continues_prev: true` (opens mid-word, `ſtance,` =
   the tail of a word broken from p033), `continues_next: false` (ends `les coups.`).
2. `blocks[1]` — heading `ANNOTAT. XIII.`, spaced capitals, `spaced_caps: true`.
3. `blocks[2]` — paragraph, 15 lines, `continues_prev: false` (new indented paragraph
   beginning `Grande eſt l'amitié…`), `continues_next: true` (ends `auoit`, running on to
   p035).

Marker/note cross-check: the single body marker `{a}` (in `blocks[2].lines[10]`, after
`de ſon mari`) has a matching `margin_notes` entry with key `a`; there are no notes without
markers. The alphabet restarts at `a` here, consistent with the start of a new annotation
(ANNOTAT. XIII); no preceding page was available as context.

## `uncertain[]` entries (8)

1. `blocks[0].lines[1]` — `quantaux` set tight; normalized to `quant aux` (§1). Colon in
   `bien:mais` verified at 4x (two dots), normalized to `bien: mais`.
2. `blocks[0].lines[4]` — `enhaine` set tight; normalized to `en haine` (§1). Mark after
   `entendre` verified at 3x as a round baseline dot → period, not comma.
3. `blocks[0].lines[6]` — `poſsibles` verified at 4x: long s + short round s, **not** `ſſ`.
4. `blocks[0].lines[14]` — `lepouuant` set tight; normalized to `le pouuant` (§1).
5. `blocks[0].lines[15]` — the `o` of `receuoir` is barely inked (faint outline only);
   reading secure from spacing and sense, but the glyph is nearly absent on the page.
6. `blocks[2].lines[10]` — `deſon` set tight; normalized to `de ſon` (§1). Line-end break
   sign after `Hyperme` is a blobbed short dash, transcribed `-`.
7. `blocks[2].lines[11]` — `auſsi` verified at 4x: long s + short round s, **not** `ſſ`.
8. `blocks[2].lines[2]` — line ends `The` with no hyphen (continues `ſalie` next line);
   recorded only so the reconciler does not insert one.

## Explicit checks performed

- **`ſſ` vs `ſs`** — every double-s on the page zoomed to 4x or more:
  `poſsibles` (ſs), `auſsi` (ſs); `eſſay`, `aſſailli`, `aſſom-`, `deſſus` (all true `ſſ`,
  two tall long s's). These two error classes are the likeliest A/B divergence on this page.
- **Punctuation at clause boundaries** — zoomed and shape-checked, not inferred from sense:
  `entendre.` (period), `comptes;` (semicolon: dot + tail), `bien:` and `amis:` and
  `eſpoux:` (colons), `d'Admetus,` (comma: tail below baseline), `mari {a}.` (period),
  `Satyre.` (period in the margin note).
- **Tildes** — `auroiẽt`, `moyẽs`, `l'auroyẽt`, `deuãt`, `tãt`, `ſãs`, `cõme` all confirmed
  as tildes, distinguished from the acute of `recherché` / `guetté` / `ſanté` / `tuée`.
- **u/v and i/j as printed** — `vne`, `vn`, `auoir`-type forms, `iuſtice`, `iadis`,
  `iuſqu'au`, `Iuuenal` (margin, as `Iu-` / `uenal`), `vj` all kept as set.

## Notes for the reconciler

- The page is clean: no damage, no tears, no gutter loss. Ink is even in the large type;
  the small annotation type is slightly lighter, and a few letters are under-inked
  (`receuoir`'s `o` is the worst; the `t` of `en terre` has a broken top serif, and the `e`
  of `ce,` in `blocks[0].lines[4]` is light). None of these change the reading.
- `pages/strips/p034/body-9.jpg` is blank paper — the annotation runs to the last line and
  there is no foot block, signature or catchword.
- The margin note sits low, beside body line 11 of the annotation paragraph rather than
  level with the top of the paragraph; `beside_line` records the line it starts against.
  Its second line is set with a wide internal gap (`la      tragedie`), normalized to one
  space.
- Proper names worth a second eye: `de Rols` (twice, round final s), `Alceſtis` /
  `Alceſte`, `Admetus`, `Apollo`, `Danaus`, `Lyncée` (confirmed `n`, not `o`), `Hypermeſtra`
  (broken `Hyperme-` / `ſtra` across lines 11–12).
- Print oddities kept as set, per §1/§7, with no `uncertain[]` entry: `a la mort` without
  grave (`blocks[2].lines[1]`), `ou l'euſt` without grave (`blocks[0].lines[12]`), and the
  lower-case `&` sentence openings `perdre. & le premier` and `& à ces fins`.
