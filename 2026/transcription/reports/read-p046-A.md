# Read report — p046, reader A (opus)

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/A/p046.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/A/p046.json` →
`1 ok, 0 failed`, exit 0, no warnings (no long-s warnings raised).

## Counts

| thing | n |
|---|---|
| body lines (paragraph lines) | 30 |
| paragraphs | 3 |
| headings | 2 (`ANNOTAT. XXV.`, `TEXTE.`, both spaced caps) |
| markers in the body | 3 (`{a}`, `{b}`, `{c}`) |
| margin notes | 2 (13 printed margin lines total) |
| foot notes | 0 |
| ornaments | 0 |
| `uncertain[]` entries | 18 (5 flagged `escalate: true`) |

## Layout

Verso. Folio `46` top left, running head `ARREST DV` in spaced capitals. No signature,
no catchword, no ornament, no decorated initial.

Top to bottom: a 9-line paragraph in the large TEXTE type continuing the sentence from
the previous page (`verificatiõ & recognoiſſãce dudit priſon | nier, …`); the heading
`ANNOTAT. XXV.`; a 9-line annotation in the small type carrying markers `a`, `b`, `c`;
the heading `TEXTE.`; a 12-line paragraph in the large type that runs on to the next page
(`… qu'il y a ſi`). The margin column holds one 13-line block of italic citations, all of
it in `margin-2.jpg`; `margin-1`, `margin-3` and `margin-4` are blank paper. `foot.jpg`
shows the last two body lines and then blank paper to the edge — no foot citation block.

Page condition is good: even inking, no damage, no tears, no bleed-through worth noting.
The crops keep a strip of the facing recto along the right (gutter) edge; ignored.

## `uncertain[]` entries (18)

Escalated (5):

1. `margin_notes[1]` — **the second margin block has no printed key letter.** Recorded
   with `key: null`. Its opening glyph was compared at 1200% with the `c.` of `c. dudũ`
   (note a, line 1), the `c.` of `c. mandata` (same block, line 4) and the italic `b` of
   `barbaris` (line 10): the three c's are identical crescents with no ascender, each
   followed by a period, and the `b` has an unmistakable tall ascender — so the block does
   not open with `b`. The whole 13-line column is set solid at a uniform ~64 px leading
   (measured from a row-ink profile), so the note boundary is inferred from the two short
   lines instead: `cauſ.` ends note a at about a third of the measure and `præſum` ends
   this block short. The block stands beside `Et quant au Preuenu…`, two body lines below
   marker c and six below marker b, so it could serve either.
2. `blocks[2].lines[5]` — marker `b` has no note with key `b` anywhere on the page.
3. `blocks[2].lines[8]` — marker `c` has no note with key `c` anywhere on the page. The
   marker itself is secure (small raised italic c + baseline period after `reputé`).
4. `blocks[4].lines[3]` — punctuation after `Martin Guerre`. Read as a **period**. At 900%
   it is a round blob centred on the baseline with no hook, against the commas after
   `ouys` and `Tilh` which hook well below the baseline, and matching the periods after
   `enfance.` and `Rols.`. Sense would prefer a comma; not decided from sense.
5. `blocks[2].lines[4]` — the word-break sign after `con` is a small **raised round dot at
   mid x-height**, not the thick horizontal bar used for every other line-end hyphen on
   the page (`reſu-`, `vertueu-`, `mar-`, `de-`, `l'a-`) and not on the baseline where the
   periods sit. Word plainly broken (`con-|tre`), so transcribed as a single `-` per §1.

Not escalated (13):

6. `blocks[0].lines[6]` — `toutle` printed with no space; normalized to `tout le` (§1).
7. `blocks[2].lines[1]` — `deRols` printed with no space; normalized to `de Rols` (§1).
   Both marks over the e's (`entẽdoit`, `riẽ`) are flat bars, i.e. tildes, not acutes
   (compared with the acute of `meſchancé`).
8. `blocks[0].lines[0]` — `recognoiſſãce` is `ſſ` (both glyphs tall, hooked) at 8x; the
   line ends `priſon` flush with no hyphen, continuing as `nier,`.
9. `blocks[4].lines[6]` — no punctuation printed after `auoit` although `D'autres,` opens
   a new sentence on the next line.
10. `blocks[4].lines[10]` — the line end reads `des teſ`: round s on `des`, hairline gap,
    then `teſ` with long ſ (the word breaks to `moins` with no hyphen). Colon after
    `Berçeau`.
11. `blocks[3]` — a speck between the first E and the X of `TEXTE.`; it sits above the
    baseline unlike the heading's own period, so read as dirt and not transcribed.
12. `margin_notes[1].lines[2]` — `remilit.` (= de re militari) is one tight token; kept
    closed up under §1's contraction exception. Same for `proſo.` (= pro socio).
13. `margin_notes[0].lines[3]` — the leading `q` carries no period (unlike the `j.` above).
14. `blocks[0].lines[2]` — `confrontez.` is a baseline dot with no tail: period, not comma.
    `fame` printed with one m.
15. `blocks[2].lines[6]` — `paſſée` is `ſſ`; the same check was made on `cognoiſſent`,
    `aſſeurent` and `recongnoiſſent`. No `ſs` spelling occurs on this page.
16. `blocks[4].lines[0]` — `quantau` and `ily` set with no gap; normalized (§1).
17. `blocks[2].lines[0]` — long-s audit: every word that would take a long s is printed and
    transcribed with `ſ`; the remaining short s's are genuinely word-final.
18. `blocks[0]` — page furniture: folio, running head, no signature/catchword/ornament,
    no foot block; facing-page strip in `foot.jpg` ignored.

## For the reconciler

- The **big open question is the margin**: three body markers, one printed key. Either the
  page prints only note `a` and the rest is an unkeyed continuation, or the compositor set
  notes `b`/`c` without their key letters. I could not resolve it from the image and did
  not invent a key. A human should decide whether `margin_notes[1]` gets `b`, `c`, or
  stays `null`.
- The marker alphabet restarts at `a` on this page (the previous annotation, ANNOT. XXIIII
  on p044, also ran `a`–`c`), consistent with a restart at each annotation.
- Two punctuation calls are likely diff points with reader B: `Martin Guerre.` (period, not
  comma) and the raised dot read as the hyphen of `con-`.
- Three tight settings were normalized per §1 and will read as spaces in the output:
  `toutle` → `tout le`, `deRols` → `de Rols`, `quantau`/`ily` → `quant au`/`il y`.
