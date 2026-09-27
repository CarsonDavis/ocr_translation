# Reader report — p060, reader B

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/B/p060.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/B/p060.json`
→ `1 ok, 0 failed`, exit 0, no warnings.

## Page furniture

- `running_head`: `ARREST DV` (spaced capitals)
- `folio`: `60` (matches the manifest's expected printed folio)
- `signature`: none (verso), `catchword`: none, `ornaments`: none, `foot_notes`: none

## Counts

| | |
|---|---|
| blocks | 4 — heading, paragraph, heading, paragraph |
| headings | 2 (`TEXTE.`, `ANNOTAT. XXXIX.`, both spaced capitals) |
| paragraphs | 2 |
| body lines | 32 total — 13 in the large TEXTE type, 19 in the smaller annotation type |
| markers | 5 (`a b c d e`) |
| margin notes | 5 (`a b c d e`), 15 note lines in all |
| foot notes | 0 |
| uncertain[] entries | 27 (6 with `escalate: true`) |

Marker/note cross-check: every `{x}` in the body has a `margin_notes` entry with
key `x`, and every note has a marker. The alphabet restarts at `a` with this
annotation and no earlier `a`–`e` occurs on the page, so the keys are plain
`a`…`e` (no `a2`). No preceding page was available as context, so the sequence
could not be checked against p059.

## uncertain[] entries — one line each

1. `blocks` — counts for the reconciler (as above).
2. `blocks[0]` — page quality: clean, sharp, well-inked type; pale brown stain in the blank head/outer margin touches no type; nothing lost to damage or the crop; facing-page strip along the gutter ignored throughout.
3. `blocks[1]` — `continues_prev` false (new sentence under the TEXTE heading), `continues_next` false (ends `ſœurs.`); `blocks[3].continues_next` true (ends `d'homici-`). No preceding-page context available.
4. `blocks[1].lines[2]` — **sic, wrong sorts**: `Preniierement` for *Premierement* — at 8x, `n` + two dotted `i`s (four stems, one arch, two dots), not `m`+`i`.
5. `blocks[1].lines[0]` — `auſsi` is long s + **round** s; by contrast `cognoiſſoyent` and `aſſeurance` are the `ſſ` ligature.
6. `blocks[1].lines[8]` — **sic**: `Cuerre` for *Guerre*, an unambiguous capital C.
7. `blocks[1].lines[4]` — `parlé:femmes` / `hõneſtes,ſ'il` set tight, spaces supplied (§1); `ſ'il` has a clear apostrophe.
8. `blocks[1].lines[5]` — `Gaſcogne:leſquelles` and `Rols:&` set tight, normalized per §1.
9. `blocks[1].lines[9]` — line ends `parfai` with no word-break hyphen (likewise `ail` on `blocks[3].lines[4]`); normal for this print.
10. `blocks[1].lines[11]` — `dõnee` read at 3.5x (two clear e's), not `dõnce`.
11. **(escalate)** `blocks[3].lines[1]` — printed `desperes` with no gap but with a **round** s, so `des` + `peres`; normalized to `des peres`. Reader A may write `deſperes` as one word.
12. `blocks[3].lines[6]` — wide space before the point in `ſeruent .` and around the comma in `premier , quand`, normalized; the final sorts of `ment`/`ſeruent` are `t` (crossbar + curved foot), not `r`; `ſ'agiroit` has a clear apostrophe.
13. `blocks[3].lines[7]` — **sic, wrong sort**: `ehoſe` for *choſe* — closed bowl with a mid-bar (an `e`), compared against the `c` of `ce` on the same line.
14. `blocks[3].lines[11]` — `partie`: the `a` is damaged/under-inked, its top curl detached; read as `a` from the bowl, foot serif and width.
15. **(escalate)** `blocks[3].lines[14]` — `parẽs`: a long horizontal nasal bar, not the short right-rising acute of `parenté`/`eſté` (compared side by side at 7x). Reader A may read `parés`.
16. **(escalate)** `blocks[3].lines[14]` — `comme'quand`: a real raised apostrophe sort stands between the two words with no space (same sort as in `qu'en`, `d'vn`); transcribed as printed although the sense wants plain `comme quand`.
17. `blocks[3].lines[8]` — markers: all five are small italic sorts set tight with their punctuation; spacing normalized per §4; keys plain `a`…`e`.
18. **(escalate)** `margin_notes[0].lines[0]` — `l 'parentes.`: a small solid mark at **ascender** height between `l` and `parentes`, not a baseline point (checked at 12x); notes b and c print the same abbreviation as a bare `l` with no point. Transcribed as an apostrophe; reader A will most likely read `l. parentes.` **This is the least certain reading on the page.**
19. `margin_notes[0].lines[1]` — `C.de teſti.` set tight, space after the point supplied (same for `tris.C.`, `cuſ.tut.`, `aud.c.literas`).
20. `margin_notes[2].lines[0]` — a raised word-break stroke is printed after `pre` (same mark as after `ma`, `pro` in note b); transcribed `-`.
21. **(escalate)** `margin_notes[2].lines[1]` — `alleguét.`: the sort before the point is the italic fount's odd `t` (curled top, **no descender**); an italic `z` would descend. Sense would prefer `alleguéz`, so reader A may differ.
22. `margin_notes[2].lines[2]` — the italic script ampersand transcribed `&` (as on p018); citation runs `Ac-curſe & Bar-tole`.
23. `margin_notes[2].lines[3]` — the single italic `ij` sort (two dots over one body), transcribed `ij`.
24. **(escalate)** `margin_notes[4].lines[0]` — `aud. c. literas`: the faintest, smallest citation on the page, at the limit of the scan; `ead.` (eadem) and a split `au d.` (au dit) are both possible for the same ink.
25. `margin_notes` — `beside_line` placement: notes a/b/c sit level with their markers, note d begins five lines below its marker, note e's single line falls **below the last body line**; a one-line disagreement here is not substantive.
26. `margin_notes` — the key letters are recorded as `key`, not in `lines` (§9); all five legible at 6x.
27. `foot_notes` — no foot citation block, no signature, no catchword, no rule; bottom third of the leaf blank.

(Entries 1–3 and 25–27 are structural notes rather than doubtful readings;
the substantive doubts are 4–24.)

## What the reconciler should know

- **Three wrong sorts on one page**, more than the usual one: `Preniierement`
  (blocks[1].lines[2]), `Cuerre` (blocks[1].lines[8]) and `ehoſe`
  (blocks[3].lines[7]). All three were checked glyph by glyph at 4–8x and are
  transcribed as printed. `Cuerre` is in the prompt's own list of known
  misprints in this book.
- **The `ſſ` / `ſs` distinction is live on this page**: `auſsi` (round s) against
  `cognoiſſoyent` and `aſſeurance` (ligature). If reader A has `auſſi`, the image
  decides against it.
- **Two printed sorts that do not belong**: the apostrophe inside
  `comme'quand` and the raised mark inside `l 'parentes.`. Both are real ink,
  both are almost certainly foul case, and both are places where a reader
  reading for sense would silently drop them.
- **`des peres` vs `deſperes`** is a genuine editorial fork (round s = two
  words under §1, but the words are set with no gap at all). Worth a ruling
  for the whole edition, since this print does it often.
- The annotation paragraph runs on to p061 (`d'homici-`), and no preceding page
  was transcribed, so `continues_prev` on the TEXTE paragraph rests on this
  page's own evidence (a TEXTE heading above a new sentence) rather than on p059.
- Layout is otherwise plain: running head + folio, TEXTE, 13 lines of large
  type, ANNOTAT. XXXIX., 19 lines of annotation, five citations down the outer
  margin, and a blank lower third with no foot block, signature or catchword.
