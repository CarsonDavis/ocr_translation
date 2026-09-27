# read-p039-B

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/B/p039.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/B/p039.json` → `1 ok, 0 failed`, exit 0, no warnings.

## Counts

| | |
|---|---|
| body lines | 36 (7 + 29), plus 1 heading line = 37 lines of type in the column |
| paragraphs | 2 (blocks[0] end of the TEXTE, 7 lines; blocks[2] the annotation, 29 lines) |
| headings | 1 (`ANNOTAT. XXI.`, spaced capitals) |
| markers | 2 (`{a}` on blocks[2].lines[17], `{b}` on blocks[2].lines[20]) |
| margin notes | 2 (keys `a`, `b`) — every marker has a note, every note a marker |
| foot notes | 0 (no foot citation block) |
| running head | `PARLEMINT DE THOLOSE.` (sic — see below) |
| folio | `39` (matches the manifest) |
| signature | `C iiij` |
| catchword | none |
| ornaments | none |
| `uncertain[]` entries | 22 |

## Layout

Recto. The page opens mid-sentence with the last seven lines of the TEXTE, set one size
larger than the rest; then the centred display line `ANNOTAT. XXI.` in spaced capitals;
then a fresh indented paragraph of annotation running to the foot of the page and on to
p040. Both margin citations sit low in the outer (right) margin, beside the last third of
the annotation. The signature `C iiij` stands alone below the last body line; the bottom
third of the leaf is otherwise blank.

## `uncertain[]` entries — one line each

1. `blocks` — counts for the reconciler (36 body lines, 2 paragraphs, 1 heading, 2 markers, 2 notes, no foot block/catchword).
2. `blocks[0]` — page quality: clean, well inked, in focus; nothing lost to damage or the gutter; the facing-page strip runs along the LEFT edge on this recto and was ignored.
3. `blocks[0]` — `continues_prev`/`continues_next` reasoning for both paragraphs (p038 not yet transcribed, so the opening flag rests on the sense).
4. `running_head` — **ESCALATED.** `PARLEMINT` sic: the sort between M and N is a capital I, not an E (symmetric serifs, no arms), verified at 14x against the E of `PARLE` and `DE`; the final E of `THOLOSE` is a true E. Compare p007's `PVRLIMENT DE TOLOSE,`.
5. `blocks[0].lines[5]` — **ESCALATED.** `perſũaſible`: a nasal bar is printed over the u; nonsensical in the word, so a wrong/stray sort, transcribed as printed.
6. `blocks[0].lines[0]` — `abordé:` is a colon (two dots at 4.5x), the space before it removed per §1.
7. `blocks[0].lines[4]` — `fait:` colon confirmed under a brown diagonal stain that crosses into the `&`; note also `fait` here vs `faict` on lines[1].
8. `blocks[2].lines[0]` — the low, thick line-end sign after `nume` is the ordinary hyphen sort (compared at 6x with `certai-`, `cele-`, `ad-`), not a point.
9. `blocks[2].lines[1]` — `roſité` accent is partly inked and could be read grave; `donnoiẽt` carries a distinctly wavy nasal tilde, not an acute.
10. `blocks[2].lines[2]` — `au Iuges` sic (for `aux Iuges`).
11. `blocks[2].lines[5]` — `paſſees` is a true `ſſ` ligature; `y a:` is a colon, space supplied after it.
12. `blocks[2].lines[6]` — `commiſſaires` and `poſſibles` both true `ſſ`; `tousmoyens` set tight, normalized to two words (with the other tight pairs listed).
13. `blocks[2].lines[11]` — the point after `interrogué` is faint but real at 4.5x.
14. `blocks[2].lines[14]` — `Themi`/`ſtocles`, `pou`/`uoit`, `ſou`/`uent` all break without a hyphen; normal for this print.
15. `blocks[2].lines[17]` — the `{a}` marker is a small italic a set tight between `brees` and `ſi`; `l'iſſue` is a true `ſſ`.
16. `blocks[2].lines[18]` — `eſplẽdeur` sic (straight nasal bar over the e).
17. `blocks[2].lines[27]` — `de fons` sic (for `de fond`).
18. `margin_notes[0].lines[1]` — **ESCALATED.** `i. des Tuſcula.` ends in a square baseline point, not the margin fount's wider raised hyphen (compare `au ſe-`, `Ora-` at 9x), although the word runs on to `nes`; same case as p018's `Pline au li.` + `ure`.
19. `margin_notes[0].lines[5]` — `vij`, `xxiiij`, `xij` are all the italic single `ij` sort, transcribed per §2 and the p018/p044 house practice.
20. `margin_notes[0].lines[4]` — `Ora-`+`tore` = *De oratore*; points set tight, spaces supplied per §1.
21. `margin_notes[1].lines[0]` — **ESCALATED.** `ſoit veue l'an`: the u is worn and could be read as n (`vene`); sense (`soit veue l'annotat. xij`) and the cross-reference favour `veue`.
22. `margin_notes` — placement: both notes sit one to two lines above the body lines that carry their markers, so `beside_line` is approximate to within a line.

## For the reconciler

- The running head is the biggest single risk of a silent A/B divergence: reader A will
  very likely normalize it to `PARLEMENT`. The sort really is an I.
- The three other escalations are all single-sort calls: the nasal bar in `perſũaſible`,
  point-vs-hyphen after `Tuſcula`, and `veue`/`vene` in note b.
- No damage, no trimming loss, no faint passage. The one physical intrusion is a brown
  diagonal stain across the colon in `fait: &` (blocks[0].lines[4]); it is dirt, not type.
- The margin fount distinguishes its line-end hyphen (a wide raised stroke) from its
  points (small square dots) clearly at 9x; that distinction drove two readings here.
