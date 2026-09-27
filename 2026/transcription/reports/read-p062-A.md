# Read report — p062, reader A

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/A/p062.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/A/p062.json` → `1 ok, 0 failed`, exit 0, no warnings.

Manifest record: `{"id": "p062", "page": 62, "image": 84, "side": "verso", "folio": "62", "source": "cudl"}`.
Running head `ARREST DV` (letterspaced, closed up), folio `62` — matches the manifest.

## Counts

| | |
|---|---|
| printed lines in the body column | 30 |
| body text lines (in paragraph blocks) | 28 |
| paragraph blocks | 4 (2 + 4 + 20 + 2 lines) |
| heading blocks | 2 (`TEXTE.`, `ANNOTAT. XLI.`) |
| markers `{x}` | 0 |
| margin notes | 0 |
| foot notes | 0 |
| ornaments | 0 |
| signature / catchword | none / none |
| `uncertain[]` entries | 19 (1 escalated) |

Block sequence, top to bottom:

0. paragraph, 2 lines, `continues_prev: true`, `continues_next: true` — the tail of the annotation carried over from p061, ending `quand il dit,`
1. paragraph, 4 lines, `continues_prev: true` — the italic Latin quotation from Q. Serenus Sammonicus (`Interdum exiſtit turpi verruca papilla:` … `Qui ſolus patriæ, cunctando reſtituit rem.`), set apart and indented, with the turn-over `hæſit,` on its own line
2. heading `TEXTE.` (`spaced_caps: true`)
3. paragraph, 20 lines, self-contained — the deposition about Pierre Guerre's conspiracy, ending `Martin Guerre, ſon nepueu.`
4. heading `ANNOTAT. XLI.` (`spaced_caps: true`)
5. paragraph, 2 lines, `continues_next: true` — the opening of Annotation XLI, running on to p063

## `uncertain[]` entries (19)

1. `blocks` — page quality: clean, sharp, evenly inked; pale damp-stain across the upper outer corner touching the folio and the first two lines without obscuring a letter; recto show-through visible but never confusable with type; nothing lost to damage or crop; facing-page strip along the gutter ignored.
2. `blocks` — counts for the reconciler (as tabulated above), and the note that the page carries three type sizes: small annotation fount, italic verse fount, large TEXTE fount.
3. `margin_notes` — the outer margin is blank on all four margin strips and there is no marker anywhere in the body, so the empty `margin_notes`/`foot_notes` are by observation, not omission.
4. `blocks[0]` — `continues_prev` rests on sense alone (p061 not yet transcribed); `continues_next` set true because the block runs into the Latin distich, but the reconciler may prefer false.
5. `blocks[1]` — the four Latin lines are quoted verse recorded as a paragraph block per §6, not as headings; the turn-over `hæſit,` kept as its own line.
6. `blocks[3].lines[0]` — **sic, wrong sort**: `teſmoius` for `teſmoins`; at 8x the sort after the dotted i is an unmistakable u, not an n.
7. `blocks[3].lines[8]` — `reffu-` is a genuine `ff`, not an `ſſ` misread: at 16x both sorts have a crossbar projecting on both sides and a baseline foot serif, unlike this fount's long s. Flagged because reader B is likely to write `reſſu-`.
8. `blocks[3].lines[15]` — a faint but real baseline point after `ladit` (abbreviating *ladite*); reader B may take it for a speck and write `ladit de`.
9. `blocks[5].lines[1]` — **ESCALATED**: the first sort of `eſtre` carries a solid hook descending below the baseline exactly where a cedilla sits, so the print may read `çſtre`. The bowl does appear to carry an e's mid-bar at 20x (compare the e of the following `de` and the bar-less c of `car`), which is why `eſtre` is given; but nothing prints below this line, so the descender cannot be an ascender from a line beneath. Needs a third eye.
10. `blocks[3].lines[3]` — punctuation checked by shape: semicolon after `mourir` (dot above a tailed comma), period after `priſonnier` despite the following lower-case `iuſqu'à`; same pattern at `lines[8]`.
11. `blocks[3].lines[2]` — `ſes femme & beaux fils` is what is printed (plural article governing both nouns); not emended to `ſa femme`.
12. `blocks[3].lines[7]` — the line ends `mou` with no word-break hyphen, running on to `rir`. Normal for this print; recorded so it is not read as an omission.
13. `blocks[3]` — spacing normalized per §1 at `Guerre,ſes`, `fils,de`, `priſonnier.iuſqu'à`, `reſte,pour`, `priſonnier.ce`, `ſauuer:car`, `parẽt,ain-`, `aſſeuré.En`, `outre,de-`, `Rols : &` (space **before** the colon), `Guerre,ſon`, plus the wide quads after `verrues.` and in `de ſa part ,`.
14. `blocks[3].lines[10]` — the mark over `parẽt` read as the nasal tilde, the same sort as in `cõiuration` and `veritablemẽt`.
15. `blocks[3].lines[16]` — `d'iceux` has an ordinary dotted i; at reading size the apostrophe plus dot can be mistaken for a diaeresis, but there is none.
16. `blocks[3].lines[9]` — `que il` is printed as two separate words, not `qu'il`.
17. `blocks[2]` — `TEXTE.` and `ANNOTAT. XLI.` are letterspaced capitals closed up per §3 with `spaced_caps: true`; the running head likewise; the folio sits at the left of the head line, not in the head.
18. `blocks[5]` — Annotation XLI opens at the foot with only two lines on this page, ending `car nous ſommes en` without a point; the lower third of the leaf is blank (no foot block, signature or catchword).
19. `blocks[3].lines[13]` — long-s audit: every `eſt`, `eſtoit`, `ceſte`, `reſte`, `pluſtoſt`, `aſſez`, `aſſeuré`, `ſommes` is printed with long s and transcribed with `ſ`; round s appears only word-finally (`vns`, `fils`, `ſes`, `quelques`, `pas`) and in the roman capital of `Serenus`.

## For the reconciler

- The only escalation is the possible cedilla in `eſtre` (blocks[5].lines[1]).
- Three readings are likely diff points against reader B, all deliberate and all checked at high magnification: `teſmoius` (sic), `reffu-` (`ff`, not `ſſ`), and the faint point in `ladit.`.
- Layout is otherwise simple and unusually clean for this book: no markers, no margin, no foot block. The page is a hinge — it closes Annotation XL's Latin quotation, gives the whole of one TEXTE section, and opens Annotation XLI — so the block sequence, not the line readings, is the thing most worth a second look.
