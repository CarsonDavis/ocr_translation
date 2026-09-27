# Reader report — p049, reader A

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/A/p049.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/A/p049.json` → `1 ok, 0 failed`, exit 0, no warnings.

## Counts

| item | count |
|---|---|
| body lines | 32 (14 in the TEXTE paragraph + 18 in the annotation) |
| paragraphs | 2 |
| headings | 2 (`TEXTE.`, `ANNOTAT. XXVII.`, both spaced capitals) |
| markers in body | 2 (`{a}`, `{b}`) |
| margin notes | 2 (keys `a`, `b`) |
| foot notes | 0 |
| ornaments | 0 |
| `uncertain[]` entries | 19 (2 escalated) |

Page furniture: running head `PARLEMENT DE TOLOSE.` (printer's TOLOSE, not THOLOSE), folio `49`, signature `D`, no catchword, no foot citation block.

## Structure

1. heading `TEXTE.`
2. paragraph, larger roman, 14 lines, `Dequoy eſt aiſé à recueillir, & enten-` … `tin Guerre.` — self-contained (`continues_prev` false, `continues_next` false)
3. heading `ANNOTAT. XXVII.`
4. paragraph, annotation type, 18 lines, `Où les teſmoins du demãdeur…` … `lable: car ces deux, de ſoy inſuffiſans, en font vn pour` — runs on to p050 (`continues_next` true)

Marker `a` sits after the comma in `…plus croyable, {a} voire` (body line 10 of the annotation); marker `b` after `…pour faire preuue {b}. Comme…` (line 13). Both have margin notes; both notes stand beside their own marker's line, no drift.

## `uncertain[]` entries, one line each

1. `blocks[2]` — the numeral of `ANNOTAT. XXVII.` is set with a wide gap before the last I; five sorts confirmed at 10x, closed up to XXVII per §3.
2. `margin_notes[1].lines[0]` — **ESCALATED**: the cluster ending note b's first line; read `au`, but a third faint ink form makes `aud` (and `ait`) possible. See below.
3. `margin_notes[1].lines[1]` — **ESCALATED**: an unidentifiable letter-sized descending mark after `eiuſdem`; transcribed `[?]`.
4. `margin_notes[0].lines[0]` — the over-inked paragraph sort read `§` for consistency with p045; numeral confirmed `iij` (i + italic ij-sort).
5. `margin_notes[0].lines[1]` — blotted `dem.`; very short title abbreviation `de te.`; a faint speck above the final point could make it a colon.
6. `margin_notes[0].lines[2]` — first sort read `c.` (capitulum), could be read `ſ.`
7. `margin_notes[0].lines[3]` — two faint specks between `lo` and `titul`, read as dirt, not a printed point.
8. `blocks[3].lines[0]` — `defendeu r` printed with a gap, `demãdeur,&` with none; both normalized; line ends `depo`, broken without a hyphen.
9. `blocks[3].lines[8]` — `ſimilitude :le` (space before the colon, none after) and `grand , ſans`, normalized per §1.
10. `blocks[3].lines[10]` — sic `le numeroſité` (masculine article), checked at 8x.
11. `blocks[3].lines[13]` — double-s check: `neceſſaires` is `ſſ`; likewise `aſſeuroient`; `inſuffiſans` has no double s (ſ + ff + i + ſ).
12. `blocks[3].lines[14]` — sic `ſen trouuent`, long s and no apostrophe.
13. `blocks[3].lines[16]` — the break sign after `va` is a short thick stroke that reads like a point; transcribed `-`.
14. `blocks[3].lines[17]` — last line of the page; tailed line-end r; `font` has a true f; tight settings normalized.
15. `blocks[1].lines[8]` — justification space before the point in `grandes . Le`, normalized; the mark is a period, not a comma.
16. `blocks[1].lines[3]` — `conflict` with the ct ligature confirmed (not `conflit`).
17. `blocks[1].lines[9]` — tilde over the a of `grãd`, not an acute; `premiere` printed unaccented.
18. `signature` — the foot holds a bare capital `D`; read as the gathering signature from the corpus pattern (A f.1, B f.17, C f.33 → D f.49), not a catchword.
19. `running_head` — spaced capitals closed up; printed `TOLOSE`, kept as printed.

## What the reconciler should know

- **The one real problem is note b, line 1.** The note plainly means *Accurse on the said § eiusdem*, keying back to note a's `l. iij. §. eiuſdem`, so the word should be `audit` ( = au dit) broken across the two lines. On the raw scan (`raw/img069.jpg`) the x-height band of that line, y 3266–3292, gives three ink runs: x 2238–2262 (w 24), x 2268–2280 (w 12), x 2282–2298 (w 16), the last carrying a thin faded stroke rising 11 px above x-height at columns 2290–2292 (true ascenders on that line — the key `b`, the `ſ` of `Accurſe` — rise 21 px). Reference widths from line 2 of the same note: d = 27, i = 13, t = 14, u = 27. I read `au`, treating x 2268–2298 (30 px) as the `u` and the raised stroke as a speck or a broken sort, because line 2 unambiguously begins `dit` (a full-ascender d with its bowl, a dotted i, a crossbarred t, all clear at 10–20x) and `aud` + `dit` would double the d. Reader B may read `aud`. A human eye on the original would settle it in seconds.
- **Note b's last mark** is genuinely unreadable in this scan: w 15, from x-height down to 6 px below the baseline, where the clean period on the same line is w 8 and sits entirely at the baseline. Comma, broken sort and `ꝫ` are all live options; I left `[?]`.
- Note a's citation resolves as *l. iij. § eiusdem, D. de te[stibus]; c. In nostra, illo titul.* — i.e. Dig. 22.5.3 on the weighing of witnesses plus the decretal *In nostra* under the same title, which fits the annotation exactly. That context was what let me read the very short `de te.` and the blotted `c.`
- The margin ink is heavier and more blotted on this page than on p044/p045: `dem`, the `c` of `c. in noſtra`, the `a` of note b and the key letters are all partly filled. The body column, by contrast, is clean and evenly inked throughout; nothing in it is damaged, faint, or lost to the gutter.
- The crops keep a slice of the facing verso along the left (inner) edge of `pages/read/p049.jpg` and `margin-1.jpg`; ignored.
- `body-8.jpg` is blank apart from the bottom of the signature `D`; the page's lower third is empty paper.
- No alphabet restart on this page: markers run `a`, `b`, so p050 should start at `c` unless a section boundary intervenes.
