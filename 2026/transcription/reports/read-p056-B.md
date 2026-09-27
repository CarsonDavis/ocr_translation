# Reader report — p056, reader B

**Output path:** `/Users/cdavis/github/translator/2026/transcription/reads/B/p056.json`

**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/B/p056.json`
→ `1 ok, 0 failed`, exit 0, no warnings (no NFC problem, no long-s normalization warning,
marker/note cross-check clean).

## Counts

| | |
|---|---|
| body lines (total printed lines in the body column) | 38 |
| — `blocks[0]` paragraph | 34 |
| — `blocks[1]` heading `TEXTE.` | 1 |
| — `blocks[2]` paragraph (the TEXTE excerpt) | 3 |
| paragraphs | 2 |
| headings | 1 (`TEXTE.`, spaced capitals) |
| markers in the body | 1 (`{a}`, `blocks[0].lines[17]`) |
| margin notes | 1 (key `a`, 4 lines) |
| foot notes | 0 |
| ornaments / decorated initials | 0 |
| signature | none |
| catchword | none |

`running_head`: `ARREST DV.`  `folio`: `56` (matches the manifest)

## Structure

- `blocks[0]` — one paragraph of 34 lines, `continues_prev: true` (opens mid-sentence,
  "tellement difficile, que pluſieurs ont penſé…"), `continues_next: false` (ends
  "ni malade." with a full point). It carries the single marker `{a}` and, at lines 28–29,
  an italic Latin tag (*Et locus, & tempus poſtulãt, vt paucis rem abſoluamus*) that runs
  on **inside** the prose — it is not set apart, so it stays in the paragraph per §6.
- `blocks[1]` — the display line `TEXTE.`, centred, spaced capitals closed up, `spaced_caps: true`.
- `blocks[2]` — the TEXTE excerpt, 3 lines in a larger roman, `continues_prev: false`
  (new sentence), `continues_next: true` (breaks off at "en tous vices:").
- `margin_notes[0]` — key `a`, beside `& amis {a}. Par extreme viellieſſe, …`:
  `Pline au li.` / `vij. c. xiiij. So-` / `lin en ſon Po` / `lihiſt. c. vij.`
- Below the last body line the leaf is blank: no foot citation block, no signature,
  no catchword.

## `uncertain[]` entries — 24 (5 flagged `escalate: true`), one line each

1. `blocks` — counts for the reconciler (38 body lines, 1 marker, 1 note, no foot block).
2. `blocks[0]` — page quality: clean, evenly inked, in focus; no damage, nothing lost to the
   gutter; the facing-page strip along the right edge of the crops ignored throughout.
3. `running_head` — **a point IS printed after DV** (verified at 10x); every other verso in
   `transcription/final` records `ARREST DV` without one, and p020/p042 say so explicitly.
4. `folio` — `56`, outer left of the head line, old-style 5; matches the manifest.
5. `blocks[0]` — `continues_prev`/`continues_next` reasoning for both paragraph blocks.
6. `lines[1]` — sic, wrong sort: **`caractaires`** for *caracteres* (clear a + dotted i at 4x).
7. `lines[2]` — (a) a small baseline speck in the wide space of "vray ˙ ſemblable", read as a
   speck/risen space, **not** a point; (b) the print sets `qu' vn` with a word space after the
   apostrophe, kept as one space. **Escalated.**
8. `lines[4]` — `qu'ou` printed without the grave the sense wants; not emended (§2).
9. `lines[5]` — the hyphen after `vieil-` is printed; the word runs on as `vieillieſſe` (sic).
10. `lines[6]` — the `ſſ` vs `ſs` check: every double-s on the page (`lieſſe`, `Meſſale`,
    `viellieſſe`, `vieilleſſe`) is long s + long s at 5x; no `ſs` spelling occurs here.
11. `lines[17]` — sic: **`viellieſſe`** (dotted i between the double l and the e); not emended.
12. `lines[24]` — `a` without grave; `Siẽne` (nasal mark read as tilde); `)qu'il` set tight,
    space supplied outside the parenthesis; line ends `Paragra` with no hyphen.
13. `lines[25]` — sic, **WRONG SORT: `ſçanoit` for `ſçauoit`** — the sort is an n (closed
    shoulder at 5x), compared directly against the true u of `ſçauoit` on lines[7] of the same
    block; a second baseline speck sits in the space after it. **Escalated.**
14. `lines[26]` — line ends `vou` with no word-break hyphen (as does lines[24]); none printed.
15. `lines[28]` — the italic Latin tag: swash italic ampersands → `&`, italic `vt`, tilde in
    `poſtulãt`; indented but part of the prose, so not a heading.
16. `blocks[1]` — `TEXTE.` is one display line in spaced capitals, closed up, point printed.
17. `blocks[2].lines[0]` — the compositor sets **`qua trieme`** with a thin space inside the
    single word; closed up to `quatrieme` per §1. **Escalated** (reader A may split it).
18. `blocks[2]` — larger roman, first line indented, new sentence, breaks off at the foot.
19. `margin_notes[0]` — the note's key letter is over-inked to a near-solid blob; read as `a`
    from its outline and from the unambiguous body marker. **Escalated** (see below).
20. `margin_notes[0].lines[1]` — **`xiiij`**: one x plus four dotted minims (14), not `xxiiij`
    (24), at 14x; noted because the usual citation for Pliny on memory is vii. 24.
    **Escalated.**
21. `margin_notes[0].lines[2]` — ends `Po` with no hyphen, running on to `lihiſt.` (Solinus,
    *Polyhistor*).
22. `margin_notes[0].lines[3]` — `lihiſt. c. vij.` set tight with a `ſt` ligature; spaces after
    the points supplied per §1, ligature decomposed.
23. `foot_notes` — no foot citation block, no signature, no catchword; the lower area of the
    leaf is blank paper with show-through only.
24. `blocks[0].lines[2]` — long-s check recorded: `toutesfois` is printed with a **round** s
    before the f (§2), so nothing was normalized; `Baſcouz` has long s.

## What the reconciler should know

- **Running head.** This verso prints `ARREST DV.` **with a terminal point.** All 23 versos
  already in `transcription/final` are recorded as `ARREST DV`, and two of them note that no
  point is printed. Either this forme differs, or the earlier pages dropped a point that is
  actually there. Worth one deliberate decision rather than a silent diff.
- **Marker alphabet.** The page carries exactly one marker, `a`, and one note. With p055 not
  yet transcribed, continuity cannot be checked. A lone `a` suggests the alphabet restarts at
  this ANNOTAT. section; confirm against p055 before finalising.
- **`xiiij` vs `xxiiij`.** Only one `x` is printed in the Pliny citation. The sense (Pliny,
  *Nat. hist.* vii. 24, the memory chapter, where Messala Corvinus appears) would prefer
  xxiiij, but the print has xiiij. Transcribed as printed; a spot-check at the original
  resolution would settle it.
- **Two baseline specks.** Small dark marks sit in wide word spaces on `lines[2]`
  ("vray · ſemblable") and `lines[25]` ("ſçanoit · il"). Both are smaller and less round than
  the page's printed points, and both fall in unusually wide spaces, so I read them as risen
  spaces or ink specks. If reader A read either as a period the diff is expected there.
- **`qua trieme`.** A real space inside one word in the TEXTE block. I closed it up under §1;
  the rule does not address a gap *inside* a word explicitly, so this is a judgement call.
- **Three wrong sorts / sic spellings, all transcribed as printed, none emended:**
  `caractaires` (lines[1]), `viellieſſe` (lines[17]), `ſçanoit` (lines[25]). The last is the
  classic n-for-u this print makes about once a page, and it was checked against a correct
  `ſçauoit` on the same page.
- **No hyphens are missing.** `Paragra`/`phe` (lines[24]/[25]), `vou`/`lut` (lines[26]/[27])
  and `Po`/`lihiſt.` in the margin all break without a hyphen, which is normal here (§1).
- Layout is otherwise plain: no damage, no faint ink, no decorated initial, no foot block,
  and the lower quarter of the leaf is blank.
