# read-p047-B

**Output:** `/Users/cdavis/github/translator/2026/transcription/reads/B/p047.json`
**Validator:** `uv run --with jsonschema python scripts/validate_page.py transcription/reads/B/p047.json` → `1 ok, 0 failed`, exit 0, no warnings.

## Counts

| item | count |
|---|---|
| body lines | 29 |
| paragraphs | 1 (`continues_prev: true`, `continues_next: true`) |
| headings | 0 |
| markers `{x}` | 0 |
| margin notes | 0 |
| foot notes | 0 |
| ornaments | 0 |
| `uncertain[]` entries | 18 |

Page furniture: `running_head` `"PARLEMENT DE THOLOSE."` (spaced capitals, closed up per §3), `folio` `"47"` (matches the manifest), `signature` null, `catchword` null.

## Layout

The whole body column is one continuous paragraph of 29 printed lines in the large text type — no heading, no decorated initial, no rule, no indent. `margin-1.jpg` … `margin-4.jpg` are completely blank (paper, foxing and a facing-page strip only), and `foot.jpg` shows nothing below the last body line: no foot citation block, no signature, no catchword. Consequently there is no letter marker anywhere in the body and no note of any kind. The marker alphabet that was running on p044 (last marker there: `c`, itself unresolved) does not continue here.

Note for the reconciler: the context page supplied was `p044`, three pages earlier; it does not adjoin p047, so the sentence and the broken word at the head of line 0 could not be checked against a preceding transcription. `continues_prev` is set true from the page itself (it opens mid-sentence, mid-clause).

## `uncertain[]` entries (18), one line each

1. `blocks[0]` — layout: one paragraph, 29 lines; margins blank, no foot block, no markers, no notes; p044's alphabet does not continue.
2. `running_head` — spaced capitals closed up; head spells THOLOSE while body line 15 spells `Toloſe`; both kept as printed.
3. `lines[5]` — **`trõt`**: letter-by-letter at 7x it is t-r-õ-t (glyphs 1 and 4 match the `t` of `fort` on line 9, not the taller right-hooked `f` of `diffe-`); sense wants *fort*/*tout* differentes, so probably a compositor error, transcribed sic.
4. `lines[4]` — `p̃uenu` (= preuenu): p with a tilde above the bowl (not `ꝑ`), transcribed p + U+0303; the marks after `Martin` and after `uenu` are colons (two dots at 6x).
5. `lines[23]` — **`aſsiſtans`**: long ſ + ROUND s at 8x; the only `ſs` on the page. Every other double-s is a true `ſſ` ligature — `aſſeurer` (1), `reſſẽble` (7), `reſſẽblẽt` (9), `aſſeurée` (21), `perſuadaſſent` (24); `ſuffiſãment` (27) is ſ-u-ff-i-ſ.
6. `lines[20]` — the mark after `preuenu` is a faint round baseline dot with no tail at 9x → read as a period; reader A may see no stop at all.
7. `lines[26]` — the mark after `l'õcle` is a round baseline dot, no tail, unlike the comma after `femme` on the same line → period, though the sense would prefer a comma.
8. `lines[9]` — the `l` of `reſſẽblẽt` is almost unprinted (broken dotted trace at 8x); word not in doubt. `martin` is lowercase here (sic) against `Martin` on lines 3, 4, 7.
9. `lines[17]` — sic `Cuerre` for *Guerre* (capital has no crossbar at 3x; compare `Guerre` on line 3); justification space before the comma normalized away.
10. `lines[18]` — sic: `lad` here carries **no** point (8x), unlike `lad.` on line 13 and `led.` on lines 10 and 20.
11. `lines[19]` — `enpleine` set solid; word division normalized to `en pleine` per §1. `aud` (= audit) ends the line with no point and no hyphen.
12. `lines[22]` — `plꝰ` (= plus): p, l and the raised 9-shaped -us sign, U+A770, per §2.
13. `lines[15]` — `laq̃lle` (= laquelle): q + U+0303, verified at 7x; `Toloſe` sic against the head's THOLOSE.
14. `blocks[0]` — tilde vs acute: all `-ent`/`-em-` contractions checked at 8–9x against known acutes (`rapporté`, `amplié`, `coſté`); tildes are wavy horizontals, acutes clean right-leaning strokes. Reads: roiẽt, reſẽ-, reſultẽt, rẽtes, reſſẽble, reſſẽblẽt, ſentẽce, cõdẽné (tilde over o AND first e, acute on the last), parlemẽt, prouidẽce, attẽdu, premieremẽt.
15. `blocks[0]` — spacing normalized per §1 at every tight/loose setting: space before comma on lines 3, 17, 25; no space after punctuation on lines 0, 4, 6, 22, 26, 28; words set solid on lines 1, 7, 8, 11, 16, 28. No letters changed.
16. `lines[0]` — `continues_prev`: opens mid-sentence; p044 is not the adjoining page so the join could not be verified. `grãde` is complete as printed.
17. `lines[28]` — last line: no point, no hyphen, sentence runs on → `continues_next` true; `in`|`ſtruite` is a word broken across lines 27/28 with no hyphen (normal for this print, §1).
18. `lines[1]` — `auſ`|`ſi` broken across lines 1/2 with no hyphen; likewise `deſ`|`quelles` (4/5) and `te`|`ſte` (11/12).

No `escalate: true` entries. No `[?]`, `[...]` or `[abbr: …]` in any line.

## For the reconciler

- The page is clean and well inked; no damage, no gutter loss, no bleed-through worth noting. Only two faint spots: the `l` of `reſſẽblẽt` (line 9) and the period after `preuenu` (line 20).
- The two likeliest A/B disagreements are **line 5 `trõt`** (a reader guided by sense will write `fort` or `tout`) and **line 23 `aſsiſtans`** (a reader who does not zoom will write `aſſiſtans`). Both were checked at 7–8x here.
- Secondary disagreement risks: the period after `preuenu` (line 20) and after `l'õcle` (line 26), and `lad` without a point (line 18).
- The gutter-side strip of the facing page visible on the crops was ignored throughout.
