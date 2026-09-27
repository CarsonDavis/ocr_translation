# Transcription conventions

These rules govern the **diplomatic master transcription** of Coras, *Arrest memorable du
Parlement de Tholose* (Paris, 1572). The master reproduces what is printed, line by line.
It is never modernized. Readers follow these rules exactly; the validator enforces the
mechanical ones; the reconciler and spot-checker judge the rest.

## 1. Unit of transcription: the printed line

- One JSON string per printed line, in reading order, inside its block. Never join two
  printed lines into one string, never split one printed line into two.
- A line ending in a hyphen keeps the hyphen: `"à la fem-"` (last line of page 3). The next
  line starts with the continuation: `"me {ſ}. meſme qu'en ceſt aage, …"` (first line of
  page 4). Do not resolve hyphenation. Some hyphens are printed as a short
  double stroke; transcribe any line-end word-break sign as a single `-`.
- Whitespace inside a line is normalized to single spaces. Do not try to reproduce the
  compositor's spacing. Leading and trailing spaces are removed.
- **Punctuation spacing is normalized too**: no space before `, . : ; ? !`; exactly one
  space after them when a letter, digit or `&` follows (`aage, on`; `L. i. parag.`;
  `xxxiij. q. j.`), even where the print sets them tight or with a space before. The
  validator enforces this. Parentheses: a space outside, none inside (`monde (qui`,
  `exemple) plei-`), whatever the print shows. `&` gets a space on both sides. Apostrophes
  and hyphens are left as they are.
- **Word division is normalized**: distinct words are always separated by one space, however
  tight the compositor set them (`on void`, `iuge de`, `conceut in-`, `de toile`). Judging
  whether a hairline gap "counts" is not the reader's job. The only exception is a genuine
  single word or contraction of the period (`ſondit`, `ledit`, `auſſi`, `pourtant`, `deſprit`
  without apostrophe), which stays as printed.
- Line-end punctuation stays on its line.
- This print often breaks a word at the line end **without** a hyphen (`nos Iu` / `riſcõſultes`).
  Transcribe exactly what is there; a missing hyphen is normal and needs **no** `uncertain[]`
  entry. Likewise `sic` spellings (`deſprit`, `on` for `ou`) need no entry unless the
  reading itself is in doubt.

## 2. Letters and signs

| Printed | Transcribe as | Notes |
|---|---|---|
| long s (looks like `f` without the full crossbar) | `ſ` (U+017F) | Used everywhere except usually at the end of a word and before `b`, `f`, `k`. If unsure whether a glyph is `ſ` or `f`, decide by the word: `ſon`, `faire`, `ſi`, `fils`. Note it in `uncertain[]` only if the word itself is ambiguous. |
| round/short `s` | `s` | word-final and where printed. |
| `u` / `v` | **as printed** | Typically `v` at word start (`vne`, `vray`), `u` inside (`auoir`, `trouuer`). Never normalize. |
| `i` / `j` | **as printed** | Almost always `i` (`iamais`, `iuge`, `Iean`). Never normalize. |
| `æ`, `œ` | `æ`, `œ` | Keep the ligature. |
| `ct`, `ſt`, `ſſ`, `ff`, `fi` ligatures | `ct`, `ſt`, `ſſ`, `ff`, `fi` | Decompose into the plain letters. |
| tilde over a vowel (= omitted `n` or `m`) | `ã ẽ ĩ õ ũ` | Precomposed characters. `Ã Ẽ Ĩ Õ Ũ` for capitals. Do not expand (`hõnesteté` stays, never `honnesteté`). |
| tilde over a consonant (`q̃` = que, `p̃`, `m̃`, `n̄`) | letter + U+0303 combining tilde | e.g. `q̃`. |
| `ꝑ` (p with stroke, = per/par), `ꝓ` (= pro), `ꝰ` (superscript us) | `ꝑ`, `ꝓ`, `ꝰ` | Rare in this roman type but present in citations. |
| the small 3-shaped `-que` sign after q (`quæcunqꝫ`) | `ꝫ` (U+A76B) | Keep the q; the sign follows it. |
| the paragraph sign in citations (`§`), which this print sets as a swash italic capital `P` | `P` as printed (e.g. `l. iij. P. eiuſdem`) | House ruling (p028–p049 reconciliations): transcribe the sort the print uses, not `§` or `Ꝑ`; a true `§` sort, where it occurs, is `§`. |
| `&` | `&` | Always. |
| `ß` or `ſs` | as printed | |
| accents: `é è à ù ç ï ë` | as printed | Acute is common on final `é`; grave on `à` and `où`. Do not add accents the print lacks (`a`, `ou` without grave are correct if printed so). Capitals carry no accents. |
| apostrophe | `'` (U+0027) | |
| hyphen inside a line (`c'eſt-à-dire` rarely) | `-` | |
| virgule `/` used as comma | `/` | Only if actually printed. |
| the "odd t" in the italic margin font (a `t` with a curled top) | `t` | It is an ordinary italic `t`. |
| superscript `e`, `r` in abbreviations (`S.`, `mre`) | plain letters | Note as `[abbr: …]` only if a sign cannot be typed. |
| Roman numerals | as printed | `IIII`, `XLIIII`, `CXI`, lowercase `iij`, `vij` in signatures and citations. |
| Arabic numerals | as printed | `1559`. |

**Rule of thumb:** if a glyph exists in Unicode, use it; if the print's form cannot be
represented, write `[abbr: description]` in the line and add an `uncertain[]` entry.

## 3. Capitals, headings, spaced type

- Capitalization exactly as printed, including `DE` and `DV` in heads. **Small capitals**
  (as in `ROLS` on page 44) are transcribed as ordinary capitals; add an `uncertain[]` note
  `"small capitals"` if it matters.
- **Spaced capitals** (letters set with visible space between them, as in the running head
  `A R R E S T   D V` or `P R AE T E X T A` in the body) are transcribed **closed up**, as
  ordinary words: `"ARREST DV"`, `"PRAETEXTA"`. The block gets `"spaced_caps": true` to
  record the letterspacing (for the running head, a heading, or a paragraph containing
  such a word). Whether tracking is wide enough to count is a judgment call; it changes
  only the flag, never the letters. Digits set apart (`1 5 6 0`) are likewise closed up.
  Do not close up two separate words that happen to be single capitals: `A M. Antoine`
  (the preposition *à* and the abbreviation *M.*) stays as two tokens.
- Section headings are their own block, `{"type": "heading", "text": "ANNOTAT. V."}` or
  `"TEXTE."`, transcribed with the punctuation printed.
- A decorated initial letter is transcribed as the plain capital it represents. The block
  gets an `ornament` entry (see §6).

## 4. Letter markers and marginal notes

- Small letters in the body text (often italic, sometimes raised) that key to marginal
  citations are **markers**. Transcribe each as `{x}` at the exact point where the letter
  sits in the line, keeping the surrounding punctuation in its printed order:
  `"à ſauteller, & bondir {u}. Et ſi le lecteur ne ſe contente,"`. Spacing around a
  marker is normalized: one space before `{x}`, and any punctuation that follows is
  attached (`Dieu {c}.`, `attaire {b}, voire`), whatever hair space the print shows.
- The marginal note for marker `x` goes in `margin_notes` with `"key": "x"`. Its `lines`
  are its own printed lines, in order, same rules as body lines (hyphens kept, `ſ`, etc.).
  The margin font is italic; do not mark italics.
- Marker letters follow the printer's alphabet: `a b c d e f g h i k l m n o p q r ſ t u x y z`
  (no `j`, no `v`, no `w`; **the letter s is printed as long s `ſ` and is transcribed as
  the key `ſ`**). If the print does use `v` or `j` as a key, transcribe the shape printed.
  **The alphabet runs continuously across pages** within a stretch of text (p002 a–f →
  p003 g–r; p009 a–d → p010 e–n → p011 o), restarting at `a` at some section boundaries. So
  a page's first marker is usually the letter after the previous page's last marker; use
  the preceding page's final transcription (given as context) to read a blotted marker.
  When the alphabet restarts on the same page, the second `a` is keyed `a2`, the third
  `a3`, both in the body and in the note.
- Notes usually start beside the line holding their marker, but may drift down when the
  margin is crowded. `beside_line` (optional) is the body line the note's first line is
  printed beside, copied verbatim; it helps placement and is not a key.
- When the margin overflows, citations continue as a **small-type block at the foot of the
  page** (below the last body line, sometimes in two columns). Those go in `foot_notes`
  with their keys, one entry per citation, lines as printed.
- A marker with no visible note (or a note with no visible marker) is recorded anyway, and
  an `uncertain[]` entry explains what is missing.
- Marginal notes that are **not** keyed by a letter (a bare reference beside a passage)
  get `"key": null` and `beside_line` filled in.

## 5. Page furniture

- `running_head`: the head line without the folio number, as printed and closed up
  (`"ARREST DV"` on versos, `"PARLEMENT DE THOLOSE."` on rectos; printer's errors such as
  `"ARRET DV"` or `"TOLOSE"` are kept as printed).
- `folio`: the printed page number as a string, exactly as printed even if wrong
  (`"24"` on true page 44). `null` if there is none.
- `signature`: the gathering mark at the foot of some rectos (`"A ij"`, `"F iij"`), else `null`.
- `catchword`: the word printed alone at the foot right anticipating the next page, else
  `null`.
- `ornaments`: free-text list, e.g. `["woodcut headpiece", "decorated initial A, 6 lines"]`.

## 6. Blocks

`blocks` is the body column top to bottom, excluding running head, folio, signature,
catchword, and the foot citation block. Types:

- `heading`: `TEXTE.`, `ANNOTAT. XII.`, the title-page lines, the `Argument` head, and
  any other **display line** (a line set in larger type, capitals, or italic as a title
  rather than as running prose). **One heading block per printed display line**, in order,
  even when several lines together form one title: the seven lines of the title on page 1
  are seven heading blocks. Never merge display lines into one string and never put them
  in a paragraph block.
- `paragraph`: `lines` plus `continues_prev` (true when the first line continues a
  sentence or word from the previous page or from before the heading) and `continues_next`
  (true when the last line does not end the paragraph). A paragraph that begins with an
  indent or a decorated initial is a new paragraph.
- **Verse and other quotations** set apart in italic (a Latin distich, a couplet) are
  `paragraph` blocks, one printed line per string, never headings: they are quoted text,
  not titles. A quotation that ends with a marker keeps the marker in its last line.
- `ornament`: a woodcut or rule with no text; `text` describes it.
- `blank`: use only for a deliberately empty area *within* the text column worth noting (rare). Empty space below the last line of a page (a section that ends early) gets **no** block.

A paragraph interrupted by a heading (`TEXTE.` in the middle of the page) is two blocks
with the heading between.

## 7. Uncertainty

- A character you cannot read: `[?]`. Several: `[??]`. A span lost to damage or the
  gutter: `[...]`.
- A reading you can make but doubt: transcribe your best reading and add to `uncertain[]`
  `{"where": "blocks[2].lines[7]", "text": "the line as transcribed", "note": "could be 'vne' or 'une'; ink faint"}`.
- Every `[?]`, `[...]`, `[abbr: …]` must have an `uncertain[]` entry.
- Never silently guess. Never "correct" the print: misprints (`Tholoſe` vs `Toloſe`,
  turned letters, wrong folio) are transcribed as printed and may be noted in `uncertain[]`
  with `"note": "sic"`.

## 8. What the validator checks mechanically

- Schema (`scripts/page_schema.json`).
- Every `{x}` marker in body lines has a `margin_notes` or `foot_notes` entry with key `x`,
  and vice versa, unless an `uncertain[]` entry mentions that key.
- No line contains a newline, a tab, or leading/trailing space.
- No `[?]`/`[...]`/`[abbr:` without an `uncertain[]` entry.
- `folio` equals the manifest's expected printed folio, or an `uncertain[]` entry says why not.
- Forbidden normalizations: any body line containing `ſ`-free words like `est`, `sont`,
  `aussi`, `sans` where the print would use long s is flagged as a **warning** (the print
  uses `eſt`, `ſont`, `auſſi`, `ſans`), so a reader that silently normalized long s is caught.

## 9. Worked example (page 18, image 40, verso; illustrative only, the readings below are approximate, the authoritative text is `transcription/final/p018.json`)

```
running_head: "ARREST DV"      folio: "18"
blocks[0] paragraph, continues_prev: true
  "chauffé, ſ'en va droit à ſa maiſon trouuer ſa femme, la"
  "iette ſur le lict, luy diſant qu'il la vouloit engroſſir d'vn"
  "diable. Ce qu'il fit, ou pour le moins d'vn fils qui eut la"
  "forme d'vn diabloton, & qui commença dés qu'il fut né"
  "à ſauteller, & bondir {u}. Et ſi le lecteur ne ſe contente,"
margin_notes[0] key "u"
  "u Loys Viues"    ← NO: the key letter itself is not part of the lines →
  "Loys Viues"
  "au xv. liur. de"
  "S. Aug. de la"
  "cité de Dieu."
```

The note's own leading key letter (`u`, `x`, `y` printed at the start of the first line of
the margin note) is **not** included in `lines`; it is the `key`.
