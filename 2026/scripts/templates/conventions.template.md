# Transcription conventions

<!-- TEMPLATE. Made by scripts/new_book.py from the Coras conventions (2026/docs/conventions.md),
with that book's facts taken out. Before any reader runs: check every rule below against
this book's print (look at five or six pages of pages/read/ through an Opus agent, never in
the coordinator), replace each `TODO` with this book's examples (quote real lines with their
page id), delete rules that cannot apply, and add any sign the print uses that is not listed.
Rules marked MECHANICAL are enforced by scripts/validate_page.py; change them only together
with the validator. Delete this comment when the file is done. -->

These rules govern the **diplomatic master transcription** of {{AUTHOR}}, *{{TITLE}}*
({{EDITION}}). The master reproduces what is printed, line by line. It is never
modernized. Readers follow these rules exactly; the validator enforces the mechanical ones;
the translator and the reviewers judge the rest.

## 1. Unit of transcription: the printed line

- One JSON string per printed line, in reading order, inside its block. Never join two
  printed lines into one string, never split one printed line into two.
- A line ending in a hyphen keeps the hyphen; the next line starts with the continuation.
  Do not resolve hyphenation. Transcribe any line-end word-break sign (a short double
  stroke included) as a single `-`. Example from this print: TODO (`"…"` last line of pNNN,
  `"…"` first line of pNNN).
- Whitespace inside a line is normalized to single spaces. Do not reproduce the
  compositor's spacing. Leading and trailing spaces are removed. (MECHANICAL)
- **Punctuation spacing is normalized too**: no space before `, . : ; ? !`; exactly one
  space after them when a letter, digit or `&` follows, even where the print sets them
  tight or with a space before. Parentheses: a space outside, none inside. `&` gets a
  space on both sides. Apostrophes and hyphens are left as they are. (MECHANICAL)
- **Word division is normalized**: distinct words are always separated by one space,
  however tight the compositor set them. Judging whether a hairline gap "counts" is not the
  reader's job. The only exception is a genuine single word or contraction of the period,
  which stays as printed. This print's examples: TODO.
- Line-end punctuation stays on its line.
- If this print breaks words at the line end **without** a hyphen, say so here: transcribe
  exactly what is there; a missing hyphen needs no `uncertain[]` entry. Likewise *sic*
  spellings need no entry unless the reading itself is in doubt. TODO: does it?

## 2. Letters and signs

One row per sign the print uses. Keep the rows that apply, add the ones this print needs,
and give this book's own example words. The decisions in the right-hand column are the
house rules; the reviewers and translators read this section to understand the French.

| Printed | Transcribe as | Notes |
|---|---|---|
| long s | `ſ` (U+017F) | Where the print uses it. If unsure whether a glyph is `ſ` or `f`, decide by the word. Note it in `uncertain[]` only if the word itself is ambiguous. Examples: TODO |
| round/short `s` | `s` | word-final and where printed. |
| `u` / `v` | **as printed** | Never normalize. This print's habit: TODO |
| `i` / `j` | **as printed** | Never normalize. This print's habit: TODO |
| `æ`, `œ` | `æ`, `œ` | Keep the ligature. |
| `ct`, `ſt`, `ſſ`, `ff`, `fi` ligatures | the plain letters | Decompose. |
| tilde over a vowel (= omitted `n` or `m`) | `ã ẽ ĩ õ ũ` (capitals `Ã Ẽ Ĩ Õ Ũ`) | Precomposed characters. Do not expand. |
| tilde over a consonant (`q̃`, `p̃`, `m̃`) | letter + U+0303 combining tilde | |
| `ꝑ`, `ꝓ`, `ꝰ`, `ꝫ` and other abbreviation signs | the Unicode character | List the ones this print uses: TODO |
| the paragraph sign in citations | as printed | Record here which sort this print uses for `§`: TODO |
| `&` | `&` | Always. |
| accents | as printed | Do not add accents the print lacks. |
| apostrophe | `'` (U+0027) | |
| virgule `/` used as comma | `/` | Only if actually printed. |
| superscript letters in abbreviations | plain letters | Note as `[abbr: …]` only if a sign cannot be typed. |
| Roman numerals | as printed | Including lowercase `iij`, `vij`. |
| Arabic numerals | as printed | |
| TODO: italic-font letter shapes that look like other letters | the letter they are | e.g. a curled italic `t` is a `t`. |

**Rule of thumb:** if a glyph exists in Unicode, use it; if the print's form cannot be
represented, write `[abbr: description]` in the line and add an `uncertain[]` entry.

## 3. Capitals, headings, spaced type

- Capitalization exactly as printed. **Small capitals** are transcribed as ordinary
  capitals; add an `uncertain[]` note `"small capitals"` if it matters.
- **Spaced capitals** (letters set with visible space between them, as in many running
  heads) are transcribed **closed up**, as ordinary words. The block gets
  `"spaced_caps": true`. Digits set apart are likewise closed up. Do not close up two
  separate words that happen to be single capitals.
- Section headings are their own block, `{"type": "heading", "text": "…"}`, transcribed
  with the punctuation printed. This book's section headings: TODO (they must match
  `headings` in `book.json`, which is what the stitch cuts sections at).
- A decorated initial letter is transcribed as the plain capital it represents. The block
  gets an `ornament` entry (see §6).

## 4. Letter markers and marginal notes

- Small letters in the body text that key to marginal citations are **markers**.
  Transcribe each as `{x}` at the exact point where the letter sits in the line, keeping
  the surrounding punctuation in its printed order. Spacing around a marker is normalized:
  one space before `{x}`, and any punctuation that follows is attached (`word {c}.`).
- The marginal note for marker `x` goes in `margin_notes` with `"key": "x"`. Its `lines`
  are its own printed lines, in order, same rules as body lines. Do not mark italics.
- Marker letters follow the printer's alphabet: TODO (list it; early printers often skip
  `j`, `v` and `w`, and print the letter s as long s, keyed `ſ`). Keys are lowercase
  letters, optionally followed by a digit. (MECHANICAL)
- Say whether **the alphabet runs continuously across pages** and where it restarts:
  TODO. When the alphabet restarts on the same page, the second `a` is keyed `a2`, the
  third `a3`, both in the body and in the note.
- Notes usually start beside the line holding their marker, but may drift down when the
  margin is crowded. `beside_line` (optional) is the body line the note's first line is
  printed beside, copied verbatim.
- When the margin overflows, citations may continue as a **small-type block at the foot
  of the page**. Those go in `foot_notes` with their keys, one entry per citation.
- A marker with no visible note (or a note with no visible marker) is recorded anyway,
  and an `uncertain[]` entry explains what is missing. (MECHANICAL)
- Marginal notes not keyed by a letter get `"key": null` and `beside_line` filled in.

## 5. Page furniture

- `running_head`: the head line without the folio number, as printed and closed up.
  This book's running heads: TODO (versos / rectos). Printer's errors are kept as printed.
- `folio`: the printed page number as a string, exactly as printed even if wrong. `null`
  if there is none. It must equal the manifest's `folio`, or an `uncertain[]` entry must
  say why not. (MECHANICAL)
- `signature`: the gathering mark at the foot of some rectos, else `null`.
- `catchword`: the word printed alone at the foot right anticipating the next page, else
  `null`.
- `ornaments`: free-text list, e.g. `["woodcut headpiece", "decorated initial A, 6 lines"]`.

## 6. Blocks

`blocks` is the body column top to bottom, excluding running head, folio, signature,
catchword, and the foot citation block. Types:

- `heading`: the section headings, the title-page lines, and any other **display line**
  (a line set in larger type, capitals, or italic as a title rather than as running
  prose). **One heading block per printed display line**, in order. Never merge display
  lines into one string and never put them in a paragraph block.
- `paragraph`: `lines` plus `continues_prev` (true when the first line continues a
  sentence or word from the previous page or from before the heading) and
  `continues_next` (true when the last line does not end the paragraph). A paragraph that
  begins with an indent or a decorated initial is a new paragraph.
- **Verse and other quotations** set apart in italic are `paragraph` blocks, one printed
  line per string, never headings.
- `ornament`: a woodcut or rule with no text; `text` describes it.
- `blank`: only for a deliberately empty area *within* the text column worth noting (rare).
  Empty space below the last line of a page gets **no** block.

A paragraph interrupted by a heading is two blocks with the heading between.

## 7. Uncertainty

- A character you cannot read: `[?]`. Several: `[??]`. A span lost to damage or the
  gutter: `[...]`.
- A reading you can make but doubt: transcribe your best reading and add to `uncertain[]`
  `{"where": "blocks[2].lines[7]", "text": "the line as transcribed", "note": "…"}`.
- Every `[?]`, `[...]`, `[abbr: …]` must have an `uncertain[]` entry. (MECHANICAL)
- Never silently guess. Never "correct" the print: misprints (wrong sorts, turned letters,
  wrong folio) are transcribed as printed and noted in `uncertain[]` with `"note": "sic"`.

## 8. What the validator checks mechanically

- Schema (`scripts/page_schema.json`).
- Every `{x}` marker in body lines has a `margin_notes` or `foot_notes` entry with key `x`,
  and vice versa, unless an `uncertain[]` entry mentions that key.
- No line contains a newline, a tab, or leading/trailing space.
- No `[?]`/`[...]`/`[abbr:` without an `uncertain[]` entry.
- `folio` equals the manifest's expected printed folio, or an `uncertain[]` entry says why.
- Forbidden normalizations: words the print would set with long s, written without it,
  are flagged as a **warning** (`scripts/validate_page.py` `LONG_S_WORDS`, a French list;
  replace it for a book in another language).

## 9. Worked example

TODO: one page of this book (page id, image, side), transcribed in the format above, with
the authoritative text named as `transcription/final/<id>.json` once it exists. Show one
paragraph with a marker, its margin note (without the note's own leading key letter, which
is the `key`, not part of `lines`), and the furniture.
