# Site data contract

What `scripts/split_pages.py` reads and what it writes.

```
manifest.json                  the page list, in reading order      (pipeline session)
transcription/final/<id>.json  the French layer                     (pipeline session)
text/sections.json             the section index                    (scripts/stitch_text.py)
translation/sections/<id>.md   the English layer                    (translation session)
        |
        |  uv run --with jsonschema python scripts/split_pages.py
        v
site/data/book.json  site/data/index.json  site/data/pages/<id>.json
```

Run it after any input changes. It never writes the inputs. Every page record is built
and validated against `scripts/site_schema.json` before anything is written, so a run
that fails leaves the previous `site/data` in place.

**`scripts/prompts/translate.md` is the authority on the section file format** — it is
what the translator is told to produce, and `scripts/check_markers.py` is what checks it.
This document describes that format as the site reads it; where they ever disagree, the
prompt wins and this script follows.

## 1. `text/sections.json` — the section index

Written by `scripts/stitch_text.py` from the finished transcriptions. The site uses it
for three things: which sections exist, **what order they come in (the list order)**, and
what heading each one prints.

```json
{"generated": "…",
 "sections": [
  {"id": "annot-001", "kind": "annotation", "number": 1, "label": "ANNOTATION I.",
   "pages": ["p002", "p003", "p004"],
   "text": "⟦p002⟧Les mariages ainſi contractez… {a}…",
   "notes": [{"key": "a", "page": "p002", "text": "Chap. dernier au titre de frigid…"}],
   "starts_mid_page": false, "ends_mid_page": true, "complete": true}]}
```

`kind` is `title`, `argument`, `texte` or `annotation`. The site reads `id`, `kind`,
`number` and the list order; `text`, `notes` and the rest are the translator's inputs.

**Headings.** The stitch step lifts the printed heading blocks out of the section text,
so neither the French `text` nor the English file contains `TEXTE.` or `ANNOTAT. V.`. The
site therefore synthesizes one heading block at the head of each section, from `label` —
the heading as the book prints it — normalized the same way the jump menu normalizes the
French (`TEXTE.` → `TEXT`, `ANNOTAT. IIII.` → `ANNOTATION IIII`, keeping the printed
numeral); a section with no `label` is named from its kind and number instead (`texte` →
`TEXT`, `annotation` → `ANNOTATION ` + the roman numeral of `number`).
`title` and `argument` get none — their display lines carry their own titles. If
the translator does write the heading as its own paragraph, that paragraph becomes the
heading block and nothing is synthesized on top of it.

## 2. `translation/sections/<id>.md` — one translated section

The format is `scripts/prompts/translate.md` §"Output format (exactly)". A whole real
section, `texte-02.md`, trimmed in the middle:

```
---
id: texte-02
pages: [p004, p005]
---
⟦p004⟧With whom she had lived nine or ten years, and by his doings begotten a son
cal⟦p005⟧led Sanxi, still living: but for some slight theft of wheat… absented himself.

## Notes
- (none: this section carries no marginal citations.)
```

and one real note line, from `annot-003.md`:

```
- {a} (p008): **Plautus, *Amphitryon*** — Plaute en ſon Amphytrio. [Coras's "first comedy":
  *Amphitruo* stands first in the alphabetical order of the plays…]
```

**Front matter** is required: `id` must equal the file name, and `pages` lists the page
ids the section covers. Either missing, or an `id` that does not match the file name, is
an error naming the file.

**Page markers** are `⟦pNNN⟧`, the page id exactly as the manifest spells it
(`⟦p000-title⟧`), which is what `check_markers.py` matches. The prompt requires every
marker of the French, in the same order, at the corresponding point in the English — so a
marker often falls **mid-word** (`cal⟦p005⟧led`, `wo⟦p004⟧man`). The site splits the prose
there and keeps both halves as written: the part before the marker ends the earlier page,
the part after opens the later one with `continued: true`.

- The first marker must precede any prose in the file.
- A marker at the start of a paragraph opens a fresh paragraph on the new page.
- A page may carry the tail of one section and the head of the next; the next section's
  heading goes between them. Once the stream has moved past a page, no later section may
  mark it again, and a marker for a page that is not in the manifest is an error.
- A page that no translated section marks gets `english: null` — the viewer shows it as
  pending. A section listed in `sections.json` with no file under `translation/sections/`
  is simply not translated yet; that is normal and silent.

**Prose.** Paragraphs are separated by blank lines; single newlines inside a paragraph are
joined. `{a}` is a note marker, kept exactly where the French prints it (keys run `a`…`z`,
restart as `a2`, and include `ſ`). `*text*` is italic. Everything else is plain text and
is HTML-escaped, so `&` reaches the viewer as `&amp;`.

**Notes** follow a `## Notes` line, one entry per line:

```
- {key} (pNNN): **citation, which may contain *italics*** — original French [optional gloss]
```

- The `(pNNN)` scopes the note to a page, so the same key may be used again on another
  page of the same section; each page's notes are its own.
- The citation is the bold run; `*x*` inside it becomes `<i>x</i>`, and so it does in the
  gloss. The original French is kept literally, as transcribed.
- The gloss is the bracketed trailer, and may be absent (then it is `null`).
- `- (none: …)` records that a section has no marginal citations.
- An aside may sit between the page and the citation — `- {t} (p007) — orphan note, no
  marker in the body: **Digest 34.5.9** — L. qui duos` — and the block may open with a
  paragraph addressed to the reviewer, as when the print's own marker alphabet is
  mis-set. Both are for the reviewer; the site does not show them. A line that has no
  bold citation at all is an error naming the file and the line.
- A note is attached to the paragraph on its page that carries its `{key}`. A note whose
  key is nowhere in that page's prose — which happens where the print omits a marker — is
  attached to the page's first paragraph, and the run says so on stderr.

## 3. `site/data/pages/<id>.json`

The schema is `scripts/site_schema.json`; it is enforced on every run.

```json
{"id": "p004", "page": 4, "folio": "4", "side": "verso",
 "image": "p004",
 "source": {"kind": "cudl", "image_no": 26,
            "url": "https://cudl.lib.cam.ac.uk/view/PR-MONTAIGNE-00001-00007-00022/26"},
 "running_head": "ARREST DV", "prev": "p003", "next": "p005",
 "english": [
   {"type": "paragraph", "html": "man <sup class=\"mk\" data-key=\"ſ\">ſ</sup>. Especially…",
    "continued": true,
    "notes": [{"key": "ſ", "citation": "Digest 23.2.4 (<i>De ritu nuptiarum</i>)",
               "original": "L. minorem D. de rit. nup", "gloss": "…"}]},
   {"type": "heading", "text": "TEXT"},
   {"type": "paragraph", "html": "With whom she had lived nine or ten years…",
    "continued": false, "notes": []}
 ],
 "french": [
   {"type": "paragraph", "lines": ["…"], "continues_prev": true, "continues_next": false,
    "spaced_caps": true},
   {"type": "heading", "text": "TEXTE.", "spaced_caps": true},
   {"type": "ornament", "text": "woodcut headpiece"}
 ],
 "french_notes": [{"key": "ſ", "kind": "margin", "lines": ["L. minorem", "…"]},
                  {"key": "q", "kind": "foot", "lines": ["…"]}],
 "uncertain": [{"where": "blocks[1].lines[3]", "text": "…", "note": "…"}]}
```

- `page` is the true page number, `null` for the front matter (`p000-title`,
  `p000-argument`), which is addressed by name. `folio` is what the page prints, which
  may disagree with `page` and may be `null`.
- `image` is the page id; the viewer builds `book.images.base + image + book.images.ext`.
- `source.kind` is `cudl` or `gallica`. p041 is missing from the Cambridge scan and comes
  from Gallica, with `image_no: null`.
- `english` and `french` are each an array of blocks, or `null` while that layer is
  pending. `french` is the final's `blocks` verbatim; `french_notes` is its margin notes
  then its foot notes (`beside_line` dropped), `[]` when there are none and `null` when
  the French layer is pending.
- `uncertain` is the final's entries with `where`, `text` and `note` (the reader's
  `escalate` flag is dropped); `[]` when there is no final.
- Block types: `heading {text}`, `paragraph` (English `{html, continued, notes[]}`,
  French `{lines[], continues_prev, continues_next, spaced_caps?}`), `ornament {text}`,
  `blank`. A viewer renders an unknown type's `text`, or its joined `lines`, in a
  `<div class="blk blk-<type>">`.
- Everything is NFC, written with `ensure_ascii=False` so the long s and the tildes are
  readable in the file.

## 4. `site/data/index.json`

```json
{"pages": [{"id": "p004", "page": 4, "folio": "4", "heading": "TEXT",
            "layers": {"fr": true, "en": true}}]}
```

Reading order, one record per manifest page. `heading` is what the jump menu shows, taken
from the **French** page: the first heading on it that is not `TEXT` (so a page that opens
with the quoted *TEXTE.* and then carries `ANNOTAT. VI.` is listed as `ANNOTATION VI`),
falling back to the first heading, `null` for the front matter and for a page with no
heading or no French layer. `layers` says which layers the page actually has, so the
viewer can disable a toggle and the landing card can count progress.

## 5. `site/data/book.json`

```json
{"slug": "martin-guerre",
 "title": "Arrest memorable du Parlement de Tholose",
 "short_title": "Martin Guerre", "author": "Jean de Coras", "year": 1572,
 "description": "one paragraph for the landing card",
 "source": {"name": "Cambridge University Library",
            "item_url": "https://cudl.lib.cam.ac.uk/view/PR-MONTAIGNE-00001-00007-00022/1",
            "license": "CC BY-NC 4.0",
            "license_url": "https://creativecommons.org/licenses/by-nc/4.0/"},
 "images": {"base": "img/", "ext": ".webp", "width": 2805},
 "layers": [{"code": "en", "label": "English"}, {"code": "fr", "label": "French"}],
 "default_layer": "en",
 "stylesheet": null,
 "first_page": "p000-title"}
```

`images.base` may be relative or an absolute CDN URL; it is the one image base constant.
`stylesheet`, when set, is a per-book CSS file the viewer loads after its own.

## 6. Running it

```
uv run --with jsonschema python scripts/split_pages.py          # -> 2026/site/data
uv run --with jsonschema python scripts/split_pages.py --root DIR --out DIR
uv run --with pytest,jsonschema pytest scripts/tests/test_split.py -q
```

It prints `162 pages written (31 french, 13 english)`. Warnings — a note with no marker
on its page, a final marked done but unusable — go to stderr. Anything that would produce
data the viewer cannot trust (a marker for a page that does not exist, a page claimed by
two sections, a note line that does not parse, a record that fails the schema) raises and
writes nothing.
