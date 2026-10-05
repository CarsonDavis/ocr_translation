# Sources data contract

How cited source texts are stored for the site, and how a citation points into them. Shared
by the corpus fetchers, the citation locator, and the viewer's source pane. Keep this file and
the code in step.

## Files

```
site/data/sources/index.json                 one entry per corpus
site/data/sources/<corpus>/<unit>.json       one unit = one small file the pane can fetch
site/data/citations.json                     every citation in the book, keyed by page and marker
```

## index.json

```json
{
  "corpora": [
    {
      "id": "digest",
      "title": "Digest of Justinian",
      "language": "la",
      "edition": "Mommsen–Krüger text as published at droitromain.univ-grenoble-alpes.fr",
      "license": "public domain (edition text); site courtesy of Université Grenoble Alpes",
      "attribution": "Text: droitromain.univ-grenoble-alpes.fr (Y. Lassard, A. Koptev)",
      "url": "https://droitromain.univ-grenoble-alpes.fr/",
      "unit_scheme": "book.title",
      "passage_scheme": "book.title.fragment[.paragraph]",
      "units": ["1.1", "1.2", "..."]
    }
  ]
}
```

Corpus ids: `digest`, `code`, `institutes`, `novels`, `vulgate`, `decretum`, `decretals` (Liber
Extra, X), `sext`, `clementines`, and one id per classical work, lower-case with hyphens
(`cicero-de-officiis`, `ovid-metamorphoses`, `augustine-de-civitate-dei`).

## Unit files

```json
{
  "corpus": "digest",
  "unit": "48.5",
  "title": "D. 48.5 Ad legem Iuliam de adulteriis coercendis",
  "passages": [
    {"id": "48.5.39.pr", "label": "D. 48.5.39 pr. (Papinianus)", "text": "…"},
    {"id": "48.5.39.4",  "label": "D. 48.5.39.4",                "text": "…"}
  ]
}
```

- `passages` are in reading order, one per smallest addressable unit (Digest/Code paragraph,
  Institutes paragraph, Novel chapter, Vulgate verse, canon, classical section or line group).
- `id` follows the corpus `passage_scheme`; `pr` is the principium. Text is plain Unicode, no
  markup, paragraphs separated by `\n\n`.
- Unit sizes: Digest and Code per book.title; Institutes per book; Novels per novel; Vulgate
  per book; Decretum per distinctio or causa (`D.10`, `C.33`); Decretals per book.title;
  classical per work-book (unit `1` for book 1) or per work if short.

## Passage schemes

| corpus | passage id | example |
|---|---|---|
| digest, code | `book.title.fragment[.par]` with `pr` | `48.5.39.4`, `9.9.29.pr` |
| institutes | `book.title[.par]` | `4.6.25` |
| novels | `novel.chapter[.par]` | `90.7` |
| vulgate | `Book chapter:verse` (Vulgate names: `1 Kings` = 1 Samuel; `Psalms` Vulgate numbering) | `Genesis 17:5` |
| decretum | `D.n c.n`, `C.n q.n c.n`, `De cons. D.n c.n` | `C.33 q.1 c.4` |
| decretals, sext, clementines | `book.title.chapter` | `4.15.7` |
| classical | `book.chapter[.section]` or `book.line` per work's convention | `1.10.33` |

Code numbering: Coras cites the medieval vulgate; store Krüger numbering and keep a
concordance `site/data/sources/code/concordance.json` (`{"9.9.30": "9.9.29"}`) for the few
titles where they differ. The locator applies it.

## citations.json

```json
{
  "p040:e": [
    {
      "ref": "Seneca the Elder, Controversiae, pref. 1 §§17–19",
      "corpus": "seneca-controversiae",
      "unit": "1",
      "passage": "1.pr.17",
      "passage_end": "1.pr.19",
      "status": "passage",
      "external_url": "https://…",
      "scan_url": null
    }
  ]
}
```

- Key is `<page>:<marker>` as in the finals (`p040:e`, `p072:a2`). A Notes entry that cites
  several works gets several objects.
- `status`: `passage` (unit + passage known and present in the unit file), `unit` (work and
  unit known, passage not), `work` (corpus known only), `scan` (no text; `scan_url` points at
  a page image at Gallica/MDZ/Google Books), `none`.
- `external_url` is the same passage at the public online edition, always set when known.

## Viewer behaviour

Clicking a sidenote with a `passage` or `unit` citation opens a right-hand source pane: fetch
the unit file, render its passages, scroll to and highlight `passage` (through
`passage_end`), show the corpus attribution and the external link. `scan` opens the scan
in a new tab. `none` does nothing extra. The pane never loads more than one unit file at a
time, and unit files stay under ~300 KB.

## Licensing

Only public-domain or openly licensed texts are stored. Each corpus entry in `index.json`
carries license and attribution; the About dialog lists them. CC BY-SA texts (Perseus) keep
their attribution and are not altered beyond normalisation.
