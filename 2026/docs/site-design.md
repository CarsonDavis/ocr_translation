# Site design: translations.codebycarson.com

Approved 2026-09-21. Supersedes design.md §6 and plan Tasks 11–12 where they differ
(hosting, multi-book layout, theme, partial data). The viewer behaviour in Task 12 still
applies and is restated in §4.

## 1. Goal

A live, cheap, static site at `https://translations.codebycarson.com/` that hosts one or
more translated early-printed books. The first is Jean de Coras, *Arrest memorable*
(1572), at `/martin-guerre/`. Each page shows the scan on the left and, on the right,
the English translation (with the marginal citations as sidenotes) or the diplomatic
French transcription. The transcription and translation are still being produced, so
the site must render pages whose layers are missing and be redeployable in one click as
data arrives.

## 2. Hosting (code-by-carson CDK stack, `cdk/stacks/portfolio_stack.py`)

Add, following the llms pattern exactly:

- `TranslationsBucket`: S3, block public access, SSL enforced, versioned, RETAIN,
  90-day noncurrent-version expiry.
- `TranslationsDistribution`: CloudFront, OAC origin, the same directory-index rewrite
  CloudFront Function (`/foo/` → `/foo/index.html`), `CACHING_OPTIMIZED`, compression
  on, redirect to HTTPS, TLS 1.2 2021, `PRICE_CLASS_100`.
- Domain `translations.codebycarson.com`: a context flag `attach_translations_domain`
  in `cdk.json` gates the certificate SAN, `domain_names`, `certificate`, and a Route 53
  A alias, the same way `attach_llms_domain` does. Adding a SAN replaces the ACM
  certificate; validation is DNS via the hosted zone, as before.
- Deploy role: `grant_read_write` on the bucket and `cloudfront:CreateInvalidation` on
  the distribution.
- Outputs: `TranslationsBucketName`, `TranslationsDistributionId`.

Images live in the same bucket under `<book>/img/<id>.webp`, uploaded once from
Carson's machine (§3.3). The deploy workflow never writes or deletes under `*/img/`.

Expected cost: S3 storage ~170 MB and CloudFront within the free tier; rounds to $0.

## 3. Repositories

### 3.1 code-by-carson (`master`)

```
translations/
  landing/index.html, style.css     root page: one card per book
  viewer/index.html, app.js, style.css   the generic viewer, no framework/build/external scripts
  viewer/data -> (gitignored symlink for local dev)
  viewer/img  -> (gitignored symlink for local dev)
  README.md                         how to run locally, how to add a book
.github/workflows/deploy-translations.yml
```

`deploy-translations.yml` (triggers: push to `master` touching `translations/**` or the
workflow; `workflow_dispatch`):

1. Checkout code-by-carson; checkout `CarsonDavis/ocr_translation` (`main`) into `book-src`.
2. Assemble `out/`: `translations/landing/*` → `out/`; `translations/viewer/*` →
   `out/martin-guerre/`; `book-src/2026/site/data/` → `out/martin-guerre/data/`.
3. OIDC credentials; resolve the two stack outputs with the same retry loop deploy-llms uses.
4. `aws s3 sync out/ s3://$BUCKET/ --delete --exclude "*/img/*"`, in two passes: JS/CSS
   `max-age=3600`; HTML and JSON `max-age=300, must-revalidate`.
5. Invalidate `/*`.

Adding a second book = another checkout step, another copy line, another landing card.

Docs to update in code-by-carson per its CLAUDE.md: `INDEX.md`, `docs/IMPLEMENTATION_LOG.md`,
`docs/REQUIREMENTS.md`.

### 3.2 ocr_translation/2026 (this repo)

```
site/data/book.json, index.json, pages/<id>.json   committed, generated
site/img/<id>.webp                                 gitignored, generated
scripts/split_pages.py                             manifest + finals (+ translation sections) -> site/data
scripts/site_images.py                             pages/full/*.jpg -> site/img/*.webp; prints/executes the upload
scripts/tests/test_split.py
docs/site-design.md (this), docs/site-handoff.md
```

`.gitignore` gains `2026/site/img/`. `pages/read/` stays as is (pipeline-owned).

### 3.3 Images

Source: `pages/full/<id>.jpg` (the full-resolution crops, e.g. 2805×3962; `raw/` is the
uncropped scan with scanner border). No resampling. Encode WebP quality 70, method 6,
with `cwebp` (installed) or Pillow. Measured on p004: 1009 KB. Whole book ≈ 165 MB.

Upload: `aws s3 sync site/img/ s3://<bucket>/martin-guerre/img/ --cache-control
"public, max-age=31536000, immutable" --content-type image/webp`, run by Carson with the
correct AWS profile after the stack exists. The script prints the command; it does not
run it unless passed `--upload --profile <name>`.

## 4. Data contract (per book)

The viewer reads only these files, relative to the book path.

`book.json`
```json
{"slug": "martin-guerre",
 "title": "Arrest memorable du Parlement de Tholose",
 "short_title": "Martin Guerre", "author": "Jean de Coras", "year": 1572,
 "description": "one paragraph for the landing card",
 "source": {"name": "Cambridge University Library", "item_url": "https://cudl.lib.cam.ac.uk/view/PR-MONTAIGNE-00001-00007-00022/1",
            "license": "CC BY-NC 4.0", "license_url": "https://creativecommons.org/licenses/by-nc/4.0/"},
 "images": {"base": "img/", "ext": ".webp", "width": 2805},
 "layers": [{"code": "en", "label": "English"}, {"code": "fr", "label": "French"}],
 "default_layer": "en",
 "stylesheet": null,
 "first_page": "p000-title"}
```
`stylesheet`, if set, is a per-book CSS file the viewer loads after its own. `images.base`
may be absolute (a CDN URL) or relative; it is the single image base constant.

`index.json`
```json
{"pages": [{"id": "p004", "page": 4, "folio": "4", "heading": "TEXT",
            "layers": {"fr": true, "en": false}}, …]}
```
Order is reading order. `heading` is the first heading on the page or null.

`pages/<id>.json` (the handoff §5 shape, with these rules)
- `english` and `french` are each either an array of blocks or `null` (pending).
- `french_notes` is `[]` when none, `null` when the layer is pending.
- Block types: `heading {text, spaced_caps?}`, `paragraph` (English: `{html, continued?,
  notes[]}`; French: `{lines[], continues_prev?, continues_next?, spaced_caps?}`),
  `ornament {text}`, `blank`. Any other `type` renders its `text` (or joined `lines`)
  in a `<div class="blk blk-<type>">`, so a book can add types via its stylesheet.
- English markers: `<sup class="mk" data-key="a">a</sup>` inside `html`; French markers:
  `{a}` inside a line. Note keys match `french_notes[].key` / `notes[].key`.
- `uncertain[]`: `{where: "blocks[i].lines[j]", note}` → dotted underline + title on that line.
- `source`: `{kind: "cudl"|"gallica", image_no, url}`; `folio` printed, `page` true.

`split_pages.py` produces all three from `manifest.json`, `transcription/final/*.json`,
and (when present) `translation/sections/*.md` split at `⟦pNNN⟧` markers; a page whose
final or translation is missing gets `null` for that layer and `false` in `index.json`.
The English path is built and tested now against a synthetic fixture, since no
translations exist yet. Heading normalization (`ANNOTAT. V.` → `ANNOTATION V`,
`TEXTE.` → `TEXT`), `continued` detection, and note attachment per plan Task 11.

## 5. Viewer behaviour

- **Routing**: `/martin-guerre/#p043`; no hash → `book.first_page`. Prev/next buttons,
  ← → keys, jump box accepting a page number or `title` / `argument`; the jump menu
  lists pages with a mark on those that have the current layer.
- **Left panel**: the page image fit to panel height (`object-fit: contain`); click
  toggles zoom; drag (mouse) and touch pan via CSS transform. Next page's image is
  prefetched.
- **Right panel**: headings and paragraphs. A paragraph with notes is a two-column grid
  at ≥1100px (text ~62ch, notes ~18rem), each sidenote aligned to the top of its
  paragraph, stacked in marker order; narrower, markers are tappable and the note
  expands inline. Hovering a marker highlights its note and vice versa. A `continued`
  paragraph shows a faint "⋯ continued" mark. Sidenote = citation, original French in a
  smaller italic line, gloss if present.
- **Layer toggle**: `English | French`. French mode = one printed line per line, serif,
  `spaced_caps` via `letter-spacing`, notes in the sidenote column keyed by letter,
  uncertain lines dotted. A layer that is `null` has its toggle disabled with
  "translation pending" / "transcription pending"; the page opens in the first available
  layer; if none, the text panel shows "Transcription in progress" and the image alone.
- **Header**: book short title (links to `/`), `p. 43`, `printed as 24` when folio
  differs, `image 63 · Cambridge University Library` linking to CUDL (Gallica line for
  p041), layer toggle, theme toggle. **Footer**: license credit and item link from `book.json`.
- **Theme**: dark by default; sun/moon toggle stored in `localStorage`; `data-theme` on
  `<html>`; palette matches codebycarson.com's dark minimalist look. Light theme complete.
- **Fonts**: system stacks. French: `"Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif`. English body: the same serif stack. UI chrome: system sans (`-apple-system, "Segoe UI", Roboto, sans-serif`).
  No external requests of any kind.
- **Responsive**: works at 390px (panels stack, image above text, zoom still works).
- **Landing page**: title, one-paragraph intro, a card per book (cover thumbnail = title
  page image, title, author/year, progress line "N of 162 pages transcribed, M translated"
  computed from `index.json` at load), dark, same palette.

## 6. Verification

- `uv run --with pytest,jsonschema pytest scripts/tests -q` passes (existing 128 + new).
- `uv run python scripts/split_pages.py` → `162 pages written`; every `pages/<id>.json`
  validates against a small JSON schema in `scripts/site_schema.json`.
- Viewer checked in a real browser (Playwright) on `p000-title`, `p004`, `p041`,
  `p159`, and one pending page, at 1440px and 390px, both themes; screenshots in
  `docs/checks/site-*.png`, reviewed by Carson before the infra phase.
- `cd cdk && npx cdk synth` succeeds with the new resources; diff reviewed by Carson.
- After deploy: `curl -I` on the root, `/martin-guerre/`, one page JSON, one image
  (expect 200, `image/webp`, immutable cache header).

## 7. Order of work and rules

1. Split script, schema, tests, data (this repo).
2. Viewer + landing in code-by-carson, run locally against this repo's data and images.
3. Images to WebP; CDK + workflow changes; `cdk synth`; docs in code-by-carson.
4. Carson: commit/push both repos, deploy, run the image upload, verify live.
5. Blog: replace "you can find it here" in
   `CarsonDavis.github.io/_posts/2025-03-09-troublesome_translations.md` (line 214) with a
   link to `https://translations.codebycarson.com/martin-guerre/`. Optional, on request:
   a Live Sites card on codebycarson.com.

Rules from the handoff still apply: no commits or `git add` without Carson's say-so; no
AI attribution; never write `manifest.json`, finals, or `pages/**`; append a dated
section to `docs/pipeline-log.md` for site work. Orchestration: Fable plans and
reviews; Opus subagents write code, one phase at a time.
