# Viewer spec checklist

Every bullet of `docs/site-design.md` §5, checked against the viewer in
`code-by-carson/translations/viewer/` as of 2026-09-22, with the assertion that
evidences it. Suites (Playwright, Chromium, against `python3 -m http.server`):

| suite | what it covers | result |
| --- | --- | --- |
| `verify.js` | shell: routing, header, footer, jump box, layer tabs, theme | 72/72 |
| `verify5.js` | text panel: both layers, sidenotes, uncertain readings | 75/75 |
| `verify6.js` | image panel: fit, zoom, pan, prefetch, per-book stylesheet | 47/47 |
| `verify-deploy.js` | the assembled deploy tree: landing page + viewer at `/martin-guerre/` | 15/15 |

209 assertions, 0 failures, 0 console errors, 0 off-origin requests.

## §5 bullets

| # | bullet | result | evidence |
| --- | --- | --- | --- |
| 1 | **Routing** — `#p043`, no hash → `first_page`, prev/next, ← →, jump box, jump menu marks pages with the current layer | **pass** | `verify.js`: "no hash -> #p000-title", "ArrowRight -> next", "prev disabled at first page", jump `12`/`title`/`xyz`, "datalist has 162 options", "datalist option shape (p004, fr available)", "#p999 -> No such page" |
| 2 | **Left panel** — image fit to panel height, click toggles zoom, mouse drag and touch pan, next image prefetched | **pass** | `verify6.js`: "rest: image height ≈ panel height" (799 = 799), "click zooms", "drag pans the image", "390: touch pan moves the image", "next image prefetched after current finished"; `site-final-p004-en-1440.png`, `site-final-zoomed-1440.png` |
| 3 | **Right panel** — two-column paragraph+notes, aside aligned to its paragraph, marker order, tappable when narrow, hover highlights both ways, `⋯ continued`, citation / original / gloss | **pass** | `verify5.js`: "aside top aligns with paragraph top" (delta 0.00), "aside note order" (ſ t u x y z a b), "hover marker highlights note" and the reverse, "390 tap opens notes", "real English: every note rendered" (8/8); `site-final-p004-en-1440.png`, `site-final-p004-notes-open-390.png` |
| 4 | **Layer toggle** — English \| French, French diplomatic lines, notes keyed by letter, uncertain dotted, disabled toggle with the pending tooltip, opens in the first available layer, "Transcription in progress" when neither | **pass** | `verify.js`: tab state, tooltips and selection all derived from the live data; `verify5.js`: "36 .ln in first paragraph", "uncertain lines marked", "p041 foot notes flagged" (3), pending page "Transcription in progress"; `site-final-p004-fr-1440.png`, `site-final-p041-fr-1440.png`, `site-final-pending-1440.png` |
| 5 | **Header / Footer** — short title, `p. 43`, `printed as 24`, source line linking CUDL (Gallica for p041), layer and theme toggles; footer licence credit and item link | **pass** | `verify.js`: "#p004 src text" (`image 26 · Cambridge University Library` + href), "#p044 folio note" (`printed as 24`), "#p041 gallica text" + href, "footer text" and its two links |
| 6 | **Theme** — dark by default, sun/moon toggle stored in `localStorage`, `data-theme` on `<html>`, light theme complete | **pass** | `verify.js`: "default theme dark", "light persists across reload", "localStorage tc.theme = light", "fresh visit is dark"; `site-final-p004-en-light-1440.png` (pixel-sampled: body `rgb(246,244,239)` light vs `rgb(24,26,27)` dark) |
| 7 | **Fonts** — system stacks, no external requests of any kind | **pass** | `style.css` uses the two stacks the doc names; `verify.js` "off-origin requests (0)", `verify-deploy.js` "bad/off-origin responses (0)"; no `@font-face`, no CDN, no `<script src>` beyond `app.js` |
| 8 | **Responsive** — works at 390px, panels stack, image above text, zoom still works | **pass** | `verify.js`/`verify5.js`/`verify6.js` at 390×844: "no horizontal scroll" (`scrollWidth` 390), "390: image full width", "390: document scrolls at rest", "390: tap zooms", "390: touch pan moves the image"; `site-final-p004-en-390.png`, `site-final-p004-notes-open-390.png` |
| 9 | **Landing page** — title, intro, a card per book (cover = title page image, title, author/year, progress from `index.json`), dark, same palette | **pass** | `verify-deploy.js` against the tree `assemble.sh` builds: "landing: cover thumbnail is the title page scan", "landing: progress line computed from index.json" (`31 of 162 pages transcribed · 13 translated`), "landing: author and year", "landing: dark by default", then through the card to a working viewer at `/martin-guerre/`; `site-final-landing-1440.png`. Built by the landing task; verified here, not authored here |

## Contract items outside §5

| item | result | evidence |
| --- | --- | --- |
| `book.json.stylesheet` — per-book CSS, loaded after the viewer's own | **pass** | `verify6.js`: "book stylesheet: link appended", "after the viewer's own", "its rules apply", "a `..` path is refused", "sanitizer refuses schemes, roots and parents" |
| Relative paths — the viewer works at any mount point | **pass** | `verify-deploy.js`: "viewer at /martin-guerre/", "deep link works at the mount point" |
| `dev/` never deploys | **pass** | assembled tree returns 404 for `martin-guerre/dev/p004-en.json`; `assemble.sh` copies three files by name |

## Deviations from the letter of §5 (all previously reviewed and accepted)

1. **Sidenotes switch to two columns on the panel's width, not the window's.** §5 says ≥1100px; the panel is half the window, so at a 1100px window the text column would be about 21ch. A container query on the text panel (≈608px) puts p004 side-by-side at 1440 and tap-to-expand at 390, which is what the bullet is after.
2. **French sidenote column is 13rem, English 18rem** (§5 says ~18rem). At 18rem the French text column left 359px, and 35 of 36 diplomatic lines re-wrapped; at 13rem none do. English citations run to a sentence and keep 18rem.
3. **The image is fitted with `max-width`/`max-height`, not `object-fit: contain`.** Same result — whole page visible, centred, full panel height — but the `<img>` box is the visible image, which the zoom maths needs.
4. **Zoom is capped at 3× the fitted size** (§5 just says "toggles zoom"). 1:1 in device pixels is a five-fold jump on a non-retina screen.
5. **`spaced_caps` letterspaces headings only.** On a paragraph the flag marks isolated spaced capitals (`P R AE T E X T A`) that the transcription closes up, per p004's own `uncertain` note; letterspacing all 36 lines would misrepresent the page.
6. **The header's link to the site root is the `translations` crumb, not the book title.** §5 puts the link on the short title; the markup was specified this way when the shell was built, and the header still gets you to `/`.
