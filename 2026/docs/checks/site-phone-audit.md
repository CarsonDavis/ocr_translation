# Phone audit of the Martin Guerre viewer and its blog embed

Audited 2026-10-04 against the live site (`https://translations.codebycarson.com/martin-guerre/`,
landing at `/`, blog embed at `https://madebycarson.com/posts/troublesome_translations/`).
Driven with Playwright on the installed Chrome using the `iPhone 13` (390×664, DPR 3),
`Pixel 5` (393×727, DPR 2.75), `iPhone SE` (320×568, DPR 2) and `iPhone 13 landscape`
(750×342) device descriptors, touch on, dark colour scheme unless stated. Screenshots
are the `site-phone-*.png` files beside this report; scripts are in the session scratchpad
(`phone/probe.py`, `interact.py`, `interact2.py`, `weight.py`, `embed*.py`, `landing.py`).
Line numbers refer to the live `app.js` and `style.css` as served on the audit date
(the working tree in `code-by-carson/translations/viewer/` had uncommitted edits during
the audit; function names are given so the references survive line drift).

## Verdict

On a phone the viewer is sound underneath — nothing overflows, the type is a comfortable
size, zoom and pan work by touch, the theme toggle works, the data loads fast once the
scan is in — but the first impression is a wall of chrome and a wall of scan. The header
takes three rows (105 px, 16% of an iPhone 13 screen; five rows and 179 px, 32%, on an
iPhone SE) and stays stuck there while you read; below it the page image fills the whole
width and runs to the bottom of the screen, so a new visitor sees an unreadable thumbnail
of a 1572 page and no hint that a translation is underneath (8 px of the text panel
clears the fold on the iPhone 13, none on the SE, and in landscape there are 3.3 screens
of scan before the first word). Sidenote markers are 7–10 px wide and, when tapped, open
the note at the end of the paragraph, 1100–1500 px below the finger, so the tap appears to
do nothing. Each page costs about 1 MB of image for a scan displayed at 14% scale, and the
next page is prefetched at the same size. The blog embed is worse off: at 358 px wide it
spends 131 px on header, shows the scan at 7.6% scale, leaves 247 px for text, and its
text panel captures five or six consecutive swipes from a reader who is only trying to
scroll past it. None of this is hard to fix; the list below is in the order I would do it.

## Problems, ranked

### 1. The first screen is all chrome and scan; the text is below the fold

Evidence: `site-phone-iphone13-first-open.png` — header 105 px (y 0–105), image panel
390×551 (y 105–656), text panel starts at y 656 in a 664 px viewport. `site-phone-iphonese-first-open.png`
— header 179 px, image 320×452, text starts at y 631 in a 568 px viewport (63 px below the
fold). Pixel 5: header 105, image 555, text at 660 of 727. Landscape
(`site-phone-iphone13-landscape-first-open.png`): header 78 px, image 750×1059, text at
y 1137 of a 342 px viewport, i.e. 3.3 screens of scan before any English. At fit the scan
is 390/2805 = 13.9% scale (11.4% on the SE): the body type is a 2–3 px x-height, legible
only as texture. Nothing overflows horizontally (`scrollWidth == clientWidth` on every
device, 390/393/320/750).

Severity: high. It is the thing every phone visitor hits first.

Fix (`style.css`): on the stacked layout cap the frame and let the image letterbox inside
it, the way the embed section already does. Add to the `@media (max-width: 899.98px)`
block (or a new one):

```css
@media (max-width: 899.98px) {
  .frame { height: clamp(240px, 52dvh, 560px); display: flex; align-items: center; justify-content: center; }
  img.scan { width: auto; height: auto; max-width: 100%; max-height: 100%; }
}
```

On the iPhone 13 that gives a 345 px frame (scan 244 px wide, 12% instead of 14% — the
fit view is a thumbnail either way; zoom is what makes it readable) and puts about 210 px
of English above the fold; on the SE about 100 px. `zoomIn()` already freezes the frame
height (`zoom.frame.style.height = …`) so zoom keeps working. In landscape the same rule
yields a 178 px frame; better still, treat landscape phones as two columns:
`@media (min-width: 640px) and (orientation: landscape)` applying the ≥900 px grid, since
the container query already puts notes beside the text at that width
(`site-phone-iphone13-landscape-reading.png`).

### 2. The sticky header is three to five rows tall and never leaves

Evidence: `site-phone-iphone13-reading.png` (bar 105 px, 16% of the viewport, while
reading mid-page), `site-phone-iphonese-reading.png` (179 px, 32%), landscape 78 px (23%).
Rows: brand + pager, page/source line, layer tabs + readings + theme. The source link
"image 26 · Cambridge University Library" is 253 px wide and alone forces a row on every
phone.

Severity: high.

Fix (`style.css` `@media (max-width: 700px)` block and `index.html`):

- Make the bar static on phones: `.bar { position: static; }` under `max-width: 700px`.
  Scrolling up to the pager is the normal phone pattern and the reader gets the full
  screen for text. If a sticky pager is wanted, keep only `.pager` sticky as a slim
  40 px strip, not the whole bar.
- Move the `.src` link out of the header into the footer (`.foot`) on phones, or shorten
  its label to "CUL image 26" in `sourceLabel()` when `matchMedia('(max-width: 700px)')`
  matches. That alone collapses the second row into the first on the SE.
- Drop `.home` + `.sep` on phones and make `.book-title` the link home, as the embed
  already does (`.embed .home, .embed .sep { display: none }`).

Target: two rows (~70 px) on the iPhone 13, three on the SE.

### 3. Tapping a sidenote marker appears to do nothing

Evidence: `site-phone-iphone13-marker-tapped.png` — marker `f` tapped at viewport y 574;
the `.notes` block opened at y 1785 (viewport 664 px), i.e. 1121 px below the finger. On
the SE: tapped at y 526, notes at y 2023. `openNote()` toggles `para.open` and highlights
the marker for 1.2 s, but the paragraph with notes on p004 is 46 lines (1200 px) long and
the notes are appended after it. The marker itself measures 7–10 × 16 px at 11.76 px type
(`.mk { font-size: 0.7em; vertical-align: super }`), far below a 44 px target; eight
markers in the first paragraph. Second observation: `para.classList.toggle('open')` means
tapping marker `a` then marker `b` in the same paragraph closes the notes (verified: open
→ false after the second tap). Once found, the open notes are readable:
`site-phone-iphone13-notes-open.png`, 13.6 px / 19 px line, 355 px wide, highlighted note
has the accent rule; tapping the same marker again collapses them.

Severity: high for the first point, medium for the others.

Fix (`app.js` `openNote()`, `style.css` `.mk`):

- In `openNote()`, when the panel is narrower than the container breakpoint (`.notes` is
  `display: none` at rest), after adding `open` call
  `para.querySelector('.note[data-key="' + key + '"]').scrollIntoView({ block: 'nearest', behavior: 'smooth' })`
  and keep the highlight until the next tap instead of 1.2 s. Better: on narrow panels
  insert the single tapped note directly after the line that holds the marker (a
  `<span class="note inline">` clone inserted after `mk.closest('.ln') || mk`), so the
  note opens under the finger like a footnote popover; the end-of-paragraph list stays for
  the "show all" case.
- Change the toggle to `if (!para.classList.contains('open') || sameKeyAsLastTap) para.classList.toggle('open')`
  so a different marker in an open paragraph re-highlights rather than closing.
- Give markers a tap area without changing the type: `.mk { padding: 0.6em 0.45em; margin: -0.6em -0.3em; }`
  (keeps the superscript look; the hit box becomes ~24 × 30 px). A `min-height: 44px`
  target is not achievable inline; this plus the scroll fix is enough.

### 4. Turning the page while scrolled lands mid-text with no scan

Evidence: scrolled to `scrollY = 1200` in p004's English and tapped →; p005 rendered with
`scrollY` 1208 (`site-phone-iphone13-after-next-scrolled.png` in the scratchpad: p005
opens in the middle of its paragraph, the header's "p. 5" is the only cue). `renderText()`
resets `panel.scrollTop = 0`, which is the desktop scroller; on phones the window is the
scroller and nothing resets it. Same on the SE (1200 → 1207).

Severity: medium-high (every page turn on a phone, once the reader has scrolled).

Fix (`app.js` `go()`/`route()`): after `renderImage`/`renderText`, if the stage is
stacked (`!matchMedia('(min-width: 900px)').matches`), call
`window.scrollTo({ top: el.imagePanel.offsetTop - barHeight, behavior: 'instant' })`
where `barHeight` is 0 once the bar is static (see 2). With the fix in 1, that shows the
new scan and the first lines of its text together.

### 5. Every page is ~1 MB, prefetch doubles it, and the phone shows it at 14%

Evidence (`weight.py`): first open of `#p004` transfers 2,002 KB in 8 responses:
`img/p004.webp` 1,034 KB, prefetched `img/p005.webp` 903 KB, `app.js` 55 KB (br),
`index.json` 22.5 KB, `pages/p004.json` 16 KB, `style.css` 16 KB, `book.json` 1.9 KB,
HTML 1.6 KB. Images are 97% of the bytes. Each page turn then costs the next prefetch
(p006 987 KB, p007 923 KB); the page you turn to is already cached, so the scan appears in
30–80 ms. Scan visible after 0.57 s unthrottled, 2.1 s under a 9 Mbps / 85 ms "4G"
emulation, 7.0 s under 1.6 Mbps / 150 ms "slow 4G" (CDP `Network.emulateNetworkConditions`).
Arithmetic: 1 MB is 0.9 s at 9 Mbps and 5.2 s at 1.6 Mbps; a 340 KB variant is 0.3 s and
1.7 s. At fit the iPhone 13 paints the image at 1,170 device px wide; only zoom uses the
full 2,805 px (`zoomWidth()` picks 1:1 device pixels = 935 CSS px). Ten pages of reading on
a phone is ~10 MB today; it would be ~3.5 MB with a phone variant, ~4.5 MB including the
full image fetched on the pages the reader actually zooms.

Severity: medium (the site is fast on wifi; on mobile data it is slow and expensive,
and the landing page thumbnail is the worst offender, see 9).

Fix: generate `img/sm/<id>.webp` at 1400 px wide (~340 KB, matching the quoted estimate)
and add `"sm": { "base": "img/sm/", "width": 1400 }` to `book.json` `images`. In
`renderImage()` set

```js
img.srcset = smallUrl(id) + ' 1400w, ' + imageUrl(id) + ' 2805w';
img.sizes  = '(min-width: 900px) 50vw, 100vw';
```

The iPhone 13 (100vw × 3 = 1,170) and every DPR ≤ 3 phone pick the 1400w candidate; a
1440 px DPR 2 laptop (50vw × 2 = 1,440) still picks the full image. In `zoomIn()` set
`img.sizes = '2805px'` before measuring: changing `sizes` makes the browser select and
fetch the 2805w candidate while continuing to show the small one until it lands
(`zoomWidth()` should use `state.book.images.width` rather than `img.naturalWidth`, which
is 1400 until the swap completes). `unzoom()` restores `sizes`. `prefetch()` should build
its `Image` with the same `srcset`/`sizes` so it warms the candidate the phone will use,
not the full one. A `swap on zoom` without `srcset` (plain `src` swap) also works but
loses the automatic selection on desktop.

### 6. The blog embed traps scrolling and spends the frame on chrome

Evidence (`site-phone-iphone13-embed.png`, `site-phone-iphonese-embed.png`): the iframe
computes to 358×680 on the iPhone 13 and 288×680 on the SE (Chirpy's `.content` column is
the viewport minus 16 px gutters; Chirpy has no `max-width`/`overflow` rule on iframes,
only `iframe { border: 0 }`, and the element is `display: inline`). Inside the frame the
header is 131 px (191 px on the SE, five rows plus "Open full viewer"), the image panel
302 px shows the scan at 214×302, 7.6% scale, and the text panel is 247 px (≈ 8.8 lines of
16.8 px text; 220 px / 7.8 lines on the SE) scrolling 1,469 px of content. Scroll trap,
measured with the iframe fully on screen: five consecutive upward touch swipes on the
iPhone 13 (six on the SE) moved only the inner text panel (289, 292, 304, 297, 41 px) and
the blog page did not move; a mouse wheel over it lost four 300 px ticks the same way.
Swipes over the header or the image panel pass through to the page. "Open full viewer ↗"
is visible (116×20 px at y 103; y 163 on the SE) and tappable, `target=_top`, correct hash.
Tap-to-zoom inside the frame works (640 px wide in a 301 px frame,
`site-phone-iphone13-embed-zoomed.png`) and is the only way to read anything in it.
Separately, the blog page itself scrolls horizontally to 470 px on a 390 px phone; the
culprit is Chirpy's Rouge code table (`.rouge-table` 798 px wide) and a `.caption` div
earlier in the post, not the iframe.

Severity: high for the blog post (it is the one place most readers will meet the viewer).

Fix: see the embed recommendation at the end. The minimum viewer-side change is in
`style.css`'s embed section: on narrow embeds hide the parts that cannot work —
`@media (max-width: 599.98px) { .embed .meta, .embed .readings-toggle, .embed .theme { display: none } .embed .text-panel { display: none } .embed .stage { grid-template-rows: 1fr } }`
— which removes the nested scroller (no trap), cuts the header to two rows, and gives the
scan the remaining ~560 px (≈ 14% scale, tap to zoom to 3×). The blog's own caption
already links to the full viewer.

### 7. The jump box triggers iOS auto-zoom and the header controls are under 44 px

Evidence: `.jump input` computes to 13.6 px type and 72×29 px (`site-phone-iphone13-first-open.png`);
iOS Safari zooms the page when an input under 16 px is focused, and this is the only
input on the page. Buttons: prev/next 32×29, theme 32×29, English 65×29, French 63×29,
contested readings 141×29. Datalist has 162 options and works on the phone (`inputmode=text`
so the letters "title"/"argument" can be typed). Jumping to 43 works.

Severity: medium.

Fix (`style.css`): under `max-width: 700px`, `.bar button, .jump input { font-size: 16px; min-height: 40px; padding: 0.35rem 0.7rem }`
and `.prev, .next, .theme { min-width: 44px }`. At 16 px the input also needs
`width: 4.5rem` → `5.5rem`. The 44 px height makes a static two-row bar ~96 px, which is
why 2 and 7 go together.

### 8. Zoomed state has no visible exit and swallows scrolling

Evidence: `site-phone-iphone13-zoomed.png` — a tap zooms to 935 CSS px (1:1 device pixels
at DPR 3; the `3 × fitted` cap of 1,170 is not reached), the frame freezes at 551 px, and
`touch-action: none` is applied, so vertical swipes pan the scan instead of scrolling the
page. Drag-pan works (transform moved −100/−150 px for a 100/150 px drag, `scrollY`
unchanged) and a second tap unzooms, so the reader is never truly stuck, and `renderImage()`
unzooms on page turn. But nothing on screen says "tap to leave"; the `aria-label` changes
to "Leave zoom" and that is all. Pinch: the viewport meta allows user scaling (good);
while zoomed `touch-action: none` blocks pinch on the frame, at rest a pinch zooms the
whole page including the header. There is no pinch-to-zoom of the scan itself, which is
the gesture phone users try first.

Severity: medium.

Fix (`app.js` `zoomIn()`/`unzoom()`, `style.css`): add a small fixed "✕ 1:1" chip in the
frame's corner while `.zoomed` (`.image-panel.zoomed .frame::after { content: 'tap to close' ... }`
is enough; a real button is better for assistive tech). Consider `touch-action: pan-y pinch-zoom`
at rest so the browser's own pinch still works, and in `bindFrame()` handle a second
pointer (two `pointerdown`s) as a pinch that scales `zoom.width` between fitted and 1:1
rather than ignoring it.

### 9. Landing page loads the full 1 MB title scan for a 200 px thumbnail

Evidence (`landing.py`): `p000-title.webp` 1,007,922 bytes (2767×3950) rendered at
200×285 CSS px, `loading=lazy` but it is above the fold so it loads immediately.
Otherwise the landing page is fine on a phone (`site-phone-iphone13-landing.png`):
no horizontal overflow, 32 px serif h1, 16 px body, card stacks, "Read →" 57×26 px (the
title is also a link, so the small target is acceptable).

Severity: medium (cheap to fix, big saving).

Fix (`landing/index.html` `renderCard()`): point `img.src` at the 1400 px variant from
problem 5 (or a dedicated `img/thumb/<id>.webp` at 400 px, ~40 KB) and set
`img.sizes = '200px'` with a `srcset` if both exist.

### 10. French layer: a printed line is two screen lines on every phone

Evidence: `site-phone-iphone13-french.png` — 15.68 px type, 35 of 38 printed lines wrap
(28 of 35 on p005; 30 of 35 on the SE), longest line 62 characters against a 354 px
column that fits ~43. The hanging indent (`padding-left: 1.2em; text-indent: -1.2em`)
keeps the lineation readable and nothing overflows (`scrollWidth` 390/320), but the
paragraph is 1,726 px tall and the eye reads it as 70 ragged lines. Letter-spaced caps
headings (`TEXTE.`, `ARREST`, `A PARIS,`, `M. D. LXXII.`) are 16.8 px with 2 px tracking
and read well on the title page. Note: `.lines.spaced` (a `spaced_caps` paragraph) gets no
rule — only `.blk-heading.spaced` has the `letter-spacing` — which is a desktop issue too.
French markers measure 8–9 × 15 px.

Severity: low-medium (it is a faithful rendering; it is just long).

Fix (`style.css`): under `max-width: 700px`, `.lines .ln { font-size: 0.92rem; }` and
`.text-panel { padding: 1.1rem 0.8rem }` recover ~6 characters; most 55–62 character lines
still wrap. The honest alternative is to accept wrapping and reduce the hanging indent
to `0.8em` so the second half of a line is not pushed so far right. Add
`.lines.spaced .ln { letter-spacing: 0.08em }` for the missing rule.

### 11. Safe-area insets in landscape (from the CSS; not measurable in Chromium)

`index.html` sets `viewport-fit=cover`, so Safari extends the page under the notch in
landscape. `style.css` only consumes `env(safe-area-inset-top)` (on `.bar { top }`) and
`env(safe-area-inset-bottom)` (footer padding). The left/right insets (47 px on an
iPhone 13 in landscape) are not applied anywhere, so the "translations" link at x 14, the
← button, and the text panel's first 24 px sit under the sensor housing on one side and
the home-indicator edge on the other.

Severity: low (landscape phone reading is rare, and problem 1 makes it unattractive anyway).

Fix (`style.css`): `.bar, .text-panel, .foot { padding-left: max(0.9rem, env(safe-area-inset-left)); padding-right: max(0.9rem, env(safe-area-inset-right)); }`
(use the panel's own 1.5rem/1.1rem in place of 0.9rem). The frame can stay edge to edge.

### 12. Small things

- `favicon.ico` is a 404 at both `/martin-guerre/favicon.ico` and `/favicon.ico`
  (console error on every first open). Add a `<link rel="icon">` or ship one.
- Dark is the default regardless of `prefers-color-scheme` (a light-mode phone with no
  stored preference gets dark; `data-theme="dark"` is hard-coded in `index.html`). That
  matches the spec ("dark by default"), the toggle works (`tc.theme` stored,
  `site-phone-iphone13-light.png` renders cleanly with paper `#e9e5dc`), and nothing odd
  happens in light mode. If system preference should win on first visit, read
  `matchMedia('(prefers-color-scheme: light)')` when `localStorage` has no value.
- `100vh`/`100dvh`: the stacked layout has no fixed height, so the mobile URL bar does
  not clip anything; the ≥900 px grid uses `100dvh` correctly; the embed's
  `height: 100dvh` resolves to the iframe height. No issue found.
- Image decode: `img.decoding = 'async'` is set and the fade-in hides decode; no jank
  observed in screenshots. A 2805×3962 WebP decodes to ~44 MB of RGBA on every page
  turn, which is another reason for the phone variant.
- Swiping left or right on the scan or the text does nothing (hash unchanged). Not a bug,
  but with the arrow keys gone a horizontal swipe on the image panel (pointer travel
  > 60 px, mostly horizontal, not zoomed) could call `go(page.next/prev)` cheaply inside
  `bindFrame()`.
- Contested-readings popover on a phone: tap on a dotted run opens a 374×225 px fixed
  panel at 12.8 px type that fits the viewport (`site-phone-iphone13-readings-popover.png`)
  and closes on a tap elsewhere. Fine, though 12.8 px is small for a reading aid;
  `.rd-pop { font-size: 0.9rem }` under 700 px would help.
- No focus traps found; the about `<dialog>` closes on backdrop tap.

## Already fine

- No horizontal overflow on any device or orientation, in either layer, with notes open
  or the popover shown.
- English body: 16.8 px Iowan Old Style / Palatino, 26 px leading, ~43 characters per
  line on a 390 px phone, ~35 on a 320 px one. Comfortable.
- Expanded notes are readable (13.6/19 px, full column width, key in the gutter, accent
  rule on the active note) and collapse on a second tap of the same marker.
- Tap to zoom lands at 1:1 device pixels centred on the finger, drag pans without
  scrolling the page, a tap leaves zoom, and turning the page leaves zoom.
- Prev/next and the jump box work by touch; the datalist offers all 162 pages; a bad
  entry shakes the box.
- Theme toggle works and persists; both palettes are clean on a phone.
- Navigation after the first page is instant: the next scan is prefetched and the page
  JSON is 16–22 KB, so a page turn shows its scan in under 100 ms even on throttled 4G.
- Landscape at 750 px already puts sidenotes beside the paragraph via the container query.
- The embed's "Open full viewer ↗" link is present, visible without scrolling, and
  targets `_top` with the current page hash.
- Headers: `cache-control` immutable on images, brotli on text, CSP `frame-ancestors`
  limited to the blog hosts.

## Recommendation: smaller image variant

Yes, warranted. A 1400 px wide WebP (~340 KB) is sharp at fit on every phone (the
iPhone 13 paints 1,170 device px) and on DPR 1 laptops, and cuts a phone page turn from
~1 MB to ~0.35 MB and the first open from 2.0 MB to ~0.75 MB; on 1.6 Mbps that is 7 s →
~2.5 s to a visible scan. Wire it with `srcset`/`sizes` as in problem 5, with `sizes`
swapped to `2805px` in `zoomIn()` so the full scan is fetched only when the reader zooms,
and make `prefetch()` and the landing thumbnail use the same selection. No change to the
data contract beyond an optional `images.sm` block in `book.json`; pages without a small
file fall back to the single `src`.

## Recommendation: blog embed

Replace it on narrow screens. Below about 700 px the frame cannot show a readable scan and
readable text at once, its header costs a fifth of the frame, and its scrolling text panel
hijacks five or six swipes from everyone scrolling past. In the post, keep the iframe for
wide screens and, under a `@media (max-width: 699.98px)` rule in the post's own HTML,
show instead a static JPEG of page 4 at ~700 px wide (~120 KB, `loading=lazy`) with the
existing "Open the full viewer ↗" line as a button beneath it; the viewer's `?embed=1`
mode still serves desktop readers unchanged. If the iframe must stay on phones, change
the viewer rather than the height: hide the text panel and the meta/readings/theme
controls in narrow embeds (problem 6) so the frame is a tappable scan with a pager and a
link out, and shorten the iframe to 520 px so it sits inside one screen. Changing only the
height (taller or shorter) does not address either the trap or the legibility, so I would
not do that on its own.
