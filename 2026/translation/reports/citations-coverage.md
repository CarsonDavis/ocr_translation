# Citations coverage: why refs stop at `unit` or `work`

Generated from `python3 scripts/cite_locate.py --absent decretum` on 2026-10-04. 1434 references.
**The Decretum corpus is treated as absent**: another fetcher was still writing
`site/data/sources/decretum/` during this pass, so all Decretum refs count as `work` here.
Rerun `python3 scripts/cite_locate.py` (without `--absent`) once that corpus is finished.

## Status counts

| status | before | after |
|---|---:|---:|
| passage | 771 | 806 |
| unit | 254 | 219 |
| work | 336 | 315 |
| scan | 0 | 21 |
| none | 73 | 73 |

On top of the counts, 22 refs that were already `passage` now point at the right place. Before,
they hit the wrong passage because a chapter number happened to match an existing section id.
Pliny `VII.10 (§53–54)` was being sent to section 7.10 and now goes to 7.53–54. The same
problem affected Cicero (*Tusc.* `I.24.59` was sent to 1.24, now 1.59; *De or.*
`II.86–88, 351–360` now goes to 2.351–360), Josephus (`17, chapter 12 (17.324–338)` was
sent to section 17.12) and *Academica* (`II, §§ 54–57` was sent to section 2).

## The 254 `unit` refs, by cause

`unit` means the corpus and unit file were found but no passage was.

| cause | refs | by corpus |
|---|---:|---|
| (a) mechanical: the passage was in the Notes in another scheme or form | 29 | decretals 5, code 3, sext 2, historia-augusta 2, josephus 2, digest 1, novels 1, clementines 1, pliny 1, herodotus 1, aristotle EN 1, aristotle Pol. 1, plato Rep. 1, plato Phd. 1, plutarch 1, tertullian 1, appian Syr. 1, valerius 1, cicero Q. fr. 1, cicero Rab. Post. 1 |
| (b) genuinely unit-level | 215 | digest 91, code 55, decretals 37, novels 11, classical 21 |
| (c) range whose start is not a passage id | 0 | — |
| (d) corpus file gap | 10 | sext 6, code 4 |

### (a) mechanical (all 29 now resolve, as `passage`)

- Classical locus written in another scheme (7 refs). Stephanus and Bekker pages:
  `Plato, Republic book 5 (460e)` → 5.460; `Phaedo (61c–62c)` → 61–62;
  `Aristotle, Politics VII.16 (1335a)` → 7.1335a. Sections given with §:
  `Josephus XII, chapter 2 (12.2.10, §§ 89–90)` → 12.89–90 (×2);
  `Pro Rabirio Postumo (cf. § 36)` → 36. Chapter.section where the file stores sections:
  `Pliny VII, chapter 4 as printed … (7.53.180–181)` → 7.180–181.
- Locus only in a parenthesis or written as "chapters" (9 refs).
  `Marcus (Historia Augusta, Marcus 19.1–7)` → marcus.19.1–7;
  `Life of Pyrrhus (ch. 18, …)` → pyrrhus.18;
  `Ad Q. fratrem book 1, letter 1 (1.1.37)` → 1.1.37. The others: Herodotus 3 chapters 61–79,
  Appian *Syr.* chapters 67–68, Tertullian (ch. 13), NE book V (V.3), Valerius book 2 (cf. 2.1),
  and HA *Hadrian* (20.7–8).
- Place in the title (12 refs). `X 4.15, last chapter` → 4.15.7 (and `X 4.15, last chapter (c. 7 *Litterae*)`);
  `VI 3.15, its single chapter (c. un.)` → 3.15.1; `Nov. 134, last chapter (c. 13)` → 134.13.
  The others: X 5.2, X 3.27, X 1.40, VI 3.11, C. 8.15 (l. fin.), C. 9.27, D. 22.3 (each "last"),
  and C. 6.57 ("penultimate").
- OCR misnumbering (1 ref). Clem. 5.3 c. 1 is stored as `5.3.50` between
  nothing and `5.3.2`, so it is matched by position.

The scheme mismatches the brief guessed at did not occur among the `unit` refs. Pliny ids,
Valerius `ext`, Plutarch life names, Vulgate book names and Psalm numbering, Digest `pr`, the
Code concordance and missing Decretals chapters were already handled, or no ref needed them.
Adapters for them are in place and tested anyway.

### (b) genuinely unit-level (215)

| sub-cause | refs | examples |
|---|---:|---|
| law or chapter cited by incipit only, number not identified | 157 | `Digest 46.3, l. *Titia*; fragment not identified`; `Code 1.3, l. *Nulli*`; `Decretals, X 4.19, c. *Significasti*` |
| last or penultimate law/paragraph where the Notes express doubt, name another incipit, or cite a paragraph | 19 | `X 2.27 the last chapter (chapter not identified)`; `Digest 48.5, the penultimate law (fragment not identified)`; `Novel 77 … the last paragraph` |
| title only | 18 | `Digest 38.8, the law cited as *l. octaui* (not identified)`; `Decretals, X 2.20, the first chapter *Veniens*` (Veniens is X 2.20.10, not c. 1); `Novel 143 = Nov. 150` |
| classical work cited with no locus | 11 | `Plautus, *Amphitryon*`; `Cicero, *For Sextus Roscius of Ameria*`; `Cicero, *Post reditum ad Quirites* (section not located)` |
| classical book only, or passage not located | 10 | `Cicero, *On Duties*, book 1`; `Pliny XI, chapter 37 as printed (… not located)`; `Herodotus, book I (*Clio*)` |

A next step that is not mechanical: about 65 of the 157 incipit-only law refs have exactly
one fragment in their unit file whose text starts with the incipit (for example `D. 46.3, l.
*Titia*` and D. 46.3.48; `D. 28.5, l. *Duo socii*` and D. 28.5.8). In many of these the Notes
say "fragment not identified", so a match is a new identification. It needs a reader to
check it, and the locator does not apply it.

### (d) corpus file gaps (10)

- `sext` 5.13 (*De regulis iuris*) is not in the corpus: `units` stops at 5.12 (6 refs).
  These refs used to show as `unit` only because the parser filed the regulae under 5.12,
  which is *De verborum significatione*. They now read `VI 5.13.n` with status `work`, and
  `scan_url` is set to the volume scan. Examples: `reg. 54, Qui prior est tempore`,
  `reg. 8, Semel malus`, `reg. 23, Sine culpa`.
- `code` `empty_at_source` (4 refs): `C. 4.20.6` *Parentes*, `C. 4.20.9` *Iurisiurandi* (×2) and
  `C. 9.6.1`. The source page has the number but no text, and the refs carry
  `source_gap: empty_at_source`.
- No ref hit `greek_not_online` or a `no_chapters` Novel.

## The `work` refs after this pass (315)

| class | refs | notes |
|---|---:|---|
| commentary | 141 | Expected. A commentator's own text is not stored, and `on` locates the text commented on. A further 11 commentaries with no locatable `on` are `none`, which makes 152 commentaries in all. |
| decretum, corpus treated as absent | 101 | Will resolve once `decretum/` is finished. 9 Decretum commentaries are counted under commentary. |
| classical, `not-found` in classical-works.json | 30 | No open text exists. These carry `source_gap: not_found`. Examples: Jerome Ep. 72, Aristotle *Hist. an.* 5.14, Macrobius *Somn.* 1.6. |
| classical, not in classical-works.json and no corpus | 26 | Mostly jurists' books parsed as classical (Albericus de Rosate, Petrus Ravennas, Guillaume Benoît, Chasseneuz), plus Ovid *Ex Ponto* 2.3.19, Sallust *Cat.*, and `History of Animals 7.4` cited without its author (slug `history-of-animals`). |
| law, unit not in corpus or no unit | 11 | sext 5.13 ×6; Digest and Code titles named without a number, e.g. `Code, title De in integrum restitutione`, `Digest, De legatis (D. 30–32), l. 1 — as printed`. |
| classical, corpus present, no locus | 4 | `Cicero, In Verrem`, `Cicero, Rhetorica` (book unidentified), `Homer, Iliad, books 22 and 24`, `Cicero, Paradoxa Stoicorum`. |
| libri-feudorum, no corpus | 2 | `LF 2.27` |

These 21 refs moved to status `scan`: `scan-only` works in classical-works.json now carry
`scan_url` = that file's `source`. They are Crinito ×8, Fregoso ×2, Paolo Emili ×2, Platina ×2,
Münster, Budé, Polydore Vergil, Volaterranus, Rhodiginus, Alexander ab Alexandro and Galeotto
Marzio. Six `work` refs became `passage` because the locus sat only in the parenthesis or in an
actio.book unit: Odyssey (II.93–110), Variae (I.45), Pausanias (6.8.2), Fam. (5.16),
Verrines II.4 § 39 and Appian *Mithr.* (ch. 112), the last in a split `all` unit.

## Adapters (scripts/cite_locate.py)

The output records which adapter produced a reading in `adapter`, and why there is no
passage in `source_gap`.

- `section-sign`, `modern-numbering`, `chapter-section`, `chapter-then-section`,
  `parenthesis-dotted`, `parenthesis-only`, `chapter`: these apply to corpora whose ids are
  sections or pages (Pliny, Josephus, Cicero, Plato, Diogenes, Appian, Aristotle *Pol.*). A
  chapter in the Notes is never taken as a section id there. When nothing else applies, the
  ref falls back to `unit`.
- `label-chapter`: matches on the unit file's "(chapter N)" or "(section N)" labels. Pliny's
  labels follow the old (Hardouin/Loeb) division, so they are used only when the Notes say
  "old chapter division".
- `stephanus-page`, `bekker-page`, `part-chapter`, `part-chapter-section` (lives), and
  `actio-book-section` (Verrines).
- `fuzzy_id`: matches ids ignoring dots, spaces and colons, with Perseus `9.15e.1` = `9.15.ext.1`
  and Digest `.0` = `.pr`. Dots between numbers still count, so 1.2 never matches 12.
- `vulgate_unit`: resolves a book name through `books` in vulgate/corpus.json (key, Latin name,
  abbreviations).
- Code concordance: forward (Coras to Krüger), then the number as given, then backwards
  (`concordance-inverse`).
- `position-*`: "last/first/penultimate law or chapter", `l. fin.`, `c. un.`, or "last chapter
  (c. N)". Not applied when the Notes express doubt, name another incipit, or name the chapter
  by incipit.
- `ocr-position`: for OCR'd canon-law corpora.
- `--absent CORPUS`: treats a corpus as not fetched.
