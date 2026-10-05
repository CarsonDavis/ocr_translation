"""Parse every citation in the Notes of translation/sections/*.md and locate it in the
fetched source corpora.

usage: cite_locate.py [--report-only] [--unparsed-file PATH] [--absent CORPUS ...]
Reads every Notes line `- {x} (pNNN): **<identification>** — <sigla> [gloss]`, parses the
bold identification into structured references (see parse_identification), checks each
against site/data/sources/<corpus>/ (corpus.json + unit files; see docs/sources-contract.md),
and writes site/data/citations.json keyed `pNNN:x`. Prints a coverage report: references
parsed and unparsed (the unparsed identifications listed, deduplicated, so the grammar can
be extended) and status counts. --report-only prints the report without writing.
--absent CORPUS (repeatable) treats that corpus as not fetched, e.g. while a fetcher is
still writing it.

Status: `passage` (unit file present and the passage found in it), `unit` (unit file
present, passage not found or not cited), `work` (the corpus is identified but not
fetched, or the unit is not in it; also commentaries, whose own text is not stored), `scan`
(a work classical-works.json lists as scan-only: scan_url is its scan), `none`
(unidentified, cross-references, unparsed).

Scheme adapters (see classical_candidates, code_candidates): the Notes cite in the
scholarship's conventions, the corpora store their edition's passage_scheme. A ref is tried
in several readings (Pliny/Josephus/Cicero "§ 53" or chapter.section -> book.section;
"7.53 in the modern numbering"; Stephanus and Bekker pages; "(ch. 18)" or "(19.1–7)" in a
parenthesis for lives; book-level refs refined by a dotted locus in the parenthesis;
Verrines actio.book units; label chapters where the Notes name the labels' division; the
Code concordance forwards, as given, and backwards) and the first that lands on a passage
wins; a chapter number is never taken for a section id. Ids are also matched ignoring dots,
spaces, 'e'/'ext' and 'pr'/'0' (fuzzy_id); the Vulgate book map in vulgate/corpus.json
resolves book names; OCR'd canon-law corpora match a misnumbered chapter by position. Classical work ids come from
site/data/sources/classical-works.json when present; otherwise from the alias table
below, otherwise an author-title slug (lower case, hyphens).

Each object carries the contract fields (ref, corpus, unit, passage, passage_end, status,
external_url, scan_url) plus `kind` (law, bible, classical, commentary, crossref, backref,
unidentified, unparsed) and, where relevant, `cf` (the translators wrote cf.),
`commentator` and `on` (the text a commentary is on, located like any reference),
`cts_urn`, `coras_numbering` (Code number before the concordance), `adapter` (which scheme
adapter produced the reading), `source_gap` (why the corpus has no passage: empty_at_source,
greek_not_online, no_chapters from corpus.json, or not_found from classical-works.json). Translators' remarks
that follow a ';' ("fragment not identified") are not references and are dropped.
Run build_sources_index.py first so index.json matches the corpora on disk.
"""
import json, pathlib, re, sys, unicodedata
from collections import Counter
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bookconf import ROOT  # noqa: E402

NOTE = re.compile(r"^- \{([^}]+)\} \((p\d{3}(?:-[a-z]+)?)\)(.*)$")
BOLD = re.compile(r"\*\*(.+?)\*\*(?=\s*(?:—|\[|$))")

# --------------------------------------------------------------------------- helpers

ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman_to_int(s):
    if not s or not re.fullmatch(r"[IVXLCDM]+", s):
        return None
    total, prev = 0, 0
    for ch in reversed(s):
        v = ROMAN[ch]
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total


def num(s):
    """'IX' -> '9', '12' -> '12', else None."""
    if s is None:
        return None
    if s.isdigit():
        return str(int(s))
    r = roman_to_int(s)
    return str(r) if r else None


def norm(s):
    """Normalise a unit or passage id for comparison: lower case, letter/digit boundaries
    and any run of non-alphanumerics become one dot."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])", ".", s)   # C33 q1 == C.33 q.1
    return re.sub(r"[^a-z0-9]+", ".", s).strip(".")


def slug(s):
    return norm(s).replace(".", "-")


def mask_parens(s):
    """Same-length copy of s with everything inside (...) and [...] replaced by spaces,
    and a depth array."""
    out, depth, depths = [], 0, []
    for ch in s:
        if ch in "([":
            depth += 1
            out.append(" ")
        elif ch in ")]":
            depth = max(0, depth - 1)
            out.append(" ")
        else:
            out.append(ch if depth == 0 else " ")
        depths.append(depth)
    return "".join(out), depths


def split_top(s, seps=(";",)):
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        if depth == 0 and ch in seps:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def plain(s):
    return re.sub(r"\s+", " ", s.replace("*", "")).strip()


DASH = r"[–—-]"

# --------------------------------------------------------------------------- Bible

# Book names as the vulgate corpus uses them (English names, Vulgate numbering of Kings and
# Psalms: 1 Samuel = 1 Kings, 1 Kings = 3 Kings) with aliases; the number is the
# book's position at drbo.org's Latin Vulgate (https://www.drbo.org/lvb/chapter/BBCCC.htm).
BIBLE = [
    ("Genesis", 1, []), ("Exodus", 2, []), ("Leviticus", 3, []), ("Numbers", 4, []),
    ("Deuteronomy", 5, []), ("Joshua", 6, ["Josue"]), ("Judges", 7, []), ("Ruth", 8, []),
    ("1 Kings", 9, ["1 Samuel", "1 Kingdoms"]), ("2 Kings", 10, ["2 Samuel", "2 Kingdoms"]),
    ("3 Kings", 11, []), ("4 Kings", 12, []),
    ("1 Chronicles", 13, ["1 Paralipomenon"]), ("2 Chronicles", 14, ["2 Paralipomenon"]),
    ("Ezra", 15, ["1 Esdras"]), ("Nehemiah", 16, ["2 Esdras"]), ("Tobit", 17, ["Tobias"]),
    ("Judith", 18, []), ("Esther", 19, []), ("Job", 20, []), ("Psalms", 21, ["Psalm"]),
    ("Proverbs", 22, []), ("Ecclesiastes", 23, []),
    ("Song of Songs", 24, ["Canticle of Canticles", "Song of Solomon", "Canticles"]),
    ("Wisdom", 25, []), ("Ecclesiasticus", 26, ["Sirach"]), ("Isaiah", 27, ["Isaias"]),
    ("Jeremiah", 28, ["Jeremias"]), ("Lamentations", 29, []), ("Baruch", 30, []),
    ("Ezekiel", 31, ["Ezechiel"]), ("Daniel", 32, []), ("Hosea", 33, ["Osee"]), ("Joel", 34, []),
    ("Amos", 35, []), ("Obadiah", 36, ["Abdias"]), ("Jonah", 37, ["Jonas"]),
    ("Micah", 38, ["Micheas"]), ("Nahum", 39, []), ("Habakkuk", 40, ["Habacuc"]),
    ("Zephaniah", 41, ["Sophonias"]), ("Haggai", 42, ["Aggeus"]), ("Zechariah", 43, ["Zacharias"]),
    ("Malachi", 44, ["Malachias"]), ("1 Maccabees", 45, ["1 Machabees"]),
    ("2 Maccabees", 46, ["2 Machabees"]), ("Matthew", 47, []), ("Mark", 48, []), ("Luke", 49, []),
    ("John", 50, []), ("Acts", 51, ["Acts of the Apostles"]), ("Romans", 52, []),
    ("1 Corinthians", 53, []), ("2 Corinthians", 54, []), ("Galatians", 55, []),
    ("Ephesians", 56, []), ("Philippians", 57, []), ("Colossians", 58, []),
    ("1 Thessalonians", 59, []), ("2 Thessalonians", 60, []), ("1 Timothy", 61, []),
    ("2 Timothy", 62, []), ("Titus", 63, []), ("Philemon", 64, []), ("Hebrews", 65, []),
    ("James", 66, []), ("1 Peter", 67, []), ("2 Peter", 68, []), ("1 John", 69, []),
    ("2 John", 70, []), ("3 John", 71, []), ("Jude", 72, []),
    ("Apocalypse", 73, ["Revelation"]),
]
BIBLE_ALIAS = {}
for _name, _n, _al in BIBLE:
    for a in [_name] + _al:
        BIBLE_ALIAS[a.lower()] = (_name, _n)
_bnames = sorted(BIBLE_ALIAS, key=len, reverse=True)
BIBLE_RE = re.compile(
    r"\b(" + "|".join(re.escape(n).replace(r"\ ", r"\s+") for n in _bnames) + r")\s+(\d+)"
    r"(?::(\d+)(?:\s*" + DASH + r"\s*(\d+))?)?(?![\d.])", re.I)

# --------------------------------------------------------------------------- classical

# (author, title) aliases -> (work id, CTS work URN or None). Ids follow the contract's
# style (author + Latin title slug). URNs only where the Perseus/Scaife URN is certain.
CLASSICAL = [
    ("cicero", ["on duties", "de officiis"], "cicero-de-officiis", "urn:cts:latinLit:phi0474.phi055"),
    ("cicero", ["pro cluentio", "for cluentius"], "cicero-pro-cluentio", None),
    ("cicero", ["tusculan disputations", "tusculanae disputationes"], "cicero-tusculanae-disputationes", "urn:cts:latinLit:phi0474.phi049"),
    ("cicero", ["on divination", "de divinatione"], "cicero-de-divinatione", "urn:cts:latinLit:phi0474.phi053"),
    ("cicero", ["de amicitia", "laelius", "on friendship"], "cicero-de-amicitia", "urn:cts:latinLit:phi0474.phi052"),
    ("cicero", ["de senectute", "on old age"], "cicero-de-senectute", "urn:cts:latinLit:phi0474.phi051"),
    ("cicero", ["de finibus"], "cicero-de-finibus", "urn:cts:latinLit:phi0474.phi048"),
    ("cicero", ["de oratore"], "cicero-de-oratore", None),
    ("cicero", ["letters to his friends", "ad familiares", "epistulae ad familiares"], "cicero-ad-familiares", "urn:cts:latinLit:phi0474.phi056"),
    ("cicero", ["letters to his brother quintus", "ad quintum fratrem"], "cicero-ad-quintum-fratrem", None),
    ("cicero", ["for sextus roscius of ameria", "pro roscio amerino", "pro sexto roscio amerino"], "cicero-pro-roscio-amerino", None),
    ("cicero", ["for fonteius", "pro m. fonteio", "pro fonteio"], "cicero-pro-fonteio", None),
    ("cicero", ["in verrem", "verrines"], "cicero-in-verrem", None),
    ("cicero", ["paradoxa stoicorum"], "cicero-paradoxa-stoicorum", None),
    ("cicero", ["academica"], "cicero-academica", None),
    ("cicero", ["pro caelio"], "cicero-pro-caelio", None),
    ("cicero", ["in vatinium"], "cicero-in-vatinium", None),
    ("cicero", ["post reditum ad quirites"], "cicero-post-reditum-ad-quirites", None),
    ("cicero", ["pro rabirio postumo"], "cicero-pro-rabirio-postumo", None),
    ("ovid", ["metamorphoses"], "ovid-metamorphoses", "urn:cts:latinLit:phi0959.phi006"),
    ("ovid", ["ars amatoria", "art of love"], "ovid-ars-amatoria", "urn:cts:latinLit:phi0959.phi004"),
    ("ovid", ["fasti"], "ovid-fasti", "urn:cts:latinLit:phi0959.phi007"),
    ("ovid", ["heroides", "epistulae heroidum"], "ovid-heroides", "urn:cts:latinLit:phi0959.phi002"),
    ("ovid", ["epistulae ex ponto"], "ovid-epistulae-ex-ponto", None),
    ("ovid", ["ibis"], "ovid-ibis", None),
    ("virgil", ["aeneid"], "virgil-aeneid", "urn:cts:latinLit:phi0690.phi003"),
    ("virgil", ["eclogues"], "virgil-eclogues", "urn:cts:latinLit:phi0690.phi001"),
    ("virgil", ["georgics"], "virgil-georgics", "urn:cts:latinLit:phi0690.phi002"),
    ("horace", ["odes", "carmina"], "horace-odes", "urn:cts:latinLit:phi0893.phi001"),
    ("horace", ["ars poetica", "art of poetry"], "horace-ars-poetica", None),
    ("pliny", ["natural history", "naturalis historia"], "pliny-naturalis-historia", "urn:cts:latinLit:phi0978.phi001"),
    ("augustine", ["city of god", "the city of god", "de civitate dei"], "augustine-de-civitate-dei", None),
    ("augustine", ["contra faustum"], "augustine-contra-faustum", None),
    ("augustine", ["de adulterinis coniugiis"], "augustine-de-adulterinis-coniugiis", None),
    ("seneca the elder", ["controversiae"], "seneca-elder-controversiae", "urn:cts:latinLit:phi1014.phi001"),
    ("seneca", ["de tranquillitate animi"], "seneca-de-tranquillitate-animi", None),
    ("seneca", ["troades"], "seneca-troades", None),
    ("valerius maximus", ["memorable doings and sayings", "facta et dicta memorabilia"], "valerius-maximus-facta-et-dicta", "urn:cts:latinLit:phi1038.phi001"),
    ("aulus gellius", ["attic nights", "noctes atticae"], "gellius-noctes-atticae", "urn:cts:latinLit:phi1254.phi001"),
    ("gellius", ["attic nights", "noctes atticae"], "gellius-noctes-atticae", "urn:cts:latinLit:phi1254.phi001"),
    ("juvenal", ["satires"], "juvenal-satires", "urn:cts:latinLit:phi1276.phi001"),
    ("propertius", ["elegies"], "propertius-elegies", "urn:cts:latinLit:phi0620.phi001"),
    ("plautus", ["menaechmi"], "plautus-menaechmi", None),
    ("plautus", ["amphitryon"], "plautus-amphitruo", None),
    ("plautus", ["poenulus"], "plautus-poenulus", None),
    ("plautus", ["truculentus"], "plautus-truculentus", None),
    ("terence", ["adelphoe"], "terence-adelphoe", None),
    ("livy", ["ab urbe condita", "history of rome"], "livy-ab-urbe-condita", "urn:cts:latinLit:phi0914.phi001"),
    ("sallust", ["bellum catilinae"], "sallust-bellum-catilinae", None),
    ("martial", ["epigrams"], "martial-epigrams", None),
    ("statius", ["thebaid"], "statius-thebaid", None),
    ("manilius", ["astronomica"], "manilius-astronomica", None),
    ("lactantius", ["divine institutes", "divinae institutiones"], "lactantius-divinae-institutiones", None),
    ("solinus", ["collectanea rerum memorabilium"], "solinus-collectanea", None),
    ("macrobius", ["saturnalia"], "macrobius-saturnalia", None),
    ("macrobius", ["commentary on the dream of scipio"], "macrobius-in-somnium-scipionis", None),
    ("justin", ["epitome of the philippic history of pompeius trogus"], "justin-epitome", None),
    ("tertullian", ["apology"], "tertullian-apologeticum", None),
    ("jerome", ["epistle"], "jerome-epistulae", None),
    ("aristotle", ["nicomachean ethics"], "aristotle-nicomachean-ethics", "urn:cts:greekLit:tlg0086.tlg010"),
    ("aristotle", ["history of animals"], "aristotle-history-of-animals", "urn:cts:greekLit:tlg0086.tlg014"),
    ("aristotle", ["politics"], "aristotle-politics", "urn:cts:greekLit:tlg0086.tlg035"),
    ("aristotle", ["metaphysics"], "aristotle-metaphysics", "urn:cts:greekLit:tlg0086.tlg025"),
    ("aristotle", ["generation of animals"], "aristotle-generation-of-animals", None),
    ("aristotle", ["problems"], "aristotle-problems", None),
    ("plato", ["republic"], "plato-republic", "urn:cts:greekLit:tlg0059.tlg030"),
    ("plato", ["phaedo"], "plato-phaedo", "urn:cts:greekLit:tlg0059.tlg004"),
    ("homer", ["iliad"], "homer-iliad", "urn:cts:greekLit:tlg0012.tlg001"),
    ("homer", ["odyssey"], "homer-odyssey", "urn:cts:greekLit:tlg0012.tlg002"),
    ("herodotus", ["histories"], "herodotus-histories", "urn:cts:greekLit:tlg0016.tlg001"),
    ("diogenes laertius", ["lives of the philosophers", "lives of eminent philosophers"], "diogenes-laertius", "urn:cts:greekLit:tlg0004.tlg001"),
    ("pausanias", ["description of greece"], "pausanias-description-of-greece", "urn:cts:greekLit:tlg0525.tlg001"),
    ("strabo", ["geography"], "strabo-geography", "urn:cts:greekLit:tlg0099.tlg001"),
    ("josephus", ["jewish antiquities"], "josephus-antiquitates", "urn:cts:greekLit:tlg0526.tlg001"),
    ("euripides", ["medea"], "euripides-medea", "urn:cts:greekLit:tlg0006.tlg003"),
    ("euripides", ["alcestis"], "euripides-alcestis", "urn:cts:greekLit:tlg0006.tlg002"),
    ("cicero", ["rhetorica", "de inventione"], "cicero-de-inventione", None),
    ("plato", ["laws"], "plato-laws", "urn:cts:greekLit:tlg0059.tlg034"),
    ("appian", ["mithridatic wars"], "appian-mithridatica", None),
    ("appian of alexandria", ["mithridatic wars"], "appian-mithridatica", None),
    ("appian", ["syrian wars"], "appian-syriaca", None),
    ("eusebius of caesarea", ["ecclesiastical history"], "eusebius-historia-ecclesiastica", None),
    ("eusebius", ["ecclesiastical history"], "eusebius-historia-ecclesiastica", None),
    ("iamblichus", ["on the mysteries of the egyptians, chaldaeans and assyrians", "de mysteriis"], "iamblichus-de-mysteriis", None),
    ("platina", ["lives of the popes"], "platina-vitae-pontificum", None),
    ("galen", ["on the usefulness of the parts of the body", "de usu partium", "de semine"], "galen", None),
    ("pseudo-sallust", ["invective against cicero"], "pseudo-sallust-in-ciceronem", None),
    ("baptista fulgosus", ["factorum dictorumque memorabilium libri ix"], "fregoso-de-dictis-factisque-memorabilibus", None),
    ("juan luis vives", ["commentary on augustine, de civitate dei"], "vives-commentary-de-civitate-dei", None),
    ("dio cassius", ["roman history"], "dio-cassius-roman-history", None),
    ("ps.-plutarch", ["de placitis philosophorum"], "pseudo-plutarch-de-placitis", None),
    ("pseudo-plutarch", ["de placitis philosophorum"], "pseudo-plutarch-de-placitis", None),
    ("varro", ["antiquitates rerum divinarum"], "varro-antiquitates", None),
    ("marcus terentius varro", ["antiquitates rerum divinarum"], "varro-antiquitates", None),
]
# Works divided into parts (one unit per life): (author pattern, "Life of X" -> corpus, unit).
PARTS_AUTHORS = {"plutarch": "plutarch-lives", "suetonius": "suetonius-de-vita-caesarum",
                 "julius capitolinus": "historia-augusta", "aelius spartianus": "historia-augusta",
                 "aelius lampridius": "historia-augusta", "vulcacius gallicanus": "historia-augusta",
                 "trebellius pollio": "historia-augusta", "flavius vopiscus": "historia-augusta"}
SCAIFE = "https://scaife.perseus.org/reader/{urn}:{loc}/"
SCAIFE_WORK = "https://scaife.perseus.org/library/{urn}/"

# --------------------------------------------------------------------------- legal corpora

LEGAL_LABEL = {"digest": "D.", "code": "C.", "institutes": "Inst.", "novels": "Nov.",
               "decretals": "X", "sext": "VI", "clementines": "Clem.", "decretum": ""}

# Each pattern yields an explicit reference. Prefix is required so stray numbers are not
# taken. Order matters only for overlap resolution (earlier wins at the same position).
P_DOT = r"(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:\.(\d+|pr))?"
LEGAL_PATTERNS = [
    ("digest", re.compile(r"(?<![\w.])(?:Digest|Dig\.|ff\.)\s*,?\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:\.(\d+|pr))?(?:\(\d+\))?(?=[^\d]|$)")),
    ("digest", re.compile(r"(?<![\w.])D\.\s*(\d+)\.(\d+)(?:\.(\d+))?(?:\.(\d+|pr))?(?:\(\d+\))?(?=[^\d]|$)")),
    ("code", re.compile(r"(?<![\w.])(?:Code|Cod\.|C\.)\s*(\d+)\.(\d+)(?:\.(\d+))?(?:\.(\d+|pr))?(?:\(\d+\))?(?=[^\d]|$)")),
    ("code", re.compile(r"(?<![\w.])Code\s+(\d+)()()()(?=[^\d.]|$)")),
    ("institutes", re.compile(r"(?<![\w.])(?:Institutes|Inst\.)\s*,?\s*(\d+)(?:\.(\d+))?(?:\.(\d+|pr))?()(?=[^\d]|$)")),
    ("novels", re.compile(r"(?<![\w.])(?:Novels?|Nov\.|Authenticum,?\s*Nov\.)\s*(\d+)(?:\.(\d+))?(?:,?\s*(?:c\.|ch\.|cap\.|chapter)\s*(\d+))?()(?=[^\d]|$)")),
    ("decretals", re.compile(r"(?<![\w.])(?:X|Decretals,?)\s*(\d+)\.(\d+)(?:\.(\d+))?()(?=[^\d]|$)")),
    ("sext", re.compile(r"(?<![\w.])(?:VI|Sext\.?|Liber Sextus,?)\s*(\d+|[IVX]+)\.(\d+)(?:\.(\d+))?()(?=[^\d]|$)")),
    ("clementines", re.compile(r"(?<![\w.])(?:Clem\.|Clementines,?)\s*(\d+)\.(\d+)(?:\.(\d+))?()(?=[^\d]|$)")),
    # Decretum: C. n q(q). n[–n | & n] [, c. n]; De cons. D. n [c. n]; D. n [c. n]
    ("decretum", re.compile(r"(?<![\w.])C\.\s*(\d+),?\s*qq?\.\s*(\d+)(?:\s*(?:" + DASH + r"|&|and)\s*\d+)?(?:,?\s*(?:c|can)\.\s*(\d+))?()(?=[^\d]|$)")),
    ("decretum-cons", re.compile(r"(?<![\w.])De\s+cons(?:ecratione)?\.?,?\s*D(?:ist)?\.\s*(\d+)(?:,?\s*c\.\s*(\d+))?()()(?=[^\d]|$)")),
    ("decretum-dist", re.compile(r"(?<![\w.])D\.\s*(\d+)(?!\s*\.\s*\d)(?:,?\s*c\.\s*(\d+))?()()(?=[^\d.]|\.(?!\d)|$)")),
]
TITLE_LEVEL_EXPAND = re.compile(
    r"\b(?:cc?|ll?|can|chapters?|reg)\.?\s+(\d+(?:\s*(?:,|and|&|" + DASH + r")\s*\d+)*)(?![\d.])")
PARA_AFTER = re.compile(r"^\s*,?\s*(?:(pr\.|pr$|pr(?=[\s,;)]))|§§\s*(\d+)\s*" + DASH + r"\s*(\d+)|§\s*(\d+)(?![\d.]))")
DECRETUM_CONTEXT = re.compile(r"Decretum|Gratian|De\s+cons|\bc\.\s*\d|\bcc?\.\s*\*", re.I)

COMMENTARY = re.compile(
    r"^(?P<who>.+?)\s+(?:on|at|after|in)\s+(?:the\s+|a\s+)?(?P<rest>\*?(?:it\b|(?:last|penultimate|first|said)\s|Digest|Code|D\.|C\.|X\b|VI\b|Clem|Decretals|Decretum|"
    r"Institutes|Novel|Nov\.|Authentic|Liber|Clementines|c\.|l\.|ll\.|cc\.|chapter|rule|title|law|fragment|§|LF\b|Libri|summa|Gratian|authentica|\*authentica).*)$", re.S)
LEGAL_WORD = re.compile(r"\b(?:Digest|Code|Decretals|Decretum|Institutes|Novels?|Nov\.|Clementines|Liber Sextus|Authenticum)\b|^\s*(?:X|VI|D\.|C\.)\s*\d")


class Ctx:
    """What exists on disk: corpora, unit files, passages (loaded lazily)."""

    def __init__(self, sources_dir, absent=()):
        self.dir = pathlib.Path(sources_dir)
        self.corpora = {}
        self.absent = set(absent)   # corpora to treat as not fetched (e.g. one being written)
        if self.dir.exists():
            for cj in self.dir.glob("*/corpus.json"):
                try:
                    e = json.loads(cj.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                cid = e.get("id", cj.parent.name) if isinstance(e, dict) else cj.parent.name
                if cid in self.absent or cj.parent.name in self.absent:
                    continue
                self.corpora[cid] = {"entry": e, "dir": cj.parent}
        self._units, self._passages = {}, {}
        self.concordance = {}
        cc = self.dir / "code" / "concordance.json"
        if cc.exists():
            try:
                self.concordance = json.loads(cc.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        self.classical = load_classical(self.dir / "classical-works.json")
        try:
            raw = json.loads((self.dir / "classical-works.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = []
        self.classical_raw = [w for w in (raw if isinstance(raw, list) else raw.get("works", []) if isinstance(raw, dict) else [])
                              if isinstance(w, dict)]

    def units(self, corpus):
        if corpus not in self._units:
            c = self.corpora.get(corpus)
            m = {}
            if c:
                for p in c["dir"].glob("*.json"):
                    if p.name in ("corpus.json", "concordance.json"):
                        continue
                    m[norm(p.stem)] = p
                for u in (c["entry"].get("units") or []) if isinstance(c["entry"], dict) else []:
                    p = c["dir"] / f"{u}.json"
                    if p.exists():
                        m[norm(u)] = p
            self._units[corpus] = m
        return self._units[corpus]

    def passages(self, path):
        return self._load(path)[:2]

    def labels(self, path):
        """{passage id: label} of a unit file."""
        return self._load(path)[2]

    def _load(self, path):
        if path not in self._passages:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                ps = [p for p in data.get("passages", []) if "id" in p]
                ids, unit = [p["id"] for p in ps], data.get("unit", path.stem)
                labels = {p["id"]: p.get("label") or "" for p in ps}
            except (OSError, json.JSONDecodeError, AttributeError):
                ids, unit, labels = [], path.stem, {}
            self._passages[path] = (unit, ids, labels)
        return self._passages[path]

    def entry(self, corpus):
        c = self.corpora.get(corpus)
        return c["entry"] if c and isinstance(c["entry"], dict) else {}


def load_classical(path):
    """classical-works.json, if present: a list (or {id: obj} map, or {"works": [...]}) of
    objects with id and optionally author, title, aliases/titles, urn."""
    out = []
    if not path.exists():
        return out
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return out
    if isinstance(data, dict):
        data = data.get("works", [dict(v, id=k) if isinstance(v, dict) else {"id": k} for k, v in data.items()])
    for w in data if isinstance(data, list) else []:
        if not isinstance(w, dict) or not (w.get("id") or w.get("work_id")):
            continue
        w = dict(w, id=w.get("id") or w["work_id"])
        raw = [w.get("title", ""), w.get("work", "")] + list(w.get("aliases", []) or []) + list(w.get("titles", []) or [])
        titles = []
        for x in raw:   # "Odes (Carmina)", "De usu partium; De semine"
            for y in re.split(r"[;()]", x or ""):
                if y.strip():
                    titles.append(y.strip())
        authors = [w.get("author", "")] + list(w.get("author_aliases", []) or [])
        out.append({"id": w["id"], "authors": [norm(a) for a in authors if a],
                    "titles": [norm(t) for t in titles if t], "urn": w.get("urn")})
    return out


# --------------------------------------------------------------------------- parsing

def make(kind, corpus=None, unit=None, passage=None, passage_end=None, label=None, **extra):
    r = {"kind": kind, "ref": label, "corpus": corpus, "unit": unit, "passage": passage,
         "passage_end": passage_end}
    r.update(extra)
    return r


def legal_candidates(seg):
    masked, depths = mask_parens(seg)
    found = []
    taken = []
    has_decretum_ctx = bool(DECRETUM_CONTEXT.search(seg))
    for corpus, pat in LEGAL_PATTERNS:
        for m in pat.finditer(seg):
            if any(a <= m.start() < b for a, b in taken):
                continue
            if corpus == "decretum-dist" and not (has_decretum_ctx or m.group(2)):
                continue
            pre = seg[max(0, m.start() - 6):m.start()]
            found.append({"corpus": corpus, "m": m, "start": m.start(), "end": m.end(),
                          "depth": depths[m.start()], "cf": bool(re.search(r"cf\.\s*$", pre))})
            taken.append((m.start(), m.end()))
    found.sort(key=lambda c: c["start"])
    return found, masked


SINGLE_TITLE_DIGEST_BOOKS = {"30", "31", "32"}


def b_single(b):
    return str(b) in SINGLE_TITLE_DIGEST_BOOKS


def legal_ref(corpus, g, tail_masked):
    """Build (corpus, unit, passage, passage_end, label) from a match's groups."""
    g = [x for x in g]
    if corpus == "digest" and b_single(g[0]):
        # D. 30-32 have one title each: unit is the book, passages book.fragment[.par]
        b, rest = g[0], [x for x in g[1:] if x]
        if len(rest) >= 2 and rest[0] == "1":
            rest = rest[1:]
        parts = [b] + rest
        if len(parts) == 2:
            pm = PARA_AFTER.match(tail_masked)
            if pm and (pm.group(1) or pm.group(4)):
                parts.append("pr" if pm.group(1) else pm.group(4))
        return dict(corpus=corpus, unit=b, passage=".".join(parts) if rest else None,
                    label=f"D. {'.'.join(parts)}", level=len(parts) + 1)
    if corpus == "digest" or corpus == "code":
        b, t, f, p = g
        if not t:
            return dict(corpus=corpus, unit=b, passage=None, label=f"{LEGAL_LABEL[corpus]} {b}", level=1)
        unit = f"{b}.{t}"
        parts = [b, t] + [x for x in (f, p) if x]
        end = None
        if f and not p:
            pm = PARA_AFTER.match(tail_masked)
            if pm:
                if pm.group(1):
                    parts.append("pr")
                elif pm.group(2):
                    parts.append(pm.group(2))
                    end = ".".join(parts[:-1] + [pm.group(3)])
                else:
                    parts.append(pm.group(4))
        passage = ".".join(parts) if f else None
        return dict(corpus=corpus, unit=unit, passage=passage, passage_end=end,
                    label=f"{LEGAL_LABEL[corpus]} {'.'.join(parts)}", level=len(parts))
    if corpus == "institutes":
        b, t, p, _ = g
        parts = [x for x in (b, t, p) if x]
        return dict(corpus=corpus, unit=b, passage=".".join(parts) if t else None,
                    label=f"Inst. {'.'.join(parts)}", level=len(parts) + 1)
    if corpus == "novels":
        n, c1, c2, _ = g
        c = c1 or c2
        return dict(corpus=corpus, unit=n, passage=f"{n}.{c}" if c else None,
                    label=f"Nov. {n}" + (f" c. {c}" if c else ""), level=3 if c else 2)
    if corpus in ("decretals", "sext", "clementines"):
        b, t, c, _ = g
        b = num(b)
        parts = [x for x in (b, t, c) if x]
        return dict(corpus=corpus, unit=f"{b}.{t}", passage=".".join(parts) if c else None,
                    label=f"{LEGAL_LABEL[corpus]} {'.'.join(parts)}", level=len(parts))
    if corpus == "decretum":
        ca, q, c, _ = g
        unit = f"C.{ca}"
        base = f"C.{ca} q.{q}"
        return dict(corpus="decretum", unit=unit, passage=f"{base} c.{c}" if c else base,
                    label=f"C. {ca} q. {q}" + (f" c. {c}" if c else ""), level=3 if c else 2,
                    question_only=not c)
    if corpus == "decretum-cons":
        d, c, _, _ = g
        unit = f"De cons. D.{d}"
        return dict(corpus="decretum", unit=unit, passage=f"De cons. D.{d} c.{c}" if c else None,
                    label=f"De cons. D. {d}" + (f" c. {c}" if c else ""), level=3 if c else 2)
    if corpus == "decretum-dist":
        d, c, _, _ = g
        return dict(corpus="decretum", unit=f"D.{d}", passage=f"D.{d} c.{c}" if c else None,
                    label=f"D. {d}" + (f" c. {c}" if c else ""), level=3 if c else 2)
    raise ValueError(corpus)


def expand_title(r, items):
    """Title-level ref + chapter/law numbers -> one ref per number (or a range)."""
    out = []
    for it in items:
        rng = re.split(r"\s*" + DASH + r"\s*", it)
        first, last = rng[0], rng[-1] if len(rng) > 1 else None
        c = r["corpus"]
        if c in ("digest", "code", "decretals", "sext", "clementines"):
            pas = f"{r['unit']}.{first}"
            end = f"{r['unit']}.{last}" if last else None
            lab = f"{LEGAL_LABEL[c]} {pas}" + (f"–{last}" if last else "")
        elif c == "novels":
            pas, end = f"{r['unit']}.{first}", (f"{r['unit']}.{last}" if last else None)
            lab = f"Nov. {r['unit']} c. {first}" + (f"–{last}" if last else "")
        elif c == "decretum":
            base = r["passage"] or r["unit"]
            pas, end = f"{base} c.{first}", (f"{base} c.{last}" if last else None)
            lab = f"{r['label']} c. {first}" + (f"–{last}" if last else "")
        else:
            continue
        out.append(dict(r, passage=pas, passage_end=end, label=lab, level=r["level"] + 1,
                        question_only=False))
    return out


def parse_legal(seg):
    seg = seg.replace("*", "")
    cands, masked = legal_candidates(seg)
    if not cands:
        return []
    top = [c for c in cands if c["depth"] == 0]
    refs = []
    for i, c in enumerate(cands):
        nxt = cands[i + 1]["start"] if i + 1 < len(cands) else len(seg)
        tail_masked = masked[c["end"]:nxt]
        r = legal_ref(c["corpus"], c["m"].groups(), masked[c["end"]:])
        r.update(depth=c["depth"], cf=c["cf"], start=c["start"])
        r["tail"] = tail_masked
        refs.append(r)
    top_refs = [r for r in refs if r["depth"] == 0]
    keep = list(top_refs)
    for r in refs:
        if r["depth"] == 0:
            continue
        # a parenthesised ref is kept when it refines a top-level ref of the same corpus
        # and unit (or when the segment has no top-level reference at all)
        if not top_refs or any(t["corpus"] == r["corpus"] and t["unit"] == r["unit"]
                               and r["level"] > t["level"] for t in top_refs):
            keep.append(r)
    if not top and keep:
        noncf = [r for r in keep if not r["cf"]]
        keep = noncf or keep
    out = []
    for r in keep:
        items = []
        title_level = (r["corpus"] in ("digest", "code", "decretals", "sext", "clementines") and r["level"] == 2) \
            or (r["corpus"] == "novels" and r["level"] == 2) \
            or (r["corpus"] == "decretum" and r["passage"] is None or r.get("question_only"))
        if title_level and (r["depth"] == 0 or not top_refs):
            for m in TITLE_LEVEL_EXPAND.finditer(r["tail"]):
                items += [x for x in re.split(r"\s*(?:,|and|&)\s*", m.group(1)) if x]
        out += expand_title(r, items) if items else [r]
    # drop refs that are strict prefixes of other refs in the same corpus
    final = []
    for r in out:
        key = norm(r["passage"] or r["unit"] or "")
        if any(o is not r and o["corpus"] == r["corpus"] and norm(o["passage"] or o["unit"] or "").startswith(key + ".")
               for o in out):
            continue
        if any(f["corpus"] == r["corpus"] and f["unit"] == r["unit"] and f["passage"] == r["passage"]
               and f.get("passage_end") == r.get("passage_end") for f in final):
            continue
        final.append(r)
    final.sort(key=lambda r: r["start"])
    res = []
    for r in final:
        extra = {"cf": True} if r["cf"] else {}
        res.append(make("law", r["corpus"], r["unit"], r["passage"], r.get("passage_end"), r["label"], **extra))
    # Liber Sextus De regulis iuris, reg. N -> VI 5.12.N
    return res


def parse_sext_regula(seg):
    m = re.search(r"(?:Liber Sextus|\bVI\b|Sext)\b.*?De regulis iuris.*?\b(?:reg\.|regula)\s*(\d+)", seg, re.S)
    if m:
        n = m.group(1)      # VI 5.13 De regulis iuris (5.12 is De verborum significatione)
        return [make("law", "sext", "5.13", f"5.13.{n}", None, f"VI 5.13.{n}")]
    return []


BIBLE_ALT = re.compile(
    r"\b(" + "|".join(re.escape(n).replace(r"\ ", r"\s+") for n in _bnames) + r")"
    r"(?:\s*,\s*chapter\s+(\d+)|\s*\((\d+):(\d+)(?:\s*" + DASH + r"\s*(\d+))?)", re.I)


def parse_bible(seg):
    s = plain(seg)
    s = re.sub(r"\s*\[[^\]]*\]", "", s)                 # "3 Kings [1 Kings] 11"
    s = re.sub(r"^((?:[1-4]\s+)?[A-Z][a-z]+)\s*\((?!\d+:)[^)]*\)\s*(?=\d)", r"\1 ", s)  # "2 Samuel (Vulgate 2 Kings) 13"
    masked, depths = mask_parens(s)
    out = []
    hits = [(m, "std") for m in BIBLE_RE.finditer(s)]
    if not hits:
        hits = [(m, "alt") for m in BIBLE_ALT.finditer(s)]
    for m, kind in hits:
        if depths[m.start()] != 0 and out:
            continue
        name, idx = BIBLE_ALIAS[re.sub(r"\s+", " ", m.group(1)).lower()]
        if kind == "std":
            ch, v1, v2 = m.group(2), m.group(3), m.group(4)
            if not v1:
                pm = re.match(r"\s*\((\d+):(\d+)(?:\s*" + DASH + r"\s*(\d+))?", s[m.end():])
                if pm and pm.group(1) == ch:
                    v1, v2 = pm.group(2), pm.group(3)
        elif m.group(2):
            ch, v1, v2 = m.group(2), None, None
        else:
            ch, v1, v2 = m.group(3), m.group(4), m.group(5)
        passage = f"{name} {ch}:{v1}" if v1 else None
        end = f"{name} {ch}:{v2}" if v2 else None
        label = f"{name} {ch}" + (f":{v1}" if v1 else "") + (f"–{v2}" if v2 else "")
        out.append(make("bible", "vulgate", name, passage, end, label, chapter=ch, book_no=idx))
    return out


LOC_PATTERNS = [
    (re.compile(r"pref(?:ace|\.)\s*(\d+|[IVX]+)\s*,?\s*§§?\s*(\d+)(?:\s*" + DASH + r"\s*(\d+))?", re.I), "pref"),
    (re.compile(r"book\s+(\d+|[IVXLC]+),?\s*(?:chapters?|c\.|ch\.)\s*(\d+)(?:\s*(?:" + DASH + r"|and)\s*(\d+))?", re.I), "bc"),
    (re.compile(r"book\s+(\d+|[IVXLC]+),?\s*lines?\s*(\d+)(?:\s*" + DASH + r"\s*(\d+))?", re.I), "bc"),
    (re.compile(r"^\s*,?\s*((?:\d+|[IVXLC]+)(?:\.(?:\d+e?|ext|pr))+)(?:\s*" + DASH + r"\s*((?:\d+|[IVXLC]+)(?:\.\d+)*))?(?![\w])"), "dotted"),
    (re.compile(r"^\s*,?\s*(\d+|[IVXLC]+)(?:\s*\((\d+)\))?(?![\w.:])"), "single"),
    (re.compile(r"book\s+(\d+|[IVXLC]+)\b", re.I), "book"),
    (re.compile(r"\b(?:lines?|vv?\.)\s*(\d+)(?:\s*" + DASH + r"\s*(\d+))?", re.I), "line"),
    (re.compile(r"\b(?:chapter|c\.|ch\.|problem)\s*(\d+)(?:\s*" + DASH + r"\s*(\d+))?", re.I), "line"),
]


def dotted(s):
    parts = s.split(".")
    conv = [num(parts[0])] + parts[1:]
    return ".".join(p for p in conv if p) if conv[0] else None


def classical_loc(rest):
    """Parse the locus after a classical work title. Returns (passage, passage_end)."""
    m = re.search(r"preface to book\s+(\d+|[IVXLC]+)\s*\(\s*§§?\s*(\d+)(?:\s*" + DASH + r"\s*(\d+))?", rest, re.I)
    if m:
        b = num(m.group(1))
        return f"{b}.pr.{m.group(2)}", (f"{b}.pr.{m.group(3)}" if m.group(3) else None), "pref"
    m = re.match(r"^\s*,?\s*\(\s*§§?\s*(\d+)(?:\s*" + DASH + r"\s*(\d+))?", rest)
    if m:                                   # "Pro M. Fonteio (§§ 21–36 ...)"
        return m.group(1), m.group(2), "line"
    masked, _ = mask_parens(rest)
    for pat, kind in LOC_PATTERNS:
        m = pat.search(masked)
        if not m:
            continue
        if kind == "pref":
            b = num(m.group(1))
            return f"{b}.pr.{m.group(2)}", (f"{b}.pr.{m.group(3)}" if m.group(3) else None), kind
        if kind == "bc":
            b = num(m.group(1))
            return f"{b}.{m.group(2)}", (f"{b}.{m.group(3)}" if m.group(3) else None), kind
        if kind == "dotted":
            start = dotted(m.group(1))
            end = None
            if m.group(2):
                e = m.group(2)
                if "." in e:
                    end = dotted(e)
                else:  # 10.300–310 -> 10.310 ; II.9.3–4 -> 2.9.4
                    end = ".".join(start.split(".")[:-1] + [e])
            return start, end, kind
        if kind == "single":
            b = num(m.group(1))
            if b is None:
                continue
            # a book-only locus refined by the parenthesis right after it:
            # "I (1.84: ...)", "I (I.24.59)", "V (203–224)"
            pm = re.match(r"\s*\((?:§+\s*)?((?:\d+|[IVXLC]+)(?:\.\d+)+)", rest[m.end():])
            if pm and dotted(pm.group(1)) and dotted(pm.group(1)).split(".")[0] == b:
                return dotted(pm.group(1)), None, kind
            pm = re.match(r"\s*\((\d+)(?:\s*" + DASH + r"\s*(\d+))?\s*[),;]", rest[m.end():])
            if pm:
                return f"{b}.{pm.group(1)}", (f"{b}.{pm.group(2)}" if pm.group(2) else None), kind
            return b, None, kind
        if kind == "book":
            b = num(m.group(1))
            pm = re.search(r"\(\s*(?:book\s+\w+,\s*)?(\d+(?:\.\d+)+)", rest[m.end():])
            if pm and pm.group(1).split(".")[0] == b:
                return pm.group(1), None, kind
            return b, None, kind
        if kind == "line":
            return m.group(1), (m.group(2) if m.group(2) else None), "line"
    return None, None, None


def _author_ok(cited, listed):
    if not cited or not listed:
        return True
    toks = [t for t in norm(cited).split(".") if len(t) > 3]
    return any(t in listed for t in toks)


def classical_work(ctx, author, title):
    """(work id, CTS URN or None). Alias table first (its ids are the classical fetcher's
    work_ids), then site/data/sources/classical-works.json by title, then a slug."""
    a, t = norm(author), norm(title)
    for auth, titles, wid, urn in CLASSICAL:
        if norm(auth) == a and t in [norm(x) for x in titles]:
            return wid, urn
    works = ctx.classical if ctx else []
    for w in works:
        if t in w["titles"] and any(_author_ok(author, x) for x in (w["authors"] or [""])):
            return w["id"], w.get("urn")
    if len(t) >= 15:
        hits = [w for w in works if any(x == t or (len(x) >= 15 and (x.startswith(t) or t.startswith(x)))
                                        for x in w["titles"])]
        if len(hits) == 1:
            return hits[0]["id"], hits[0].get("urn")
    return slug(f"{author} {title}"), None


def parts_work(ctx, author, title):
    """'Plutarch, *Life of Lycurgus*' -> ('plutarch-lives', 'lycurgus')."""
    corpus = PARTS_AUTHORS.get(norm(author).replace(".", " "))
    m = re.match(r"(?:the\s+)?life of (.+)$", title.strip(), re.I)
    if not corpus or not m:
        return None
    name = slug(re.sub(r"\s+the\s+philosopher$", "", m.group(1), flags=re.I))
    units = []
    if ctx and corpus in ctx.corpora and isinstance(ctx.corpora[corpus]["entry"], dict):
        units = ctx.corpora[corpus]["entry"].get("units") or []
    if units and name not in units:
        words = name.split("-")
        cand = [u for u in units if u in words or any(u == "-".join(words[i:]) for i in range(len(words)))]
        if cand:
            name = cand[0]
    return corpus, name


CLASSICAL_HEAD = re.compile(r"^(?P<author>[A-Z][^*;]*?)(?:\s*\([^)]*\))?,\s*(?:the\s+)?\*(?P<title>[^*]+)\*(?P<rest>.*)$", re.S)
NON_CLASSICAL_WORKS = {"decretum", "glossa.ordinaria"}


def parse_classical(seg, ctx):
    m = CLASSICAL_HEAD.match(seg)
    if not m:
        m2 = re.match(r"^\*(?P<title>[^*]+)\*(?P<rest>.*)$", seg, re.S)
        if not m2:
            return None
        author, title, rest = "", m2.group("title"), m2.group("rest")
    else:
        author, title, rest = m.group("author").strip(), m.group("title").strip(), m.group("rest")
    if norm(title) in NON_CLASSICAL_WORKS:
        return None
    pw = parts_work(ctx, author, title)
    passage, end, lkind = classical_loc(rest)
    if pw:
        corpus, part = pw
        label = f"{author}, {title}" + (f" {passage}" if passage else "") + (f"–{end}" if end else "")
        return make("classical", corpus, part, f"{part}.{passage}" if passage else None,
                    f"{part}.{end}" if end else None, label, _rest=rest, _lkind=lkind)
    wid, urn = classical_work(ctx, author, title)
    label = f"{author + ', ' if author else ''}{title}" + (f" {passage}" if passage else "") + (f"–{end}" if end else "")
    if lkind == "line":      # a line or chapter of an undivided work: unit decided at locate time
        r = make("classical", wid, None, passage, end, label)
    else:
        unit = passage.split(".")[0] if passage else None
        r = make("classical", wid, unit, passage if passage and "." in passage else None,
                 end if passage and "." in passage else None, label)
    if urn:
        r["cts_urn"] = urn + (f":{passage}" + (f"-{end}" if end else "") if passage else "")
        r["_urn"] = urn
    r["_rest"], r["_lkind"] = rest, lkind    # the locus text, for the scheme adapters
    return r


def strip_lead(seg):
    return re.sub(r"^(?:and|also|cf\.|see|the same,?)\s+", "", seg.strip(), flags=re.I).strip(" ,.")


def prenormalize(s):
    """Spelled-out Decretum forms and print variants -> the abbreviated grammar."""
    s = re.sub(r"⟨alt\??:[^⟩]*⟩", "", s)
    s = re.sub(r"\bCausa\s+(\d+)", r"C. \1", s)
    s = re.sub(r"\bquaestio(?:nes)?\s+(\d+)", r"q. \1", s)
    s = re.sub(r"\bDistinctio\s+(\d+)", r"D. \1", s)
    s = re.sub(r"(C\.\s*\d+)\s*\([^)]*\)\s*(qq?\.)", r"\1 \2", s)
    return s


# A commentator, gloss or authentica on a legal text: "Bartolus on D. 48.5.39".
WHO_OK = re.compile(r"^(?:[A-Z]|the\s+\*?(?:Gloss|Glossa|Doctors|Masters|commentators|authentica|Authentica))")
WHO_BAD = re.compile(r"^(?:c|l|cc|ll|§)\.?\s|^(?:The|the)\s+(?:law|chapter|rule|title|fragment|same)\b")
LEGAL_HEAD = re.compile(
    r"^(?:the\s+)?(?:Digest|Dig\.|Code|Institutes|Inst\.|Novels?|Nov\.|Authentic(?:um|a)\b(?!\s*\*)|Decretals|Decretum|"
    r"Liber Sextus|Clementines|Clem\.|Gratian|Libri [Ff]eudorum|LF\s+\d|X\s+\d|VI\s+\d|D\.\s*\d|C\.\s*\d|ff\.|De\s+cons)")
HEAD_CORPUS = [(r"Digest|Dig\.|ff\.", "digest"), (r"Code", "code"), (r"Institutes|Inst\.", "institutes"),
               (r"Novels?|Nov\.|Authenticum", "novels"), (r"Decretals", "decretals"),
               (r"Decretum|Gratian", "decretum"), (r"Liber Sextus", "sext"),
               (r"Clementines|Clem\.", "clementines"), (r"Libri [Ff]eudorum", "libri-feudorum")]
LF = re.compile(r"(?:Libri [Ff]eudorum|LF)\s*(\d+)\.(\d+)")
BACKREF = re.compile(r"as above|cited above|already cited|at the places? cited|place cited|as at \{|the same chapter|"
                     r"as note|\bsaid\b|prealleg|same title|as printed above", re.I)
# Authors cited without a work title, mapped to the one work Coras cites them for.
AUTHOR_ONLY = {
    "valerius maximus": ("Valerius Maximus", "Memorable Doings and Sayings"),
    "valerius": ("Valerius Maximus", "Memorable Doings and Sayings"),
    "livy": ("Livy", "Ab urbe condita"), "herodotus": ("Herodotus", "Histories"),
    "justin": ("Justin", "Epitome of the Philippic History of Pompeius Trogus"),
    "pliny": ("Pliny", "Natural History"), "solinus": ("Solinus", "Collectanea rerum memorabilium"),
    "aulus gellius": ("Aulus Gellius", "Attic Nights"), "gellius": ("Gellius", "Attic Nights"),
    "pietro crinito": ("Pietro Crinito", "De honesta disciplina"),
    "diogenes laertius": ("Diogenes Laertius", "Lives of the Philosophers"),
}
AUTHOR_ONLY_RE = re.compile(r"^(" + "|".join(sorted(map(re.escape, (k.title() for k in AUTHOR_ONLY)), key=len, reverse=True))
                            + r")(?:\s*\([^)]*\))?,?\s+(?=(?:book|chapter|c\.|[IVXLC]+\b|\d))(?P<rest>.*)$", re.S)


def law_work_level(p):
    for pat, corpus in HEAD_CORPUS:
        if re.match(r"^(?:the\s+)?(?:" + pat + r")", p):
            return make("law", corpus, None, None, None, p)
    return None


def parse_segment(seg, ctx=None):
    """One ;-separated part of an identification -> list of refs. Kinds: law, bible,
    classical, commentary, crossref, backref, unidentified, remark (a translator's
    comment split off by ';', not a reference), unparsed."""
    s = prenormalize(strip_lead(seg))
    p = plain(s)
    low = p.lower()
    if not p:
        return []
    if re.match(r"^(?:unidentified|not identified|source not identified)", low):
        return [make("unidentified", label=p)]
    if re.match(r"^(?:Coras(?:'s)?\b.*\bAnnotation|Coras's own Annotation|cross-reference|below, Coras's Annotation)", p) \
            or re.match(r"^Coras,? Annotation", p):
        return [make("crossref", label=p)]
    masked, _ = mask_parens(s)
    if re.match(r"^(?:Liber Sextus|VI\b)", p):
        reg = parse_sext_regula(s)
        if reg:
            return reg
    head_legal = LEGAL_HEAD.match(p)
    cm = None if head_legal else COMMENTARY.match(masked)
    if cm:
        who_m = masked[:cm.end("who")].strip()
        if (not WHO_OK.match(who_m) or WHO_BAD.match(who_m) or LEGAL_WORD.search(who_m)
                or re.search(r",\s*\*[^*]+\*", who_m)):
            cm = None
    if cm:
        who = plain(s[:cm.end("who")])
        rest = s[cm.start("rest"):]
        under = parse_legal(rest) or parse_sext_regula(rest) or parse_lf(rest) or parse_bible(rest)
        if not under:
            wl = law_work_level(plain(rest))
            under = [wl] if wl else []
        if under:
            return [make("commentary", label=p, commentator=who, on=u) for u in under]
        return [make("commentary", label=p, commentator=who)]
    if head_legal:
        refs = parse_legal(s) or parse_lf(s) or parse_bible(s)
        if refs:
            return refs
        if BACKREF.search(p) and not re.search(r"\d", p):
            return [make("backref", label=p)]
        wl = law_work_level(p)
        if wl:
            return [wl]
    if re.match(r"^(?:\d\s+)?[A-Z][a-z]+", p):
        refs = parse_bible(s)
        if refs and re.match(r"^(?:[1-4]\s+)?[A-Z][a-z]+(?:\s+of\s+[A-Z][a-z]+)?\s*(?:\d|\(|\[|,\s*chapter)", p):
            return refs
    cl = parse_classical(s, ctx)
    if cl:
        return [cl]
    am = AUTHOR_ONLY_RE.match(s)
    if am and not BACKREF.search(p):
        author, title = AUTHOR_ONLY[am.group(1).lower()]
        cl = parse_classical(f"{author}, *{title}* " + am.group("rest"), ctx)
        if cl:
            return [cl]
    refs = parse_legal(s) or parse_bible(s)
    if refs and (re.match(r"^(?:Gratian|Decretum|Decretals|c\.|l\.|cc\.|ll\.)", p) or not re.match(r"^[a-z]", p)):
        return refs
    # a lower-case part that names a text outside parentheses ("the text meant is C. 5.18.3")
    if refs and (parse_legal(masked) or parse_bible(masked)):
        return refs
    if BACKREF.search(p):
        return [make("backref", label=p)]
    if re.match(r"^(?:[a-z]|Coras's |Coras cites )", p):
        return [make("remark", label=p)]
    return [make("unparsed", label=p)]


def parse_lf(seg):
    m = LF.search(seg)
    if m:
        return [make("law", "libri-feudorum", f"{m.group(1)}.{m.group(2)}", None, None, f"LF {m.group(1)}.{m.group(2)}")]
    return []


NEW_SOURCE_AFTER_AND = re.compile(
    r",?\s+and\s+(?=(?:Digest|Code|Decretals|Decretum|Novel|Institutes|Gratian|Liber Sextus|Clementines|X\s+\d|"
    r"C\.\s*\d+\s*q|(?:[1-4]\s+)?[A-Z][a-z]+\s+\d+(?::|\s*\(|$)|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*\*))")


def split_and(part):
    """Split at top-level ', and ' / ' and ' when what follows starts a new source."""
    masked, _ = mask_parens(part)
    pieces, i = [], 0
    for m in NEW_SOURCE_AFTER_AND.finditer(masked):
        pieces.append(part[i:m.start()])
        i = m.end()
    pieces.append(part[i:])
    return [p for p in pieces if p.strip()]


def parse_identification(ident, ctx=None):
    """Bold identification text -> list of refs (dicts with kind, ref, corpus, unit,
    passage, passage_end, ...)."""
    refs = []
    for part in split_top(ident, (";",)):
        for piece in split_and(part):
            got = parse_segment(piece, ctx)
            nlaw = sum(1 for r in got if r["kind"] == "law")
            for r in got:
                if r["kind"] == "law":
                    r["_seg"], r["_nseg"] = piece, nlaw   # for the position adapter
            refs += got
    return refs


# --------------------------------------------------------------------------- locating

def apply_concordance(ref, conc):
    if ref.get("corpus") != "code" or not conc or not ref.get("passage"):
        return ref
    for field in ("passage", "passage_end"):
        v = ref.get(field)
        if not v:
            continue
        parts = v.split(".")
        for n in (len(parts), 3):
            key = ".".join(parts[:n])
            if key in conc:
                new = conc[key] + ("." + ".".join(parts[n:]) if parts[n:] else "")
                ref[field] = new
                ref["coras_numbering"] = ref.get("coras_numbering") or v
                ref["unit"] = ".".join(new.split(".")[:2])
                break
    if ref.get("coras_numbering"):
        ref["ref"] = f"C. {ref['passage']} (Coras: {ref['coras_numbering']})"
    return ref


DROITROMAIN = "https://droitromain.univ-grenoble-alpes.fr/Corpus/"


def droitromain_url(ref):
    """Same passage at droitromain.univ-grenoble-alpes.fr. Scheme verified from fetched
    pages: one page per Digest book (d-48.htm), Code book (CJ9.htm), Institutes book
    (just4.gr.htm), Novel (Nov22.htm, Nov117.htm); anchors are the passage ids with 'pr.'
    for the principium (#48.5.39.4, #9.9.4.pr.), or the title (#48.5)."""
    c, unit, pas = ref.get("corpus"), ref.get("unit"), ref.get("passage")
    if not unit:
        return None
    book = str(unit).split(".")[0]
    if not book.isdigit():
        return None
    page = {"digest": f"d-{int(book):02d}.htm", "code": f"CJ{int(book)}.htm",
            "institutes": f"just{int(book)}.gr.htm", "novels": f"Nov{int(book):02d}.htm"}[c]
    anchor = pas or (unit if c in ("digest", "code") and "." in str(unit) else None)
    if anchor:
        anchor = re.sub(r"\.pr$", ".pr.", anchor)
        return f"{DROITROMAIN}{page}#{anchor}"
    return DROITROMAIN + page


def external_url(ref):
    c = ref.get("corpus")
    if c == "vulgate" and ref.get("book_no"):
        return f"https://www.drbo.org/lvb/chapter/{ref['book_no']:02d}{int(ref['chapter']):03d}.htm"
    if c in ("digest", "code", "institutes", "novels"):
        return droitromain_url(ref)
    if ref.get("_urn"):
        p = ref.get("passage") or ref.get("unit")
        if p:
            return SCAIFE.format(urn=ref["_urn"], loc=p + (f"-{ref['passage_end']}" if ref.get("passage_end") else ""))
        return SCAIFE_WORK.format(urn=ref["_urn"])
    return None


def livy_passage(p):
    """Livy is stored part.book.chapter (part = decade): 21.62.5 -> 3.21.62."""
    if not p:
        return p
    b = p.split(".")
    if not b[0].isdigit():
        return p
    return ".".join([str((int(b[0]) - 1) // 10 + 1)] + b[:2])


def locate(ref, ctx):
    """Set status (and refine passage/passage_end) from the files on disk."""
    if ref.get("corpus") == "livy-ab-urbe-condita" and ref.get("passage"):
        ref["passage"], ref["passage_end"] = livy_passage(ref["passage"]), livy_passage(ref.get("passage_end"))
        ref["unit"] = ref["passage"].split(".")[0]
    c = ref.get("corpus")
    if ref["kind"] in ("unidentified", "crossref", "unparsed") or not c:
        ref["status"] = "none"
        return ref
    if ref["kind"] == "commentary":
        ref["status"] = "work"
        return ref
    if c not in ctx.corpora:
        ref["status"] = "work"
        return ref
    units = ctx.units(c)
    unit = ref.get("unit")
    path = units.get(norm(unit)) if unit else None
    if path is None and c == "vulgate" and unit:
        vu = vulgate_unit(ctx, unit)            # the corpus's own book map
        if vu:
            path = units.get(norm(vu))
            if path is None:
                path = split_unit(ctx, c, vu, ref.get("passage") or
                                  (f"{unit} {ref['chapter']}:1" if ref.get("chapter") else None))
    if path is None and ref["kind"] == "classical" and "all" in (ctx.entry(c).get("split_units") or {}):
        path = split_unit(ctx, c, "all", ref.get("passage") or unit)   # one work split in parts
    if path is None and len(units) == 1 and ref["kind"] == "classical":
        path = next(iter(units.values()))  # a work stored as one file
        if unit and not ref.get("passage") and ref.get("unit") == unit and "." not in unit:
            ref["passage"] = unit           # "Pro Cluentio 54": the number is a section
    if path is None and unit:
        path = split_unit(ctx, c, unit, ref.get("passage") or
                          (f"{unit} {ref['chapter']}:1" if ref.get("chapter") else None))
    if path is None:
        ref["status"] = "work"
        return ref
    uname, ids = ctx.passages(path)
    book = ref.get("unit")
    ref["unit"] = uname
    pas = ref.get("passage")
    nids = [norm(i) for i in ids]
    if not pas and ref.get("kind") == "bible" and ref.get("chapter"):
        # a chapter citation: the whole chapter
        pre = norm(f"{book} {ref['chapter']}") + "."
        chap = [i for i, n in zip(ids, nids) if n.startswith(pre)]
        if chap:
            ref["passage"], ref["passage_end"] = chap[0], (chap[-1] if len(chap) > 1 else None)
            ref["status"] = "passage"
            return ref
    if not pas:
        ref["status"] = "unit"
        return ref
    if norm(pas) not in nids and pas.endswith(".pr") and norm(pas[:-3]) in nids:
        pas = ref["passage"] = pas[:-3]
    if norm(pas) not in nids:                   # same id written another way (9.15e.1, 1.2.0)
        pas = ref["passage"] = fuzzy_id(pas, ids) or pas
        if ref.get("passage_end") and norm(ref["passage_end"]) not in nids:
            ref["passage_end"] = fuzzy_id(ref["passage_end"], ids) or ref["passage_end"]
    # cited more finely than the edition divides: fall back to the enclosing passage
    parts = pas.split(".")
    floor = len(str(book).split(".")) + 1 if book and norm(pas).startswith(norm(book) + ".") else 1
    while (len(parts) > floor and norm(".".join(parts)) not in nids
           and not any(n.startswith(norm(".".join(parts)) + ".") for n in nids)):
        parts = parts[:-1]
        if ref.get("passage_end"):
            ref["passage_end"] = None
    if ".".join(parts) != pas and len(parts) >= floor:
        pas = ref["passage"] = ".".join(parts)
    np_ = norm(pas)
    if np_ in nids:
        ref["passage"] = ids[nids.index(np_)]
        hit = True
    else:
        sub = [i for i, n in zip(ids, nids) if n.startswith(np_ + ".")]
        hit = bool(sub)
        if sub:
            ref["passage"] = sub[0]
            if not ref.get("passage_end") and len(sub) > 1:
                ref["passage_end"] = sub[-1]
    if hit and ref.get("passage_end"):
        ne = norm(ref["passage_end"])
        if ne in nids:
            ref["passage_end"] = ids[nids.index(ne)]
        else:
            sub = [i for i, n in zip(ids, nids) if n.startswith(ne + ".")]
            ref["passage_end"] = sub[-1] if sub else None
    ref["status"] = "passage" if hit else "unit"
    return ref


def id_numbers(x):
    return [int(n) for n in re.findall(r"\d+", str(x))]


def split_unit(ctx, corpus, unit, passage):
    """A unit split into parts (corpus.json split_units: {unit: [{unit, passage_range}]}):
    the part whose range holds the passage, or the first part."""
    entry = ctx.corpora[corpus]["entry"]
    parts = (entry.get("split_units") or {}).get(norm(unit).replace(".", "-")) \
        or (entry.get("split_units") or {}).get(unit) if isinstance(entry, dict) else None
    if not parts:
        return None
    units = ctx.units(corpus)
    chosen = parts[0]
    if passage:
        key = id_numbers(passage)
        for prt in parts:
            lo, hi = prt.get("passage_range", [None, None])
            if lo and hi and id_numbers(lo) <= key <= id_numbers(hi):
                chosen = prt
                break
    return units.get(norm(chosen["unit"]))


OUT_FIELDS = ["ref", "corpus", "unit", "passage", "passage_end", "status", "external_url", "scan_url"]

# --------------------------------------------------------------------------- scheme adapters
#
# The Notes cite passages in the conventions of the printed scholarship (book.chapter,
# chapter.section, Stephanus and Bekker pages, "§§ 53–54", "(ch. 18)" in a parenthesis);
# the corpora store them in their edition's scheme (corpus.json passage_scheme). The
# adapters below turn one parsed ref into ordered candidate readings; finalize() keeps the
# first that resolves to a passage in the unit file, else the fallback reading. The corpus
# files are never changed.

# Corpora whose "book.section" ids are sections or pages, not chapters, so a chapter in the
# Notes is not a passage id there (Aristotle's Ethics and Metaphysics, whose Perseus
# "sections" are the chapters, are not in this set).
SECTION_IDS_PREFIXES = ("cicero-", "josephus-", "plato-", "pliny-", "diogenes-", "appian-", "aristotle-politics")
# Corpora where the Notes' bare book.N (VII.10) means book.chapter.
CHAPTER_CITED = {"pliny-naturalis-historia", "aristotle-politics"}
# Unit-file labels that name the chapter: "Plin. NH 7.56 (chapter 12)", "Arist. Pol. 7.1335a (section 16)".
LABEL_CHAPTER = re.compile(r"\((?:chapter|section)\s+(\w+)\)\s*$")
# When a corpus's label chapters follow a division other than the one the Notes use by
# default, the Notes must name it before the labels are used: Pliny's labels carry the old
# (Hardouin/Loeb) chapters, while the translators' bare VII.10 is the modern chapter.
LABEL_DIVISION = {"pliny-naturalis-historia": re.compile(r"\bold(?:er)?\s+(?:chapter\s+)?division", re.I)}
NOT_LOCATED = re.compile(r"not (?:been )?(?:located|found)", re.I)
RN = r"(?:\d+|[IVXLC]+)"


def passage_scheme(ctx, corpus):
    return str(ctx.entry(corpus).get("passage_scheme") or "")


def section_ids(corpus, sch):
    return sch.startswith(("book.section", "section", "book.bekker")) and (
        "label" in sch or corpus.startswith(SECTION_IDS_PREFIXES))


def _alt(ref, tag, **fields):
    r = dict(ref, _adapter=tag)
    r.update(fields)
    return r


def _span(pre, a, b):
    """(pre+a, pre+b or None)."""
    return f"{pre}{a}", (f"{pre}{b}" if b else None)


def classical_candidates(ref, ctx):
    """Candidate readings of a classical ref in its corpus's scheme; returns (candidates,
    fallback). The fallback is the parse itself, except where the parse is a chapter number
    in a corpus whose ids are sections (Pliny VII.10 is not section 7.10): then the
    unit-level reading, so a coincidental id is never claimed."""
    c, rest = ref["corpus"], ref.get("_rest") or ""
    sch = passage_scheme(ctx, c)
    book, pas, lkind = ref.get("unit"), ref.get("passage"), ref.get("_lkind")
    lost = bool(NOT_LOCATED.search(rest))
    out, fallback = [], ref
    if sch.startswith("part."):                                   # Plutarch, Suetonius, HA
        if not pas and book and not lost:
            m = re.search(r"(?<![\w.])(\d+)\.(\d+)(?:\s*" + DASH + r"\s*(\d+))?(?![\d.])", rest)
            if m:
                a, b = _span(f"{book}.{m.group(1)}.", m.group(2), m.group(3))
                out.append(_alt(ref, "part-chapter-section", passage=a, passage_end=b))
            m = re.search(r"\b(?:ch\.|chapters?|c\.)\s*(\d+)(?:\s*" + DASH + r"\s*(\d+))?", rest)
            if m:
                a, b = _span(f"{book}.", m.group(1), m.group(2))
                out.append(_alt(ref, "part-chapter", passage=a, passage_end=b))
        return out, fallback
    if sch.startswith("actio.book"):                              # Verrines II.4 § 39 -> unit 2.4, 2.4.39
        if pas and re.fullmatch(r"\d+\.\d+", pas) and not lost:
            m = re.search(r"§§?\s*(\d+)(?:\s*" + DASH + r"\s*(\d+))?", rest)
            if m:
                a, b = _span(f"{pas}.", m.group(1), m.group(2))
                out.append(_alt(ref, "actio-book-section", unit=pas, passage=a, passage_end=b))
            fallback = _alt(ref, "actio-book", unit=pas, passage=None, passage_end=None)
        return out, fallback
    divided = sch.startswith("book.")
    secs = section_ids(c, sch)

    def P(n):
        return f"{book}.{n}" if divided and book else str(n)

    if secs and not lost:
        m = re.search(r"§§?\s*(\d+)(?:\s*" + DASH + r"\s*(\d+))?", rest)
        if m and (book or not divided):
            out.append(_alt(ref, "section-sign", passage=P(m.group(1)),
                            passage_end=P(m.group(2)) if m.group(2) else None))
        m = re.search(r"(" + RN + r")\.(\d+)(?:\s*" + DASH + r"\s*(\d+))?\s+in the modern", rest)
        if m and num(m.group(1)):
            b = num(m.group(1))
            a, e = _span(f"{b}.", m.group(2), m.group(3))
            out.append(_alt(ref, "modern-numbering", unit=b, passage=a, passage_end=e))
        m = re.search(r"(?<![\w.])" + RN + r"\.\d+(?:\s*" + DASH + r"\s*\d+)?,\s*(\d+)(?:\s*" + DASH + r"\s*(\d+))?(?![\d.])", rest)
        if m and divided and book:                                # De oratore II.86–88, 351–360
            out.append(_alt(ref, "chapter-then-section", passage=P(m.group(1)),
                            passage_end=P(m.group(2)) if m.group(2) else None))
        if pas and divided and re.fullmatch(r"\d+\.\d+\.\d+", pas):  # Tusc. I.24.59 -> 1.59
            e = ref.get("passage_end")
            out.append(_alt(ref, "chapter-section", passage=f"{book}.{pas.split('.')[2]}",
                            passage_end=f"{book}.{e.split('.')[-1]}" if e else None))
        # Stephanus / Bekker pages: Republic 460e -> 5.460, Politics 1335a -> 7.1335a, Phaedo 61c -> 61
        m = re.search(r"(?<![\w.])(\d{2,4})([a-e])(?:\d+)?(?:\s*" + DASH + r"\s*(\d{2,4})[a-e]?)?(?![\w])", rest)
        if m:
            for n, tag in ((m.group(1) + m.group(2), "bekker-page"), (m.group(1), "stephanus-page")):
                out.append(_alt(ref, tag, passage=P(n), passage_end=P(m.group(3)) if m.group(3) and tag == "stephanus-page" else None))
    chapter_cited = secs and pas and divided and re.fullmatch(r"\d+\.\d+", pas) and (
        lkind == "bc" or (c in CHAPTER_CITED and lkind in ("dotted", "single")))
    if (not pas or chapter_cited) and book and not lost:
        # a refinement in the parenthesis: "book 1, letter 1 (Ad Q. fratrem 1.1.37)",
        # "book V (V.3, 1131a)", "book 17, chapter 12 (17.324–338)"
        for m in re.finditer(r"(?<![\w.])(" + RN + r")\.(\d+)(?:\.(\d+))?(?:\s*" + DASH + r"\s*(\d+))?(?![\d.])", rest):
            if num(m.group(1)) != str(book).split(".")[0]:
                continue
            if secs and m.group(3):          # chapter.section where the ids are sections: 7.53.180 -> 7.180
                a, end = f"{book}.{m.group(3)}", (f"{book}.{m.group(4)}" if m.group(4) else None)
            elif secs and c in CHAPTER_CITED:
                continue                     # a bare 7.53 there is a chapter again
            elif m.group(3):
                a = f"{book}.{m.group(2)}.{m.group(3)}"
                end = f"{book}.{m.group(2)}.{m.group(4)}" if m.group(4) else None
            else:
                a, end = f"{book}.{m.group(2)}", (f"{book}.{m.group(4)}" if m.group(4) else None)
            if chapter_cited and a == pas:
                continue
            out.append(_alt(ref, "parenthesis-dotted", passage=a, passage_end=end))
            break
    if not pas and not book and divided and not lost:
        # no locus after the title, only in the parenthesis: "Odyssey (II.93–110; ...)", "Variae (I.45, ...)"
        m = re.search(r"(?<![\w.])(" + RN + r")\.(\d+)(?:\.(\d+))?(?:\s*" + DASH + r"\s*(\d+))?(?![\d.])", rest)
        if m and num(m.group(1)):
            b = num(m.group(1))
            a = f"{b}.{m.group(2)}" + (f".{m.group(3)}" if m.group(3) else "")
            e = (a.rsplit(".", 1)[0] + "." + m.group(4)) if m.group(4) else None
            if secs and m.group(3):
                a, e = f"{b}.{m.group(3)}", (f"{b}.{m.group(4)}" if m.group(4) else None)
            out.append(_alt(ref, "parenthesis-only", unit=b, passage=a, passage_end=e))
    if not pas and not lost and not (secs and divided) and (bool(book) == divided):
        m = re.search(r"\b(?:ch\.|chapters?|c\.)\s*(\d+)(?:\s*" + DASH + r"\s*(\d+))?", rest)
        if m:                                       # Herodotus book 3, chapters 61–79; Tertullian (ch. 13)
            out.append(_alt(ref, "chapter", passage=P(m.group(1)),
                            passage_end=P(m.group(2)) if m.group(2) else None))
    if chapter_cited:
        fallback = dict(ref, passage=None, passage_end=None)
        div = LABEL_DIVISION.get(c)
        if not lost and (div is None or div.search(rest)):
            out.append(_alt(ref, "label-chapter", passage=None, passage_end=None,
                            _label_chapter=pas.split(".")[1]))
    else:
        out.append(ref)
    return out, fallback


def resolve_label_chapter(ref, ctx):
    """Fill passage/passage_end with the first and last passage whose label names the chapter."""
    units = ctx.units(ref["corpus"])
    path = units.get(norm(ref.get("unit") or ""))
    if path is None:
        return ref
    labels = ctx.labels(path)
    want = ref["_label_chapter"]
    hit = [i for i, lab in labels.items() if (m := LABEL_CHAPTER.search(lab)) and num(m.group(1)) == want]
    if hit:
        ref["passage"], ref["passage_end"] = hit[0], (hit[-1] if len(hit) > 1 else None)
    return ref


def vulgate_unit(ctx, name):
    """Unit slug for a book name from vulgate/corpus.json `books` (keys, Latin names,
    abbreviations), e.g. 'Gen.' -> 'genesis', 'Paralipomenon I' -> '1-chronicles'."""
    books = ctx.entry("vulgate").get("books") or {}
    k = norm(name)
    for key, b in books.items():
        if not isinstance(b, dict):
            continue
        names = [key, b.get("latin") or ""] + list(b.get("abbreviations") or [])
        if any(norm(x) == k for x in names if x):
            return b.get("unit")
    return None


def fuzzy_key(x):
    """Id tokens ignoring dots, spaces and colons; 'pr'/'0' principium and Perseus 'e'
    (external examples: 9.15e.1) normalised."""
    toks = [t for t in re.split(r"[^a-z0-9]+", norm(x)) if t]
    out = []
    for i, t in enumerate(toks):
        if t in ("e", "ext", "externa") and out and out[-1].isdigit():
            t = "ext"
        elif t in ("pr", "praef", "princ") or (t == "0" and i == len(toks) - 1 and i > 0):
            t = "pr"
        elif t.isdigit():
            t = str(int(t))
        elif i == 0 and roman_to_int(t.upper()):
            t = str(roman_to_int(t.upper()))
        out.append(t)
    return tuple(out)


def fuzzy_id(x, ids):
    k = fuzzy_key(x)
    hits = [i for i in ids if fuzzy_key(i) == k]
    return hits[0] if len(hits) == 1 else None


def code_candidates(ref, ctx):
    """Code numbering: the concordance (Coras's vulgate numbering -> Krüger) first, then the
    number as given, then the concordance backwards (the Notes already gave a Krüger number
    that the edition stores under the vulgate one)."""
    orig = dict(ref)
    fwd = apply_concordance(dict(ref), ctx.concordance)
    out = [fwd]
    if fwd.get("passage") != orig.get("passage"):
        out.append(orig)
    inv = {v: k for k, v in (ctx.concordance or {}).items()}
    back = apply_concordance(dict(orig), inv)
    if back.get("passage") != orig.get("passage"):
        back.pop("coras_numbering", None)
        back["ref"] = orig.get("ref")
        back["_adapter"] = "concordance-inverse"
        out.append(back)
    return out, fwd


POS_EXPLICIT = re.compile(r"\b(?:last|first|penultimate)\s+(?:chapter|law|constitution|fragment)\s*\(\s*(?:c\.|chapter|l\.|law)\s*(\d+)", re.I)
POS_UNICUS = re.compile(r"\bc\.\s*un\.|\bsingle chapter\b|\bsole (?:law|chapter)\b|\bonly chapter\b", re.I)
POS_WORD = re.compile(r"\b(last|first|penultimate)\s+(?:chapter|law|constitution|fragment)\b|\b[lc]\.\s*\*?(fin)\.", re.I)
POS_UNSURE = re.compile(r"not (?:been )?(?:identified|verified|located)|unverified|as printed|perhaps|\bor\b", re.I)
OTHER_INCIPIT = re.compile(r"\b(?:l|ll|c|cc)\.\s*\*(?!fin\b)")


def position_candidates(ref, ctx):
    """A law named by its place in the title: 'the last chapter', 'l. fin.', 'the first law',
    'c. un.' (its single chapter), or 'last chapter (c. 7)'. Only for a title-level ref that
    is the only law its segment names, with no other incipit and no doubt expressed."""
    seg = ref.get("_seg") or ""
    unit = ref.get("unit")
    if ref.get("passage") or not unit or ref.get("_nseg", 1) != 1 or ref.get("corpus") not in ctx.corpora:
        return []
    m = POS_EXPLICIT.search(seg)
    if m:
        return [_alt(ref, "position-explicit", passage=f"{unit}.{m.group(1)}")]
    if POS_UNSURE.search(seg) or OTHER_INCIPIT.search(seg):
        return []
    path = ctx.units(ref["corpus"]).get(norm(unit))
    if path is None:
        return []
    depth = len(str(unit).split(".")) + 1
    frags = []
    for i in ctx.passages(path)[1]:
        f = ".".join(i.split(".")[:depth])
        if f.split(".")[-1].isdigit() and f not in frags:
            frags.append(f)
    if not frags:
        return []
    if POS_UNICUS.search(seg):
        return [_alt(ref, "position-unicus", passage=frags[0])] if len(frags) == 1 else []
    m = POS_WORD.search(seg)
    if not m or re.match(r"\s*\*\w", seg[m.end():]):
        return []                       # "the first chapter *Veniens*": the incipit decides, not the place
    word = (m.group(1) or m.group(2)).lower()
    pick = {"first": 0, "last": -1, "fin": -1, "penultimate": -2}[word]
    if len(frags) < abs(pick) + (1 if pick >= 0 else 0):
        return []
    return [_alt(ref, f"position-{'last' if word == 'fin' else word}", passage=frags[pick])]


def candidates(ref, ctx):
    c = ref.get("corpus")
    if ref.get("kind") == "classical" and c in ctx.corpora:
        return classical_candidates(ref, ctx)
    pos = position_candidates(ref, ctx) if ref.get("kind") == "law" else []
    if c == "code":
        cands, fb = code_candidates(ref, ctx)
        return pos + cands, fb
    return pos + [ref], ref


def ocr_position(ref, ctx):
    """OCR'd canon-law corpora (corpus.json quality 'ocr'): chapter N absent but the N-th
    passage carries a misread number between N-1 and N+1 (Clem. 5.3: '5.3.50', '5.3.2')."""
    if ref.get("status") != "unit" or ctx.entry(ref["corpus"]).get("quality") != "ocr":
        return ref
    m = re.fullmatch(r"(\d+\.\d+)\.(\d+)", ref.get("passage") or "")
    path = ctx.units(ref["corpus"]).get(norm(ref.get("unit") or ""))
    if not m or path is None:
        return ref
    pre, n = m.group(1), int(m.group(2))
    _, ids = ctx.passages(path)
    if not (1 <= n <= len(ids)):
        return ref
    ok_prev = n == 1 or ids[n - 2] == f"{pre}.{n - 1}"
    ok_next = n == len(ids) or ids[n] == f"{pre}.{n + 1}"
    if ok_prev and ok_next and f"{pre}.{n}" not in ids and ids[n - 1].startswith(pre + "."):
        ref["passage"], ref["status"], ref["_adapter"] = ids[n - 1], "passage", "ocr-position"
    return ref


def source_gap(ref, ctx):
    """Why a unit-level hit has no passage, when the corpus says so (corpus.json
    empty_at_source; passages flagged greek_not_online; units flagged no_chapters)."""
    e = ctx.entry(ref["corpus"])
    pas = ref.get("passage")
    if pas:
        parts = pas.split(".")
        for n in range(len(parts), 2, -1):
            if ".".join(parts[:n]) in (e.get("empty_at_source") or []):
                return "empty_at_source"
    path = ctx.units(ref["corpus"]).get(norm(ref.get("unit") or ""))
    if path is not None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        if data.get("no_chapters"):
            return "no_chapters"
        if pas and any(p.get("greek_not_online") and str(p.get("id", "")).startswith(pas) for p in data.get("passages", [])):
            return "greek_not_online"
    return None


def classical_entry(ctx, corpus):
    for w in getattr(ctx, "classical_raw", []):
        if w.get("work_id") == corpus or w.get("id") == corpus:
            return w
    return None


def finalize(ref, ctx):
    if ref.get("kind") == "commentary" and ref.get("on"):
        on = finalize(dict(ref["on"]), ctx)
        ref["on"] = on
        ref["corpus"], ref["unit"] = on.get("corpus"), on.get("unit")
        ref["passage"], ref["passage_end"] = on.get("passage"), on.get("passage_end")
        locate(ref, ctx)
    else:
        cands, fallback = candidates(ref, ctx)
        hit = None
        for cand in cands:
            cand = dict(cand)
            if cand.get("_label_chapter"):
                cand = resolve_label_chapter(cand, ctx)
                if not cand.get("passage"):
                    continue
            r = locate(cand, ctx)
            if r["status"] == "passage":
                hit = r
                break
        ref = hit or locate(dict(fallback), ctx)
    if ref.get("status") == "unit" and ref.get("corpus") in ctx.corpora:
        ref = ocr_position(ref, ctx)
        if ref["status"] == "unit":
            gap = source_gap(ref, ctx)
            if gap:
                ref["source_gap"] = gap
    if ref.get("status") == "work" and ref.get("kind") in ("classical", "law"):
        w = classical_entry(ctx, ref.get("corpus"))
        if w and w.get("status") == "scan-only" and w.get("source"):
            ref["status"], ref["scan_url"] = "scan", w["source"]
        elif w and w.get("status") == "not-found":
            ref["source_gap"] = "not_found"
        elif ref.get("corpus") in ctx.corpora and ctx.entry(ref["corpus"]).get("scan_url_template"):
            ref["scan_url"] = ctx.entry(ref["corpus"])["scan_url_template"]   # whole-volume scan
    if ref.get("_urn"):
        p = ref.get("passage")
        ref["cts_urn"] = ref["_urn"] + (f":{p}" + (f"-{ref['passage_end']}" if ref.get("passage_end") else "") if p else "")
    ref["external_url"] = external_url(ref)
    ref.setdefault("scan_url", None)
    out = {k: ref.get(k) for k in OUT_FIELDS}
    out["kind"] = ref["kind"]
    for k in ("cf", "commentator", "cts_urn", "coras_numbering", "source_gap"):
        if ref.get(k):
            out[k] = ref[k]
    if ref.get("_adapter"):
        out["adapter"] = ref["_adapter"]
    if ref.get("on"):
        out["on"] = {k: v for k, v in ref["on"].items() if k in OUT_FIELDS + ["kind"]}
    return out


# --------------------------------------------------------------------------- driver

def read_notes(sections_dir):
    """Yield (key, identification, section_id) for each keyed Notes line."""
    for f in sorted(pathlib.Path(sections_dir).glob("*.md")):
        text = f.read_text(encoding="utf-8")
        _, sep, notes = text.partition("\n## Notes")
        if not sep:
            continue
        for line in notes.splitlines():
            m = NOTE.match(line)
            if not m:
                continue
            marker, page, rest = m.groups()
            b = BOLD.search(rest)
            yield f"{page}:{marker}", (b.group(1) if b else None), f.stem


def build(root=ROOT, absent=()):
    ctx = Ctx(root / "site/data/sources", absent)
    cites, unparsed, remarks, nobold = {}, Counter(), Counter(), 0
    for key, ident, _sec in read_notes(root / "translation/sections"):
        if ident is None:
            nobold += 1
            continue
        for r in parse_identification(ident, ctx):
            if r["kind"] == "remark":
                remarks[r["ref"]] += 1
                continue
            if r["kind"] == "unparsed":
                unparsed[r["ref"]] += 1
            cites.setdefault(key, []).append(finalize(r, ctx))
    return cites, unparsed, remarks, nobold, ctx


def report(cites, unparsed, remarks, nobold, ctx, out=sys.stdout):
    refs = [r for v in cites.values() for r in v]
    kinds = Counter(r["kind"] for r in refs)
    status = Counter(r["status"] for r in refs)
    parsed = len(refs) - kinds["unparsed"]
    print(f"keys {len(cites)}; notes without a bold identification {nobold}; "
          f"remarks split off by ';' (not references) {sum(remarks.values())}", file=out)
    print(f"refs {len(refs)}: parsed {parsed}, unparsed {kinds['unparsed']}", file=out)
    print("kinds: " + ", ".join(f"{k} {v}" for k, v in kinds.most_common()), file=out)
    print("status: " + ", ".join(f"{k} {v}" for k, v in status.most_common()), file=out)
    byc = Counter(r["corpus"] for r in refs if r["corpus"] and r["kind"] in ("law", "bible", "classical"))
    print("corpora present: " + (", ".join(sorted(ctx.corpora)) or "(none)"), file=out)
    print("refs by corpus (top 20): " + ", ".join(f"{k} {v}" for k, v in byc.most_common(20)), file=out)
    print("classical ids: " + ("site/data/sources/classical-works.json" if ctx.classical
                                else "built-in alias table, else author-title slug (no classical-works.json)"), file=out)
    print(f"unparsed identifications ({len(unparsed)} distinct):", file=out)
    for k, v in unparsed.most_common():
        print(f"  {v}x {k}", file=out)


def main(argv):
    absent = [argv[i + 1] for i, a in enumerate(argv) if a == "--absent" and i + 1 < len(argv)]
    cites, unparsed, remarks, nobold, ctx = build(absent=absent)
    if "--report-only" not in argv:
        out = ROOT / "site/data/citations.json"
        out.write_text(json.dumps(cites, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {out.relative_to(ROOT)}")
    if "--unparsed-file" in argv:
        p = pathlib.Path(argv[argv.index("--unparsed-file") + 1])
        p.write_text("\n".join(unparsed) + "\n", encoding="utf-8")
    report(cites, unparsed, remarks, nobold, ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
