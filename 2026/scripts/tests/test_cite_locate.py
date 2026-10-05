import json, pathlib, sys
import pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import cite_locate as cl
import build_sources_index as bsi


def refs(s):
    return [(r["kind"], r["corpus"], r["unit"], r["passage"], r["passage_end"])
            for r in cl.parse_identification(s)]


def first(s):
    return refs(s)[0]


# ------------------------------------------------------------------ grammar

@pytest.mark.parametrize("ident,expected", [
    ("Digest 48.5.39.4", ("law", "digest", "48.5", "48.5.39.4", None)),
    ("D. 48.5.39 pr.", ("law", "digest", "48.5", "48.5.39.pr", None)),
    ("Digest 23.2.?? (*De ritu nuptiarum*, l. *Minorem*)", ("law", "digest", "23.2", None, None)),
    ("Digest 29.5.3 § 1 (*De senatus consulto Silaniano*, l. 3)", ("law", "digest", "29.5", "29.5.3.1", None)),
    ("Digest 18.1.28 (*De contrahenda emptione*, l. *Rem alienam*)", ("law", "digest", "18.1", "18.1.28", None)),
    ("Digest 32, l. 39 (*De legatis* III)", ("law", "digest", "32", "32.39", None)),
    ("Code 9.9.29", ("law", "code", "9.9", "9.9.29", None)),
    ("C. 4.30.13", ("law", "code", "4.30", "4.30.13", None)),
    ("Code 9.13 (*De raptu virginum*), the sole law (C. 9.13.1, Justinian)", ("law", "code", "9.13", "9.13.1", None)),
    ("Institutes 4.6.25", ("law", "institutes", "4", "4.6.25", None)),
    ("Institutes 1.10 (*De nuptiis*), § *Adfinitatis* (Inst. 1.10.6)", ("law", "institutes", "1", "1.10.6", None)),
    ("Novel 90 c. 7", ("law", "novels", "90", "90.7", None)),
    ("Nov. 22 § *Per occasionem*", ("law", "novels", "22", None, None)),
    ("Authenticum, Nov. 131.11, § *Si autem*", ("law", "novels", "131", "131.11", None)),
    ("Decretum C. 33 q. 1 c. 4", ("law", "decretum", "C.33", "C.33 q.1 c.4", None)),
    ("C. 32 q. 7 c. 16", ("law", "decretum", "C.32", "C.32 q.7 c.16", None)),
    ("D. 10 c. 3", ("law", "decretum", "D.10", "D.10 c.3", None)),
    ("Decretum Gratiani D. 61, c. *Statuimus*", ("law", "decretum", "D.61", None, None)),
    ("De cons. D. 2 c. 5", ("law", "decretum", "De cons. D.2", "De cons. D.2 c.5", None)),
    ("Decretum of Gratian, Part II, Causa 2, quaestio 7, c. 47 *Quapropter*", ("law", "decretum", "C.2", "C.2 q.7 c.47", None)),
    ("X 4.15.7", ("law", "decretals", "4.15", "4.15.7", None)),
    ("Decretals 2.19.3", ("law", "decretals", "2.19", "2.19.3", None)),
    ("Decretals, X 2.20 (*De testibus*), c. *Sicut* (X 2.20.9, Alexander III)", ("law", "decretals", "2.20", "2.20.9", None)),
    ("VI 5.11.6", ("law", "sext", "5.11", "5.11.6", None)),
    ("Liber Sextus, *De regulis iuris*, reg. 54, *Qui prior est tempore*", ("law", "sext", "5.13", "5.13.54", None)),
    ("Clem. 2.1.1", ("law", "clementines", "2.1", "2.1.1", None)),
    ("Genesis 17:5", ("bible", "vulgate", "Genesis", "Genesis 17:5", None)),
    ("Genesis 29:23–25 (Coras cites chapter 30)", ("bible", "vulgate", "Genesis", "Genesis 29:23", "Genesis 29:25")),
    ("Genesis 19 (19:30–38, Lot and his daughters)", ("bible", "vulgate", "Genesis", "Genesis 19:30", "Genesis 19:38")),
    ("1 Kings 3", ("bible", "vulgate", "1 Kings", None, None)),
    ("1 Samuel 28", ("bible", "vulgate", "1 Kings", None, None)),
    ("3 Kings [1 Kings] 11", ("bible", "vulgate", "3 Kings", None, None)),
    ("Psalm 11", ("bible", "vulgate", "Psalms", None, None)),
    ("Malachi, chapter 4 (Malachi 4:5)", ("bible", "vulgate", "Malachi", "Malachi 4:5", None)),
    ("Cicero, *On Duties* 1.10", ("classical", "cicero-de-officiis", "1", "1.10", None)),
    ("Ovid, *Metamorphoses* 10.300–310", ("classical", "ovid-metamorphoses", "10", "10.300", "10.310")),
    ("Augustine, *City of God* 15.16", ("classical", "augustine-de-civitate-dei", "15", "15.16", None)),
    ("Pliny, *Natural History* 7.3", ("classical", "pliny-naturalis-historia", "7", "7.3", None)),
    ("Pliny, *Natural History*, book 7, chapter 14 as printed", ("classical", "pliny-naturalis-historia", "7", "7.14", None)),
    ("Propertius, *Elegies* II.9.3–4", ("classical", "propertius-elegies", "2", "2.9.3", "2.9.4")),
    ("Seneca the Elder, *Controversiae* pref. 1 §§17–19", ("classical", "seneca-elder-controversiae", "1", "1.pr.17", "1.pr.19")),
    ("Cicero, *Tusculan Disputations* I (1.84: Callimachus's epigram)", ("classical", "cicero-tusculanae-disputationes", "1", "1.84", None)),
    ("Plautus, *Menaechmi*, lines 1089–1090", ("classical", "plautus-menaechmi", None, "1089", "1090")),
    ("Valerius Maximus IX.14 ext.", ("classical", "valerius-maximus-facta-et-dicta", "9", "9.14", None)),
    ("Pietro Crinito, *De honesta disciplina* VI.11", ("classical", "pietro-crinito-de-honesta-disciplina", "6", "6.11", None)),
    ("Plutarch, *Life of Lycurgus*, chapter 15", ("classical", "plutarch-lives", "lycurgus", "lycurgus.15", None)),
    ("Homer, *Odyssey* V (203–224)", ("classical", "homer-odyssey", "5", "5.203", "5.224")),
    ("Cicero, *Pro M. Fonteio* (§§ 21–36, on the Gallic witnesses)", ("classical", "cicero-pro-fonteio", None, "21", "36")),
])
def test_grammar(ident, expected):
    assert first(ident) == expected


def test_cicero_pro_cluentio_section():
    r = cl.parse_identification("Cicero, *Pro Cluentio* 54")[0]
    assert r["corpus"] == "cicero-pro-cluentio" and r["unit"] == "54" and r["passage"] is None


def test_title_with_law_numbers_expands():
    assert [r[3] for r in refs("Digest 48.4 (*Ad legem Iuliam maiestatis*), ll. 1, 2 and 4")] == \
        ["48.4.1", "48.4.2", "48.4.4"]
    assert [r[3] for r in refs("Decretals, X 4.17 (*Qui filii*), c. 2 (*Quum inter*), c. 10 *Referente* and c. 14 *Ex tenore*")] == \
        ["4.17.2", "4.17.10", "4.17.14"]
    assert [r[3] for r in refs("C. 34 (the print's xxxiij) qq. 1–2, cc. 6 (*In lectum*) and 5 (*Si virgo*)")] == \
        ["C.34 q.1 c.6", "C.34 q.1 c.5"]


def test_parenthesised_refinements_kept_others_dropped():
    out = refs("Digest 48.5 (*Ad legem Iuliam*), l. *Miles* (D. 48.5.12, Papinian; vulgate l. 11), and the "
               "penultimate law of the title (D. 48.5.44, Gaius; vulgate l. 43)")
    assert [r[3] for r in out] == ["48.5.12", "48.5.44"]
    out = cl.parse_identification("Decretals X 2.20, c. *Cum causam* (cf. X 2.20.37; X 2.19.13 fits better)")
    assert [(r["passage"], r.get("cf")) for r in out] == [("2.20.37", True)]


def test_multiple_refs_semicolon_and_and():
    out = refs("Matthew 19 (19:9); 1 Corinthians 7 (7:10–11)")
    assert [r[3] for r in out] == ["Matthew 19:9", "1 Corinthians 7:10"]
    out = refs("Genesis 19 (19:30–38); and Gratian, *Decretum*, C. 15 q. 1 c. 9 (*Inebriaverunt*)")
    assert [r[1] for r in out] == ["vulgate", "decretum"] and out[1][3] == "C.15 q.1 c.9"
    out = refs("Digest 2.10.1, § 1, and Decretum, D. 86, c. *Si quid*, as at {a}")
    assert [(r[1], r[3] or r[2]) for r in out] == [("digest", "2.10.1.1"), ("decretum", "D.86")]


def test_commentary():
    r = cl.parse_identification("Bartolus on D. 48.5.39")[0]
    assert r["kind"] == "commentary" and r["commentator"] == "Bartolus"
    assert r["on"]["corpus"] == "digest" and r["on"]["passage"] == "48.5.39"
    r = cl.parse_identification("Accursius (*Accurſe*), the Gloss on Digest 22.5.3 (*De testibus*, l. 3), § *Eiusdem* (D. 22.5.3.2)")[0]
    assert r["kind"] == "commentary" and r["on"]["passage"] == "22.5.3.2"
    r = cl.parse_identification("the *Glossa ordinaria* on the Decretum, Part III, *De consecratione*, D. 1, c. *Sicut*")[0]
    assert r["kind"] == "commentary" and r["on"]["unit"] == "De cons. D.1"


def test_unidentified_crossref_backref_remark():
    assert first("unidentified")[0] == "unidentified"
    assert first("unidentified (sigla incomplete: no canon of D. 10 begins *Sive adulterium*)")[0] == "unidentified"
    assert first("Coras, Annotation II (annot-002)")[0] == "crossref"
    assert first("Solinus, at the place cited above (at {c})")[0] == "backref"
    out = cl.parse_identification("Digest 47.2 (*De furtis*), l. *Verum*; fragment not identified")
    assert [r["kind"] for r in out] == ["law", "remark"]
    assert first("The *Extravagans* *Ad reprimendum* of Henry VII")[0] == "unparsed"


def test_cf_prefix_stripped():
    assert first("cf. Code 9.9.29")[3] == "9.9.29"


# ------------------------------------------------------------------ keys

def test_keys_from_notes(tmp_path):
    d = tmp_path / "translation/sections"
    d.mkdir(parents=True)
    (d / "annot-001.md").write_text(
        "---\nid: annot-001\n---\nBody {a}\n\n## Notes\n"
        "- {a} (p040): **Digest 48.5.39.4** — l. 39. [gloss]\n"
        "- {a2} (p072): **Genesis 17:5; Code 9.9.29** — Gen. xvij.\n"
        "- {_} (p110): **unidentified** — c. sive adult.\n"
        "- {ſ} (p017): **Psalm 11** — Psal. xj.\n"
        "- {c} (p044) — marker with no note in the margin.\n"
        "- Verse (p092): **Virgil, *Eclogues* 8.69** — Carmina\n", encoding="utf-8")
    keys = [(k, i) for k, i, _ in cl.read_notes(d)]
    assert keys == [("p040:a", "Digest 48.5.39.4"), ("p072:a2", "Genesis 17:5; Code 9.9.29"),
                    ("p110:_", "unidentified"), ("p017:ſ", "Psalm 11"), ("p044:c", None)]


# ------------------------------------------------------------------ status with a fixture corpus

def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")


@pytest.fixture
def ctx(tmp_path):
    src = tmp_path / "site/data/sources"
    write(src / "digest/corpus.json", {"id": "digest", "units": ["48.5"]})
    write(src / "digest/48.5.json", {"corpus": "digest", "unit": "48.5", "passages": [
        {"id": "48.5.1", "text": "a"}, {"id": "48.5.39.pr", "text": "b"},
        {"id": "48.5.39.1", "text": "c"}, {"id": "48.5.39.4", "text": "d"}]})
    write(src / "code/corpus.json", {"id": "code"})
    write(src / "code/9.9.json", {"corpus": "code", "unit": "9.9", "passages": [
        {"id": "9.9.29", "text": "x"}, {"id": "9.9.30.pr", "text": "y"}]})
    write(src / "code/concordance.json", {"9.9.31": "9.9.29"})
    write(src / "vulgate/corpus.json", {"id": "vulgate", "split_units": {"psalms": [
        {"unit": "psalms-a", "passage_range": ["Psalms 1:1", "Psalms 76:21"]},
        {"unit": "psalms-b", "passage_range": ["Psalms 77:1", "Psalms 150:6"]}]}})
    write(src / "vulgate/genesis.json", {"corpus": "vulgate", "unit": "genesis", "passages": [
        {"id": "Genesis 17:4", "text": "."}, {"id": "Genesis 17:5", "text": "."}, {"id": "Genesis 18:1", "text": "."}]})
    write(src / "vulgate/psalms-b.json", {"corpus": "vulgate", "unit": "psalms-b", "passages": [
        {"id": "Psalms 90:1", "text": "."}, {"id": "Psalms 90:2", "text": "."}]})
    write(src / "cicero-pro-cluentio/corpus.json", {"id": "cicero-pro-cluentio"})
    write(src / "cicero-pro-cluentio/all.json", {"corpus": "cicero-pro-cluentio", "unit": "all", "passages": [
        {"id": "53", "text": "."}, {"id": "54", "text": "."}]})
    return cl.Ctx(src)


def located(s, ctx):
    return [cl.finalize(r, ctx) for r in cl.parse_identification(s, ctx)]


def test_status_passage_unit_work_none(ctx):
    assert located("Digest 48.5.39.4", ctx)[0]["status"] == "passage"
    r = located("D. 48.5.39", ctx)[0]           # fragment -> its paragraphs
    assert (r["status"], r["passage"], r["passage_end"]) == ("passage", "48.5.39.pr", "48.5.39.4")
    assert located("Digest 48.5.77", ctx)[0]["status"] == "unit"
    assert located("Digest 48.5 (*Ad legem Iuliam*), l. *Miles*", ctx)[0]["status"] == "unit"
    assert located("Digest 1.1.1", ctx)[0]["status"] == "work"        # unit not fetched
    assert located("X 4.15.7", ctx)[0]["status"] == "work"            # corpus not fetched
    assert located("unidentified", ctx)[0]["status"] == "none"
    assert located("Bartolus on D. 48.5.39.4", ctx)[0]["status"] == "work"
    assert located("Bartolus on D. 48.5.39.4", ctx)[0]["on"]["status"] == "passage"


def test_status_bible_chapter_and_split_units(ctx):
    r = located("Genesis 17", ctx)[0]
    assert (r["status"], r["unit"], r["passage"], r["passage_end"]) == ("passage", "genesis", "Genesis 17:4", "Genesis 17:5")
    r = located("Psalm 90", ctx)[0]
    assert (r["status"], r["unit"], r["passage"]) == ("passage", "psalms-b", "Psalms 90:1")
    assert r["external_url"] == "https://www.drbo.org/lvb/chapter/21090.htm"


def test_single_file_classical_work(ctx):
    r = located("Cicero, *Pro Cluentio* 54", ctx)[0]
    assert (r["status"], r["unit"], r["passage"]) == ("passage", "all", "54")


def test_concordance_applied(ctx):
    r = located("Code 9.9.31", ctx)[0]
    assert (r["passage"], r["status"], r["coras_numbering"]) == ("9.9.29", "passage", "9.9.31")
    assert located("Code 9.9.29", ctx)[0]["passage"] == "9.9.29"
    r = cl.apply_concordance({"corpus": "code", "unit": "9.9", "passage": "9.9.31.2"}, {"9.9.31": "9.10.4"})
    assert (r["unit"], r["passage"]) == ("9.10", "9.10.4.2")


def test_external_urls(ctx):
    assert located("Digest 48.5.39 pr.", ctx)[0]["external_url"] == \
        "https://droitromain.univ-grenoble-alpes.fr/Corpus/d-48.htm#48.5.39.pr."
    assert located("Novel 90 c. 7", ctx)[0]["external_url"] == \
        "https://droitromain.univ-grenoble-alpes.fr/Corpus/Nov90.htm#90.7"
    assert located("Ovid, *Metamorphoses* 10.300–310", ctx)[0]["external_url"] == \
        "https://scaife.perseus.org/reader/urn:cts:latinLit:phi0959.phi006:10.300-10.310/"
    assert located("Pietro Crinito, *De honesta disciplina* VI.11", ctx)[0]["external_url"] is None


def test_classical_works_json_overrides(tmp_path):
    src = tmp_path / "s"
    write(src / "classical-works.json", [{"work_id": "crinito-dhd", "author": "Pietro Crinito",
                                          "work": "De honesta disciplina (Commentarii)"}])
    c = cl.Ctx(src)
    assert cl.parse_identification("Pietro Crinito, *De honesta disciplina* VI.11", c)[0]["corpus"] == "crinito-dhd"
    assert cl.parse_identification("Crinito, *Commentarii* 2.1", c)[0]["corpus"] == "crinito-dhd"
    # the alias table's ids are the fetcher's work ids
    assert cl.parse_identification("Cicero, *On Duties* 1.10", c)[0]["corpus"] == "cicero-de-officiis"


# ------------------------------------------------------------------ index builder

def test_build_sources_index(tmp_path):
    src = tmp_path / "sources"
    write(src / "digest/corpus.json", {"id": "digest", "title": "Digest", "units": ["48.5"]})
    write(src / "code/corpus.json", {"id": "code", "title": "Code"})
    write(src / "code/9.9.json", {"passages": []})
    write(src / "code/9.10.json", {"passages": []})
    write(src / "code/concordance.json", {})
    data = bsi.collect(src)
    assert [c["id"] for c in data["corpora"]] == ["code", "digest"]
    assert data["corpora"][0]["units"] == ["9.9", "9.10"]
    assert bsi.collect(src) == data


def test_finer_than_edition_falls_back_to_enclosing_passage(ctx):
    r = located("Digest 48.5.39.9", ctx)[0]
    assert (r["status"], r["passage"], r["passage_end"]) == ("passage", "48.5.39.pr", "48.5.39.4")
    assert located("Digest 48.5.77.2", ctx)[0]["status"] == "unit"   # never widened to the whole title


def test_livy_decade_parts():
    assert cl.livy_passage("21.62.5") == "3.21.62"
    assert cl.livy_passage("1.39") == "1.1.39"


def test_novel_named_by_title_with_chapter():
    assert first("Novels, *Ut nulli iudicum* (Nov. 134), chapter 10, § *Si vero*")[3] == "134.10"


def test_norm_matches_spacing_variants():
    assert cl.norm("C.33 q.1 c.4") == cl.norm("C33 q1 c4") == cl.norm("c-33-q-1-c-4")
    assert cl.norm("1 Kings") == cl.norm("1-kings")
    assert cl.slug("Cicero De officiis") == "cicero-de-officiis"


# ------------------------------------------------------------------ scheme adapters

def _unit(src, corpus, unit, ids, labels=None, **extra):
    ps = [{"id": i, "label": (labels or {}).get(i, ""), "text": "t " + i} for i in ids]
    write(src / corpus / f"{unit}.json", dict({"corpus": corpus, "unit": unit, "passages": ps}, **extra))


@pytest.fixture
def actx(tmp_path):
    src = tmp_path / "site/data/sources"
    write(src / "pliny-naturalis-historia/corpus.json", {"id": "pliny-naturalis-historia", "unit_scheme": "book",
          "passage_scheme": "book.section (the chapter is given in the label)"})
    _unit(src, "pliny-naturalis-historia", "7", [f"7.{n}" for n in range(1, 60)],
          {f"7.{n}": f"Plin. NH 7.{n} (chapter {3 if 33 <= n <= 35 else 12 if n > 50 else 2})" for n in range(1, 60)})
    write(src / "cicero-tusculanae-disputationes/corpus.json", {"id": "cicero-tusculanae-disputationes",
          "passage_scheme": "book.section"})
    _unit(src, "cicero-tusculanae-disputationes", "1", [f"1.{n}" for n in range(1, 120)])
    write(src / "cicero-de-oratore/corpus.json", {"id": "cicero-de-oratore", "passage_scheme": "book.section"})
    _unit(src, "cicero-de-oratore", "2", [f"2.{n}" for n in range(1, 370)])
    write(src / "plato-republic/corpus.json", {"id": "plato-republic", "passage_scheme": "book.section"})
    _unit(src, "plato-republic", "5", [f"5.{n}" for n in range(449, 481)])
    write(src / "plato-phaedo/corpus.json", {"id": "plato-phaedo", "passage_scheme": "section"})
    _unit(src, "plato-phaedo", "all", [str(n) for n in range(57, 119)])
    write(src / "aristotle-politics/corpus.json", {"id": "aristotle-politics",
          "passage_scheme": "book.bekker_page (the section is given in the label)"})
    _unit(src, "aristotle-politics", "7", ["7.1334b", "7.1335a", "7.1335b"],
          {"7.1334b": "Arist. Pol. 7.1334b (section 15)", "7.1335a": "Arist. Pol. 7.1335a (section 16)",
           "7.1335b": "Arist. Pol. 7.1335b (section 16)"})
    write(src / "historia-augusta/corpus.json", {"id": "historia-augusta", "passage_scheme": "part.chapter.section",
          "units": ["marcus", "hadrian"]})
    _unit(src, "historia-augusta", "marcus", [f"marcus.19.{n}" for n in range(1, 8)])
    write(src / "plutarch-lives/corpus.json", {"id": "plutarch-lives", "passage_scheme": "part.chapter.section",
          "units": ["pyrrhus"]})
    _unit(src, "plutarch-lives", "pyrrhus", ["pyrrhus.18.1", "pyrrhus.18.2"])
    write(src / "cicero-ad-quintum-fratrem/corpus.json", {"id": "cicero-ad-quintum-fratrem",
          "passage_scheme": "book.letter.section"})
    _unit(src, "cicero-ad-quintum-fratrem", "1", ["1.1.36", "1.1.37"])
    write(src / "tertullian-apologeticum/corpus.json", {"id": "tertullian-apologeticum", "passage_scheme": "chapter[.section]"})
    _unit(src, "tertullian-apologeticum", "all", ["12.1", "13.1", "13.2", "14.1"])
    write(src / "appian-mithridatica/corpus.json", {"id": "appian-mithridatica",
          "passage_scheme": "section (the chapter is given in the label)",
          "split_units": {"all": [{"unit": "alla", "passage_range": ["1", "111"]},
                                  {"unit": "allb", "passage_range": ["112", "121"]}]}})
    _unit(src, "appian-mithridatica", "alla", ["1", "111"])
    _unit(src, "appian-mithridatica", "allb", ["112", "121"])
    write(src / "cicero-in-verrem/corpus.json", {"id": "cicero-in-verrem", "passage_scheme": "actio.book.section"})
    _unit(src, "cicero-in-verrem", "2.4", ["2.4.38", "2.4.39"])
    write(src / "josephus-antiquitates/corpus.json", {"id": "josephus-antiquitates", "passage_scheme": "book.section"})
    _unit(src, "josephus-antiquitates", "17", [f"17.{n}" for n in range(1, 340)])
    write(src / "valerius-maximus-facta-et-dicta/corpus.json", {"id": "valerius-maximus-facta-et-dicta",
          "passage_scheme": "book.chapter.section"})
    _unit(src, "valerius-maximus-facta-et-dicta", "9", ["9.15.5", "9.15.ext.1"])
    write(src / "vulgate/corpus.json", {"id": "vulgate", "books": {
        "1 Chronicles": {"unit": "1-chronicles", "latin": "Paralipomenon I", "abbreviations": ["1 Par."]}}})
    _unit(src, "vulgate", "1-chronicles", ["1 Chronicles 1:1"])
    write(src / "code/corpus.json", {"id": "code", "empty_at_source": ["4.20.6"]})
    _unit(src, "code", "4.20", ["4.20.2", "4.20.11.pr"])
    _unit(src, "code", "9.9", ["9.9.31"])
    write(src / "code/concordance.json", {"9.9.31": "9.9.29"})
    write(src / "digest/corpus.json", {"id": "digest"})
    _unit(src, "digest", "1.2", ["1.2.2.pr", "1.2.2.1"])
    write(src / "clementines/corpus.json", {"id": "clementines", "quality": "ocr"})
    _unit(src, "clementines", "5.3", ["5.3.50", "5.3.2", "5.3.3"])
    write(src / "sext/corpus.json", {"id": "sext", "units": ["5.12"], "scan_url_template": "https://archive.org/details/X"})
    _unit(src, "sext", "5.12", ["5.12.1"])
    write(src / "classical-works.json", [
        {"work_id": "crinito-de-honesta-disciplina", "author": "Pietro Crinito", "work": "De honesta disciplina",
         "status": "scan-only", "source": "https://archive.org/details/crinito"},
        {"work_id": "galen", "author": "Galen", "work": "De usu partium", "status": "not-found"}])
    return cl.Ctx(src)


def loc(s, ctx):
    r = located(s, ctx)[0]
    return r["status"], r["unit"], r["passage"], r["passage_end"], r.get("adapter")


@pytest.mark.parametrize("ident,expected", [
    # Pliny: ids are sections, the Notes cite book.chapter
    ("Pliny, *Natural History* VII.10 (§53–54; Coras's c. 12 follows the old chapter division)",
     ("passage", "7", "7.53", "7.54", "section-sign")),
    ("Pliny, *Natural History*, book 7, chapter 12 (7.53 in the modern numbering)",
     ("passage", "7", "7.53", None, "modern-numbering")),
    ("Pliny, *Natural History* VII, chapter 4 as printed — at 7.53 (7.53.55–56), chapter 53 in the old division",
     ("passage", "7", "7.55", "7.56", "parenthesis-dotted")),
    ("Pliny, *Natural History* VII.3 in the old chapter division (cf. modern VII.4–6)",
     ("passage", "7", "7.33", "7.35", "label-chapter")),
    ("Pliny, *Natural History* VII.10", ("unit", "7", None, None, None)),   # never the coincidental section 7.10
    ("Pliny, *Natural History* VII, chapter 4 as printed", ("unit", "7", None, None, None)),
    # Cicero book.section: chapter.section and chapter-then-section
    ("Cicero, *Tusculan Disputations* I (I.24.59)", ("passage", "1", "1.59", None, "chapter-section")),
    ("Cicero, *De oratore* II (II.86–88, 351–360)", ("passage", "2", "2.351", "2.360", "chapter-then-section")),
    ("Cicero, *Tusculan Disputations* I (1.84: Callimachus)", ("passage", "1", "1.84", None, None)),
    # Stephanus and Bekker pages
    ("Plato, *Republic* book 5 (460e)", ("passage", "5", "5.460", None, "stephanus-page")),
    ("Plato, *Phaedo* (61c–62c)", ("passage", "all", "61", "62", "stephanus-page")),
    ("Aristotle, *Politics* VII.16 (1335a)", ("passage", "7", "7.1335a", None, "bekker-page")),
    ("Aristotle, *Politics* VII.16", ("passage", "7", "7.1335a", "7.1335b", "label-chapter")),
    # parts (life names)
    ("Julius Capitolinus, *Life of Marcus Antoninus the Philosopher* (*Historia Augusta*, *Marcus* 19.1–7)",
     ("passage", "marcus", "marcus.19.1", "marcus.19.7", "part-chapter-section")),
    ("Plutarch, *Life of Pyrrhus* (ch. 18, Cineas's embassy)", ("passage", "pyrrhus", "pyrrhus.18.1", "pyrrhus.18.2", "part-chapter")),
    ("Julius Capitolinus, *Life of Marcus Antoninus*; chapter not located", ("unit", "marcus", None, None, None)),
    # refinements in the parenthesis
    ("Cicero, *Letters to his brother Quintus*, book 1, letter 1 (*Ad Q. fratrem* 1.1.37)",
     ("passage", "1", "1.1.37", None, "parenthesis-dotted")),
    ("Josephus, *Jewish Antiquities*, book 17, chapter 12 (17.324–338)",
     ("passage", "17", "17.324", "17.338", "parenthesis-dotted")),
    ("Tertullian, *Apology* (ch. 13: the statue)", ("passage", "all", "13.1", "13.2", "chapter")),
    ("Appian of Alexandria, *Mithridatic Wars* (ch. 112)", ("passage", "allb", "112", None, "chapter")),
    ("Cicero, *Verrines* II.4 (*De signis*), § 39", ("passage", "2.4", "2.4.39", None, "actio-book-section")),
])
def test_scheme_adapters(ident, expected, actx):
    assert loc(ident, actx) == expected


def test_fuzzy_ids(actx):
    assert loc("Valerius Maximus IX.15e.1", actx)[:3] == ("passage", "9", "9.15.ext.1")
    assert loc("Digest 1.2.2.0", actx)[:3] == ("passage", "1.2", "1.2.2.pr")
    assert cl.fuzzy_key("9.15e.1") == cl.fuzzy_key("9.15.ext.1") == cl.fuzzy_key("IX.15 ext. 1")
    assert cl.fuzzy_id("7.53", ["7.53?", "7.54"]) == "7.53?"
    assert cl.fuzzy_id("1.2", ["12", "1.2.3"]) is None     # dots are not dropped between numbers


def test_vulgate_books_map(actx):
    assert cl.vulgate_unit(actx, "Paralipomenon I") == cl.vulgate_unit(actx, "1 Par.") == "1-chronicles"
    r = cl.locate({"kind": "bible", "corpus": "vulgate", "unit": "1 Par.", "passage": "1 Chronicles 1:1",
                   "passage_end": None}, actx)
    assert (r["status"], r["unit"]) == ("passage", "1-chronicles")


def test_code_concordance_both_ways(actx):
    # forward: Coras's 9.9.31 is Krüger 9.9.29, absent here; the number as given is present
    assert loc("Code 9.9.31", actx)[:3] == ("passage", "9.9", "9.9.31")
    # backward: the Notes give the Krüger number, the file stores it under the vulgate one
    r = located("Code 9.9.29", actx)[0]
    assert (r["status"], r["passage"], r.get("adapter")) == ("passage", "9.9.31", "concordance-inverse")


def test_ocr_position_and_gaps(actx):
    r = located("Clementines, Clem. 5.3 (*De haereticis*), c. 1 (*Multorum querela*)", actx)[0]
    assert (r["status"], r["passage"], r["adapter"]) == ("passage", "5.3.50", "ocr-position")
    assert located("Clem. 5.3.3", actx)[0]["passage"] == "5.3.3"
    r = located("Code 4.20.6", actx)[0]
    assert (r["status"], r["source_gap"]) == ("unit", "empty_at_source")


def test_sext_regulae_iuris_is_title_13(actx):
    r = located("Liber Sextus, *De regulis iuris*, reg. 54, *Qui prior est tempore*", actx)[0]
    assert (r["unit"], r["passage"], r["status"]) == ("5.13", "5.13.54", "work")
    assert r["scan_url"] == "https://archive.org/details/X"


def test_classical_works_scan_and_not_found(actx):
    r = located("Pietro Crinito, *De honesta disciplina* VI.11", actx)[0]
    assert (r["status"], r["scan_url"]) == ("scan", "https://archive.org/details/crinito")
    r = located("Galen, *De usu partium* 14.10", actx)[0]
    assert (r["status"], r["source_gap"]) == ("work", "not_found")


def test_absent_corpus_ignored(tmp_path):
    src = tmp_path / "s"
    write(src / "decretum/corpus.json", {"id": "decretum"})
    _unit(src, "decretum", "C.33", ["C.33 q.1 c.4"])
    assert located("Decretum C. 33 q. 1 c. 4", cl.Ctx(src))[0]["status"] == "passage"
    assert located("Decretum C. 33 q. 1 c. 4", cl.Ctx(src, ["decretum"]))[0]["status"] == "work"


def test_position_in_title(tmp_path):
    src = tmp_path / "s"
    write(src / "decretals/corpus.json", {"id": "decretals"})
    _unit(src, "decretals", "4.15", [f"4.15.{n}" for n in range(1, 8)])
    write(src / "sext/corpus.json", {"id": "sext"})
    _unit(src, "sext", "3.15", ["3.15.1"])
    write(src / "digest/corpus.json", {"id": "digest"})
    _unit(src, "digest", "22.3", ["22.3.28", "22.3.29.pr", "22.3.29.1"])
    c = cl.Ctx(src)
    assert loc("Decretals, X 4.15 (*De frigidis*), last chapter, and the Gloss there", c) == \
        ("passage", "4.15", "4.15.7", None, "position-last")
    assert loc("Decretals, X 4.15, last chapter (c. 6 *Litterae*)", c)[2:] == ("4.15.6", None, "position-explicit")
    assert loc("Liber Sextus, VI 3.15 (*De voto*), its single chapter (c. un., Boniface VIII)", c)[2] == "3.15.1"
    assert loc("Digest 22.3 (*De probationibus*), last law", c)[2:4] == ("22.3.29.pr", "22.3.29.1")
    assert loc("Digest 22.3 (*De probationibus*), the penultimate law", c)[2] == "22.3.28"
    # doubt, another incipit, or an incipit naming the chapter: left at the title
    assert loc("Decretals X 4.15, the last chapter (chapter not identified)", c)[0] == "unit"
    assert loc("Decretals, X 4.15, c. *Fraternitatis* and the last chapter of the same title", c)[0] == "unit"
    assert loc("Decretals, X 4.15, the first chapter *Veniens*", c)[0] == "unit"


def test_locus_only_in_parenthesis(tmp_path):
    src = tmp_path / "s"
    write(src / "homer-odyssey/corpus.json", {"id": "homer-odyssey", "passage_scheme": "book.line"})
    _unit(src, "homer-odyssey", "2", [f"2.{n}" for n in range(90, 112)])
    write(src / "pausanias-description-of-greece/corpus.json", {"id": "pausanias-description-of-greece",
          "passage_scheme": "book.chapter.section"})
    _unit(src, "pausanias-description-of-greece", "6", ["6.8.1", "6.8.2"])
    c = cl.Ctx(src)
    assert loc("Homer, *Odyssey* (II.93–110; XIX.137–156, the web of Penelope)", c) == \
        ("passage", "2", "2.93", "2.110", "parenthesis-only")
    assert loc("Pausanias, *Description of Greece*, the *Eliaca* (book VI, 6.8.2: Damarchus)", c)[:3] == \
        ("passage", "6", "6.8.2")
