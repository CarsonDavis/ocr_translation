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
    ("Liber Sextus, *De regulis iuris*, reg. 54, *Qui prior est tempore*", ("law", "sext", "5.12", "5.12.54", None)),
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
