# scripts/tests/test_fetch_canon_law.py
"""Offline tests for scripts/fetch_canon_law.py.

The OCR fixture below imitates the archive.org text of Friedberg vol. 2: running heads,
column numbers, apparatus blocks, OCR-mangled numerals ('n', 'm', 'CAl*.'), a lowercase
'titulus ir.', a lost chapter heading and a book break without a LIBER line. No network.

    uv run --with pytest python -m pytest scripts/tests/test_fetch_canon_law.py
"""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import fetch_canon_law as fcl  # noqa: E402

OCR = """\
SEXTI  DECRETALIUM

LIBER  PRIMUS.


TITULUS  PRIMUS.

DE  SUMMA  TRINITATE  ET  FIDE  CATHOLICA.

CAP.  UN.

Spiritus  sanctus  aeternaliter  a Patre  et  Filio  procedit.  Ioann.  Andr.

Gregorius  X.  in  generali  concilio  Lugdunensi».

Fideli  ac  devota  professione  fatemur,  quod  Spiritus
sanctus  aeternaliter  ex  Patre  et  Filio,  non  tanquam  ex  duo-
bus principiis  procedit.

TITULUS  n.

DE  CONSTITUTIONIBUS.

CAP.  i.

Nova  constitutio  tollit  primam  contrariam.  Ioann.  Andr.

Bonifacius  VIII.

Licet  Romanus  Pontifex  constitutionem  condendo  posterio-
rem, priorem  revocare1*  noscatur.

Tit.  I.  Cap.  un.  a)  Cone.  Lugd.  II.  c.  i.  (1274.);  Comp.
I.,  un.  b)  ex  duab. : He  c)  et:  deest:  Had

943  SEXTI  DECRETAL.  LIB.  I.  TIT.  II.  DE  CONSTITUTIONIBUS,  c.  2.

CAl*.  II.

Statuta  ordinariorum  non  ligant  ignorantes.  Ioann.  Andr.
Idem.
Ut  animarum  periculis  obvietur.

titulus  ir.

D£  RESCRIPTIS.

Ipso  iure  rescriptum  non  valeat.

cap.  n.

Per  clausulam:  Quidam  alii.

TITULUS  I.

DE  IUDICIIS.

CAP.  I.

Si  is,  qui  in  iudicio  convenitur.
"""


def segmented():
    return fcl.segment_ocr(OCR.split("\n"), "sext")


# --- numerals -----------------------------------------------------------------

@pytest.mark.parametrize("s,v", [("XIV", 14), ("xl", 40), ("MCM", 1900), ("IV.", 4)])
def test_roman(s, v):
    assert fcl.roman(s) == v


def test_roman_rejects_non_numerals():
    assert fcl.roman("PRIMUS") is None
    assert fcl.roman("") is None


@pytest.mark.parametrize("s,v,clean", [
    ("XIV.", 14, True), ("n", 2, False), ("m", 3, False), ("XVHL", 18, False),
    ("xxrv", 24, False), ("ffl", 3, False), ("XXX VH", 37, False), ("XIIL", 13, False),
])
def test_ocr_roman(s, v, clean):
    assert fcl.ocr_roman(s) == (v, clean)


def test_plausible_trusts_clean_and_repairs_garbled():
    assert fcl.plausible("IX", 3) == 9            # clean numeral trusted even if a jump
    assert fcl.plausible("m", 2) == 3             # OCR repair that fits the sequence
    assert fcl.plausible("m", 7) == 8             # OCR repair that does not: prev + 1
    assert fcl.plausible("@@", None) == 1


# --- OCR segmenter ----------------------------------------------------------

def test_segment_units_and_ids():
    u = segmented()
    assert list(u) == ["1.1", "1.2", "1.3", "2.1"]
    assert [p["id"] for p in u["1.2"]["passages"]] == ["1.2.1", "1.2.2"]
    assert u["1.1"]["passages"][0]["label"] == "VI 1.1.1"
    assert u["1.2"]["title"] == "VI 1.2 De Constitutionibus."


def test_segment_joins_hyphenation_and_keeps_paragraphs():
    t = segmented()["1.1"]["passages"][0]["text"]
    assert "duobus principiis" in t
    assert t.count("\n\n") == 2                   # summary / inscription / text


def test_segment_drops_apparatus_and_running_heads():
    t = segmented()["1.2"]["passages"][0]["text"]
    assert "deest" not in t and "Comp." not in t
    t2 = segmented()["1.2"]["passages"][1]["text"]
    assert "DECRETAL" not in t2 and t2.startswith("Statuta")


def test_segment_strips_footnote_calls():
    assert "revocare noscatur" in segmented()["1.2"]["passages"][0]["text"]


def test_segment_lost_chapter_heading_becomes_chapter_one():
    ps = segmented()["1.3"]["passages"]
    assert [p["id"] for p in ps] == ["1.3.1", "1.3.2"]
    assert ps[0]["text"].startswith("Ipso iure")


def test_segment_title_restart_starts_next_book():
    assert segmented()["2.1"]["passages"][0]["text"].startswith("Si is")


def test_slice_vol2():
    text = "\n".join(["front", "SEXTI  DECRETALIUM", "a", "CLEMENTIS  PAPAE  V. ", "b",
                      "EXTRA VAGANTES", "c"])
    parts = fcl.slice_vol2(text)
    assert parts["sext"][0].startswith("SEXTI") and parts["sext"][-1] == "a"
    assert parts["clementines"][-1] == "b"


# --- Decretum (MDZ pages) ---------------------------------------------------

def mdz(head, *lines):
    body = "<br />\n".join(lines)
    return (f'<html><h2 class="content">{head}</h2>\n{head}<br />\n{body}<br />'
            f'<form class="turn"></form></html>')


def test_mdz_page_extracts_heading_and_text():
    head, text = fcl.mdz_page(mdz("C. IV. De his, qui maleficiis inpediti",
                                  "Si per sortiarias atque maleficas", "occulto Dei iudicio"))
    assert head == "C. IV. De his, qui maleficiis inpediti"
    assert text.endswith("Si per sortiarias atque maleficas occulto Dei iudicio")


def test_decretum_units_state_machine():
    raw = [
        (4, "DISTINCTIO PRIMA.", "GRATIANUS. Humanum genus duobus regitur."),
        (5, "C. I. Diuinae leges natura constant.", "Omnes leges aut diuinae sunt."),
        (1077, "DECRETI PARS SECUNDA", ""),
        (3375, "CAUSA XXXIII.", "GRATIANUS. Quidam uir maleficiis inpeditus."),
        (3376, "QUESTIO I.", "GRATIANUS. Quod autem propter inpossibilitatem."),
        (3377, "C. I. Licet mulieri alteri nubere.", "Quod autem interrogasti."),
        (3378, "[C. II.] Item ex epistola eiusdem.", "De his requisisti."),
        (3400, "QUESTIO III.", "GRATIANUS. Utrum sola cordis contritione."),
        (3401, "DISTINCTIO I.", "GRATIANUS. Utrum sola contritione."),
        (3402, "C. I. Lacrimae lauant delictum.", "Petrus doluit et fleuit."),
        (3500, "QUESTIO IV.", "GRATIANUS. Quod autem tempore orationis."),
        (3767, "DECRETI PARS TERTIA DE CONSECRATIONE", ""),
        (3842, "DISTINCTIO II.", "Comperimus autem."),
        (3843, "C. V. Corpus Christi.", "Text."),
    ]
    u = fcl.decretum_units(raw)
    assert set(u) == {"D.1", "C.33", "de-pen-D.1", "de-cons-D.2"}
    assert [p["id"] for p in u["D.1"]["passages"]] == ["D.1 pr", "D.1 c.1"]
    assert [p["id"] for p in u["C.33"]["passages"]] == [
        "C.33 pr", "C.33 q.1 pr", "C.33 q.1 c.1", "C.33 q.1 c.2", "C.33 q.4 pr"]
    assert "De pen. D.1 c.1" in [p["id"] for p in u["de-pen-D.1"]["passages"]]
    assert u["C.33"]["passages"][2]["label"] == "C. 33 q. 1 c. 1 (Licet mulieri alteri nubere.)"
    assert u["de-cons-D.2"]["passages"][-1]["id"] == "De cons. D.2 c.5"


# --- Liber Extra (Augustana) ------------------------------------------------

AUG = """<DL><SPAN CLASS="f_viridis">T i t u l u s&nbsp;&nbsp; X V . <BR>
De frigidis et maleficiatis, <BR>
et impotentia coeundi.</SPAN><BR>
<DT><SPAN CLASS="f_roseus">____</SPAN><BR>
<DT><SPAN CLASS="f_ruber">Capitulum I.</SPAN><BR>
Si, marito provocante ad divortium.<BR>
<BR>
<DD>Ex Brocardico libr. XIX.<BR>
<BR>
Accepisti <i>mulierem</i> et habuisti.<BR>
<BR><DT><SPAN CLASS="f_ruber">CapitulumII.</SPAN><BR>
Impotens ad copulam.<BR>
</DL>
<A HREF="gre_0000.html">&lt;&lt;&lt; operis index</A>"""


def test_augustana_title():
    title, chaps = fcl.augustana_title(AUG)
    assert title == "Titulus XV. De frigidis et maleficiatis, et impotentia coeundi."
    assert [c for c, _ in chaps] == [1, 2]
    assert chaps[0][1].split("\n\n") == ["Si, marito provocante ad divortium.",
                                         "Ex Brocardico libr. XIX.",
                                         "Accepisti mulierem et habuisti."]
    assert "index" not in chaps[1][1]


# --- writing ----------------------------------------------------------------

def test_unit_sort_key_orders_decretum_parts():
    units = ["C.2", "de-cons-D.1", "D.10", "C.10", "de-pen-D.1", "D.2", "C.33-b", "C.33-a"]
    assert sorted(units, key=fcl.unit_sort_key) == [
        "D.2", "D.10", "C.2", "C.10", "C.33-a", "C.33-b", "de-pen-D.1", "de-cons-D.1"]
    assert sorted(["4.15", "2.24", "4.2"], key=fcl.unit_sort_key) == ["2.24", "4.2", "4.15"]


def test_split_unit_at_passage_boundaries():
    u = {"corpus": "decretum", "unit": "C.33", "title": "Decretum, C. 33",
         "passages": [{"id": f"C.33 q.1 c.{i}", "label": "", "text": "x" * 1000}
                      for i in range(1, 11)]}
    parts = fcl.split_unit(u, limit=4000)
    assert [p["unit"] for p in parts] == ["C.33-a", "C.33-b", "C.33-c"]
    assert sum(len(p["passages"]) for p in parts) == 10
    assert parts[1]["passages"][0]["id"] == "C.33 q.1 c.%d" % (len(parts[0]["passages"]) + 1)
    assert fcl.split_unit(u) == [u]


def test_write_corpus(tmp_path):
    units = {"4.15": {"corpus": "decretals", "unit": "4.15", "title": "X 4.15",
                      "passages": [{"id": "4.15.7", "label": "X 4.15.7", "text": "t"}]},
             "4.16": {"corpus": "decretals", "unit": "4.16", "title": "X 4.16",
                      "passages": []}}
    meta = fcl.write_corpus(tmp_path, {"id": "decretals", "quality": "clean"}, units)
    assert meta["units"] == ["4.15"] and meta["passages"] == 1 and meta["status"] == "text"
    c = json.loads((tmp_path / "decretals" / "corpus.json").read_text())
    assert c["units"] == ["4.15"]
    u = json.loads((tmp_path / "decretals" / "4.15.json").read_text())
    assert u["passages"][0]["id"] == "4.15.7"


def test_write_corpus_empty_is_scan_only(tmp_path):
    meta = fcl.write_corpus(tmp_path, {"id": "sext"}, {})
    assert meta["status"] == "scan-only" and meta["units"] == []
