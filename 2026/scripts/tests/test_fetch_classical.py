# scripts/tests/test_fetch_classical.py
"""Fixture tests for scripts/fetch_classical.py: Perseus TEI and Latin Library HTML parsing,
unit grouping/splitting and the Notes tally. All fixtures are inline; no network.

    python3 -m pytest scripts/tests/test_fetch_classical.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import fetch_classical as fc  # noqa: E402


def tei(refs: str, body: str) -> bytes:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><titleStmt>
<title>Fixture Work</title></titleStmt></fileDesc>
<encodingDesc><refsDecl n="CTS">{refs}</refsDecl></encodingDesc></teiHeader>
<text><body><div type="edition" n="urn:cts:latinLit:phi9999.phi001.perseus-lat2">
{body}</div></body></text></TEI>""".encode()


def pat(n, xpath):
    return f'<cRefPattern n="{n}" matchPattern="x" replacementPattern="#xpath({xpath})"/>'


PROSE = tei(
    pat("section", "/tei:TEI/tei:text/tei:body/tei:div/tei:div[@n='$1']/tei:div[@n='$2']"
                   "/tei:div[@n='$3']")
    + pat("book", "/tei:TEI/tei:text/tei:body/tei:div/tei:div[@n='$1']"),
    """<head>Title to skip</head>
<div type="textpart" subtype="book" n="1"><head>LIBER I</head>
 <div type="textpart" subtype="chapter" n="1">
  <div type="textpart" subtype="section" n="1"><p>Prima <note>app. crit.</note>pars
   <choice><orig>vrbis</orig><reg>urbis</reg></choice>.</p></div>
  <div type="textpart" subtype="section" n="2"><p>Secunda<lb/>pars.</p><p>Alter.</p></div>
 </div>
</div>
<div type="textpart" subtype="book" n="2">
 <div type="textpart" subtype="chapter" n="15e">
  <div type="textpart" subtype="section" n="1"><p>Externum <del>x</del>exemplum.</p></div>
 </div>
</div>""")


def test_prose_levels_and_text():
    r = fc.parse_tei(PROSE)
    assert r["levels"] == ["div", "div", "div"] and not r["is_line"]
    assert r["subtypes"] == {0: "book", 1: "chapter", 2: "section"}
    got = dict(r["passages"])
    assert got[("1", "1", "1")] == "Prima pars urbis."
    assert got[("1", "1", "2")] == "Secunda pars.\n\nAlter."
    assert got[("2", "15e", "1")] == "Externum exemplum."
    assert r["title"] == "Fixture Work"


VERSE = tei(
    pat("line", "/tei:TEI/tei:text/tei:body/tei:div/tei:div[@n='$1']//tei:l[@n='$2']"),
    """<div type="textpart" subtype="book" n="4">
 <div type="textpart" subtype="card" n="1">
  <l n="27">Sed mihi vel tellus</l><l>optem prius ima dehiscat</l>
  <sp><speaker>A.</speaker><l n="29" part="I">ille meos,</l></sp>
  <sp><speaker>B.</speaker><l part="F">primus qui</l></sp>
  <l>abstulit;</l>
 </div>
</div>""")


def test_verse_lines_numbering_and_parts():
    r = fc.parse_tei(VERSE)
    assert r["levels"] == ["div"] and r["is_line"]
    got = dict(r["passages"])
    assert got[("4", "27")] == "Sed mihi vel tellus"
    assert got[("4", "28")] == "optem prius ima dehiscat"
    assert got[("4", "29")] == "ille meos, primus qui"  # split line merged, speakers dropped
    assert got[("4", "30")] == "abstulit;"
    assert fc.default_unit_levels(r) == 1  # book of a poem


MILESTONES = tei(
    pat("chapter", "/tei:TEI/tei:text/tei:body/tei:div/tei:div/tei:div[@n='$1']"
                   "/tei:div[@n='$2']"),
    """<div type="textpart" subtype="part" n="1">
 <div type="textpart" subtype="book" n="7">
  <div type="textpart" subtype="chapter" n="12"><p><milestone unit="section" n="52"/>Alpha.
   <milestone unit="section" n="53"/>Beta.</p></div>
  <div type="textpart" subtype="chapter" n="13"><p>Gamma.<milestone unit="section" n="54"/>
   Delta.</p></div>
 </div>
</div>""")


def test_milestone_sections_and_skipped_part_level():
    r = fc.parse_tei(MILESTONES)
    assert r["ms_unit"] == "section"
    assert r["subtypes"] == {0: "book", 1: "chapter"}  # the decade/part level is skipped
    got = dict(r["passages"])
    assert got[("7", "12", "52")] == "Alpha."
    assert got[("7", "12", "53")] == "Beta."
    assert got[("7", "13", "53")] == "Gamma."  # text before the chapter's first milestone
    assert got[("7", "13", "54")] == "Delta."


WORK = {"work_id": "fixture-work", "author": "Auctor", "work": "Opus", "abbr": "Auct."}


def test_group_units_drop_and_id_sub():
    rows = [(("7", "12", "52"), "a"), (("7", "13", "54"), "b"), (("8", "1", "1"), "c")]
    units = fc.group_units(WORK, rows, 1, drop=1, drop_name="chapter")
    assert [u["unit"] for u in units] == ["7", "8"]
    p = units[0]["passages"][1]
    assert p["id"] == "7.54" and p["label"] == "Auct. 7.54 (chapter 13)"
    units = fc.group_units(WORK, [(("9", "15e", "1"), "x")], 1,
                           id_sub=[(r"^(\d+)\.(\d+)e\.", r"\1.\2.ext.")])
    assert units[0]["passages"][0]["id"] == "9.15.ext.1"
    whole = fc.group_units(WORK, [(("22",), "t")], 0)
    assert whole[0]["unit"] == "all" and whole[0]["passages"][0]["label"] == "Auct. 22"
    multi = fc.group_units(WORK, [(("15", "1"), "t")], 0, prefix=("alexander",),
                           part_label="Alexander")
    assert multi[0]["unit"] == "alexander"
    assert multi[0]["passages"][0]["id"] == "alexander.15.1"
    assert multi[0]["passages"][0]["label"] == "Auct. Alexander 15.1"


def test_split_unit_under_limit():
    u = {"corpus": "c", "unit": "3", "title": "T",
         "passages": [{"id": f"3.{i}", "label": "L", "text": "x" * 1000} for i in range(50)]}
    parts = fc.split_unit(u, limit=12_000)
    assert len(parts) > 1 and [p["unit"] for p in parts[:2]] == ["3a", "3b"]
    assert all(fc.unit_bytes(p) <= 12_000 for p in parts)
    assert sum(len(p["passages"]) for p in parts) == 50
    assert parts[0]["passage_range"][0] == "3.0" and parts[0]["parent_unit"] == "3"


def page(body: str) -> str:
    return (f"<html><head><title>x</title></head><body><p class=pagehead>HEAD</p>"
            f"<p class=border></p>{body}<p class=border></p><p><a href=\"index.html\">The "
            f"Latin Library</a></p></body></html>")


def test_ll_bracket_chapters():
    src = page("<p>Praefatio quae ante capitula legitur et satis longa est ut servetur.</p>"
               "<p>[I] Omnium certa &aelig; sententia.</p><p>Pergit &lt;stulta&gt; liber.</p>"
               "<p>[II] Sed non est.</p>")
    got = dict(fc.parse_ll_prose(src))
    assert got["1"] == "Omnium certa æ sententia.\n\nPergit ⟨stulta⟩ liber."
    assert got["2"] == "Sed non est."
    assert got["pr"].startswith("Praefatio")


def test_ll_heading_with_bracket_sections():
    src = page("<p><b>LIBER PRIMUS</B></p><p><b>I. ANASTASIO REX.</B></p>"
               "<p>[1] Oportet nos.</p><p>[2] Omni quippe.</p>"
               "<p><b>II.</b> Comitis Stephani.</p>")
    got = dict(fc.parse_ll_prose(src))
    assert got["1.1"] == "ANASTASIO REX.\n\nOportet nos."
    assert got["1.2"] == "Omni quippe."
    assert got["2"] == "Comitis Stephani."


def test_ll_roman_dot_with_font_sections():
    src = page("<p>I. <font size=2>1</FONT> Inquirenti mihi.</p>"
               "<p><font size=2>2</FONT> Illum tamen. <font size=2>3</FONT> Non est.</p>"
               "<p>II. Sed cum. <FONT size=2>2</FONT> Cum quo.</p>")
    got = dict(fc.parse_ll_prose(src))
    assert got == {"1.1": "Inquirenti mihi.", "1.2": "Illum tamen.", "1.3": "Non est.",
                   "2.1": "Sed cum.", "2.2": "Cum quo."}


def test_ll_verse_numbering():
    lines = "<br>\n".join(f"versus {i}" + ("&nbsp;&nbsp;<font size=2>%d</FONT>" % i
                                             if i % 5 == 0 else "") for i in range(1, 12))
    got = fc.parse_ll_verse(page(f"<p>{lines}</p>"))
    assert got[0] == ("1", "versus 1") and got[4] == ("5", "versus 5")
    assert got[-1] == ("11", "versus 11") and len(got) == 11


def test_note_tally(tmp_path):
    (tmp_path / "annot-001.md").write_text(
        "# x\n\nbody - {a} (p001): **Pliny, *Natural History* VII.10** — not in notes\n\n"
        "## Notes\n\n"
        "- {a} (p009): **Cicero, *De amicitia* (*Laelius*) 22** — Ciceron.\n"
        "- {b} (p009): **Pliny, *Natural History* VII.10; Solinus as above** — Pline.\n"
        "- {c} (p010): **Gellius (*Aule Gelle*), *Attic Nights* IX.4 (citing Pliny's "
        "examples)** — Gelle.\n"
        "- {d} (p010): **Juan Luis Vives, commentary on Augustine, *De civitate Dei*** — Viues.\n"
        "- Verse (p092): **Virgil, *Eclogues* 8.69** — Carmina.\n", encoding="utf-8")
    ids = fc.note_identifications(tmp_path)
    assert [i[1] for i in ids] == ["p009", "p009", "p010", "p010"]
    c = fc.tally(ids)
    assert c["cicero-de-amicitia"] == 1
    assert c["pliny-naturalis-historia"] == 1  # "Pliny's" inside Gellius does not count
    assert c["solinus-collectanea"] == 1 and c["gellius-noctes-atticae"] == 1
    assert c["augustine-de-civitate-dei"] == 0  # excluded: Vives's commentary
    assert c["vives-commentary-de-civitate-dei"] == 1
    assert c["virgil-eclogues"] == 0  # Verse lines are quotations, not Notes citations


def test_select_works_keeps_sourceless_and_stops_at_coverage():
    cat = [dict(work_id=w, src=s) for w, s in
           [("a", {"type": "perseus"}), ("b", None), ("c", {"type": "perseus"}),
            ("d", {"type": "perseus"})]]
    counts = {"a": 6, "b": 2, "c": 1, "d": 1}
    sel, skip = fc.select_works(counts, cat, 0.6, False)
    assert [w["work_id"] for w in sel] == ["a", "b"]
    assert [w["work_id"] for w in skip] == ["c", "d"]


def test_roman():
    assert fc.roman_to_int("XIV") == 14 and fc.roman_to_int("ix") == 9
    assert fc.roman_to_int("ABC") is None
