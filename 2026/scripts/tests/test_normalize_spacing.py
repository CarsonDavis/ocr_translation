import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from normalize_spacing import normalize_line

def test_space_before_punct_removed():
    assert normalize_line("nu , de tant , &") == "nu, de tant, &"

def test_space_after_punct_added():
    assert normalize_line("aage,on void") == "aage, on void"
    assert normalize_line("L.i.parag.") == "L. i. parag."
    assert normalize_line("xxxiij.q.j.") == "xxxiij. q. j."

def test_untouched_cases():
    assert normalize_line("d'Artigat, au") == "d'Artigat, au"
    assert normalize_line("vertueux {z}. Et") == "vertueux {z}. Et"
    assert normalize_line("[...] ſuite") == "[...] ſuite"
    assert normalize_line("fin.") == "fin."
    assert normalize_line("nevou-") == "nevou-"

def test_spaced_caps_closed_up():
    assert normalize_line("A R R E S T  DV", spaced_caps=True) == "ARREST DV"
    assert normalize_line("A R R E S T  D V") == "ARREST D V"   # two-letter runs are the reader's job
    assert normalize_line("P A R L E M E N T  DE  T H O L O S E.") == "PARLEMENT DE THOLOSE."
    assert normalize_line("DV  P A R L E M E N T") == "DV PARLEMENT"
    assert normalize_line("M.  D.  L X X I I.") == "M. D. LXXII."
    assert normalize_line("Septembre. 1 5 6 0.") == "Septembre. 1560."
    assert normalize_line("qu'ils appeloyẽt P R AE") == "qu'ils appeloyẽt PRAE"
    assert normalize_line("A  P A R I S,") == "A PARIS,"
    assert normalize_line("Antoine, A M. Antoine") == "Antoine, A M. Antoine"   # two-letter runs untouched
    assert normalize_line("le xij. C. V. chap.") == "le xij. C. V. chap."


def test_parentheses():
    assert normalize_line("d'eux(qu'elle ſignoit)luy") == "d'eux (qu'elle ſignoit) luy"
    assert normalize_line("ici) ſans (foy)") == "ici) ſans (foy)"
    assert normalize_line("faire {a}(voire)") == "faire {a} (voire)"


def test_ampersand_spaced():
    assert normalize_line("zieme,&mieux") == "zieme, & mieux"
    assert normalize_line("belles & doctes") == "belles & doctes"


def test_marker_spacing():
    assert normalize_line("à Dieu{c}. collo-") == "à Dieu {c}. collo-"
    assert normalize_line("attaire{b} , voire") == "attaire {b}, voire"
    assert normalize_line("{a} Au debut") == "{a} Au debut"


def test_no_space_inside_parentheses():
    assert normalize_line("ſuperieur ) peuuent") == "ſuperieur) peuuent"
    assert normalize_line("( ſil en y a aucun ) de") == "(ſil en y a aucun) de"
