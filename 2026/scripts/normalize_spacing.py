"""Apply the punctuation-spacing rule of conventions §1 to every printed line of a page file.

usage: normalize_spacing.py PATH [PATH ...]   (rewrites in place; prints changed line counts)
The rule is deterministic, so applying it to a read never changes what was read, only how
it is spaced. Also importable: normalize_line(text) -> text.
"""
import json, pathlib, re, sys

_BEFORE = re.compile(r"\s+([,.:;?!])")
_AFTER = re.compile(r"([,.:;?!])(?=[^\s,.:;?!\]\)}'\"])")
_OPEN = re.compile(r"(?<=[^\s(\[{])\(")          # d'eux(qu'elle -> d'eux (qu'elle
_CLOSE = re.compile(r"\)(?=[^\s,.:;?!)\]}])")     # ſignoit)luy -> ſignoit) luy


_SPACED_RUN = re.compile(r"(?<![^\s])((?:(?:[A-Z0-9ÆŒÃẼÕÑ]|AE|OE) ){2,}(?:[A-Z0-9ÆŒÃẼÕÑ]|AE|OE)[.,:;]?)(?![^\s])")


def collapse_spaced_caps(text: str) -> str:
    """`A R R E S T  D V` -> `ARREST DV`; `L X X I I.` -> `LXXII.`; `1 5 6 0` -> `1560`.
    A run is THREE or more single capitals/digits separated by exactly one space; the last
    may carry punctuation. Two-letter runs (`A M.` = preposition + abbreviation, `D V`) are
    left alone, so readers must write two-letter spaced words closed up themselves.
    Tokens with their own punctuation (`M. D.`) are never joined."""
    return _SPACED_RUN.sub(lambda m: m.group(1).replace(" ", ""), text)


def normalize_line(text: str, spaced_caps: bool = False) -> str:
    text = collapse_spaced_caps(text)
    text = _BEFORE.sub(r"\1", text)
    text = _AFTER.sub(r"\1 ", text)
    text = re.sub(r"(?<=[^\s])&", " &", text)      # zieme,&mieux -> zieme, & mieux
    text = re.sub(r"&(?=[^\s])", "& ", text)
    text = re.sub(r"(?<=[^\s])\{", " {", text)     # Dieu{c} -> Dieu {c}
    text = re.sub(r"\}\s+(?=[,.:;?!])", "}", text)  # {c} . -> {c}.
    text = re.sub(r"\s+\)", ")", text)               # aucun ) -> aucun)
    text = re.sub(r"\(\s+", "(", text)               # ( ſil -> (ſil
    text = _OPEN.sub(" (", text)
    text = _CLOSE.sub(") ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def normalize_page(page: dict) -> int:
    changed = 0
    for block in page.get("blocks") or []:
        if not isinstance(block, dict):
            continue
        spaced = bool(block.get("spaced_caps"))
        if block.get("type") == "paragraph":
            new = [normalize_line(t, spaced) if isinstance(t, str) else t for t in block.get("lines") or []]
            changed += sum(1 for a, b in zip(block["lines"], new) if a != b)
            block["lines"] = new
        elif block.get("type") == "heading" and isinstance(block.get("text"), str):
            new = normalize_line(block["text"], spaced)
            changed += new != block["text"]; block["text"] = new
    if isinstance(page.get("running_head"), str):
        new = normalize_line(page["running_head"])
        changed += new != page["running_head"]; page["running_head"] = new
    for key in ("margin_notes", "foot_notes"):
        for note in page.get(key) or []:
            if isinstance(note, dict):
                new = [normalize_line(t) if isinstance(t, str) else t for t in note.get("lines") or []]
                changed += sum(1 for a, b in zip(note["lines"], new) if a != b)
                note["lines"] = new
    return changed


def main():
    for arg in sys.argv[1:]:
        path = pathlib.Path(arg)
        page = json.loads(path.read_text())
        n = normalize_page(page)
        path.write_text(json.dumps(page, indent=1, ensure_ascii=False) + "\n")
        print(f"{path}: {n} lines changed")


if __name__ == "__main__":
    main()
