"""Scaffold a new book root for the transcription + translation pipeline.

usage: new_book.py DIR --title T --author A --year Y [--short-title S] [--edition E]
                   [--language French] [--target-language English] [--slug S]

Creates DIR with:
  book.json             the book's facts and section-heading words (see bookconf.py);
                        `site` is a stub for split_pages.py, fill its TODOs before the site
  manifest.json         one record per image found in DIR/raw/ (sorted by name), ids
                        p001, p002, …, every stage pending; page/folio/side are null and
                        must be filled (and front matter renamed p000-<name>) before reads
  scripts -> toolkit    a symlink to this scripts/ directory, so the runbook commands
                        (`uv run python scripts/wave.py status`) work from DIR unchanged
  prompts/              copies of read_single.md, translate.md and review.md to adapt
  docs/conventions.md   from templates/conventions.template.md
  docs/case-file.md     from templates/case-file.template.md
  docs/pipeline-log.md, docs/handoff.md   stubs
  hyphen_keep.txt, .gitignore, and the empty stage directories.

Refuses to touch a DIR that already has a book.json. Put the scans in DIR/raw/ first
(or re-run on a fresh DIR) to get a populated manifest.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import sys

SCRIPTS = pathlib.Path(__file__).resolve().parent
TEMPLATES = SCRIPTS / "templates"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".jp2"}
STAGES = ["acquired", "cropped", "readA", "readB", "diffed", "final", "spotchecked",
          "translated", "reviewed"]
PROMPTS = ("read_single.md", "translate.md", "review.md")
DIRS = ("raw", "pages", "transcription/reads/A", "transcription/reads/B",
        "transcription/final", "transcription/arbitration/queue",
        "transcription/arbitration/decisions", "transcription/reports", "text",
        "translation/sections", "translation/reports", "translation/alt-choices",
        "translation/review", "docs/checks", "docs/reference", "prompts", "site/data")
GITIGNORE = """# images never enter git (the repo-root .gitignore also ignores image extensions)
raw/
pages/full/
pages/read/
pages/strips/
transcription/arbitration/crops/
site/img/
"""


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "book"


def book_json(a) -> dict:
    return {
        "title": a.title,
        "short_title": a.short_title or a.title,
        "author": a.author,
        "year": a.year,
        "edition": a.edition or "",
        "language": a.language,
        "target_language": a.target_language,
        "description": "TODO: one paragraph: what the book is, its structure, the copy-text.",
        "front_matter": {"p000-title": "title"},
        "headings": {"texte": ["TEXTE"],
                     "annotation": ["ANNOT", "ANNOTAT", "ANNOTATION", "ANNOTATIONS"]},
        "hyphen_keep": "hyphen_keep.txt",
        "site": {
            "slug": a.slug or slugify(a.short_title or a.title),
            "title": a.title,
            "short_title": a.short_title or a.title,
            "author": a.author,
            "year": a.year,
            "description": "TODO",
            "about": ["TODO: how the text was made (two independent transcriptions, how "
                      "disagreements were resolved), as in the Coras record in split_pages.py."],
            "source": {"name": "TODO: holding library", "item_url": "TODO",
                       "license": "TODO", "license_url": "TODO"},
            "images": {"base": "img/", "ext": ".webp", "width": 2805},
            "layers": [{"code": "en", "label": a.target_language},
                       {"code": "fr", "label": a.language}],
            "default_layer": "en",
            "stylesheet": None,
            "first_page": "p000-title",
        },
    }


def manifest(raw_dir: pathlib.Path, edition: str) -> dict:
    images = sorted(p for p in raw_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXT) if raw_dir.is_dir() else []
    pages = []
    for n, img in enumerate(images, 1):
        pages.append({"id": f"p{n:03d}", "page": None, "image": n, "side": None,
                      "folio": None, "source": "other", "url": None, "raw": img.name,
                      "status": {s: "pending" for s in STAGES}})
    return {"item": "", "edition": edition, "native_size": None, "pages": pages}


def fill(text: str, a) -> str:
    for key, value in (("TITLE", a.title), ("AUTHOR", a.author),
                       ("EDITION", a.edition or str(a.year)), ("YEAR", str(a.year)),
                       ("LANGUAGE", a.language), ("TARGET_LANGUAGE", a.target_language)):
        text = text.replace("{{" + key + "}}", value)
    return text


def scaffold(root, a) -> list[str]:
    root = pathlib.Path(root).resolve()
    if (root / "book.json").exists():
        raise SystemExit(f"{root}/book.json exists; refusing to overwrite a book")
    made = []
    for d in DIRS:
        (root / d).mkdir(parents=True, exist_ok=True)

    def write(rel, text):
        (root / rel).write_text(text, encoding="utf-8")
        made.append(rel)

    write("book.json", json.dumps(book_json(a), indent=1, ensure_ascii=False) + "\n")
    write("manifest.json", json.dumps(manifest(root / "raw", a.edition or ""), indent=1,
                                      ensure_ascii=False) + "\n")
    write("docs/conventions.md", fill((TEMPLATES / "conventions.template.md").read_text(), a))
    write("docs/case-file.md", fill((TEMPLATES / "case-file.template.md").read_text(), a))
    write("docs/pipeline-log.md", f"# Pipeline log: {a.title} ({a.year})\n\nRunning record "
          "of what ran, when, with what result. Dated sections, newest at the bottom.\n")
    write("docs/handoff.md", f"# Handoff: {a.author}, *{a.title}* ({a.year})\n\nRead this "
          "first. Keep it current at every checkpoint.\n\n## Where things stand\n\n"
          "| stage | state |\n|---|---|\n| scaffold | done |\n\n## Open items\n\n- \n")
    write("hyphen_keep.txt", "# compounds whose hyphen survives a line break, one per line\n")
    write(".gitignore", GITIGNORE)
    for name in PROMPTS:
        shutil.copyfile(SCRIPTS / "prompts" / name, root / "prompts" / name)
        made.append(f"prompts/{name}")
    link = root / "scripts"
    if not link.exists():
        link.symlink_to(os.path.relpath(SCRIPTS, root), target_is_directory=True)
        made.append("scripts -> " + os.readlink(link))
    return made


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir")
    ap.add_argument("--title", required=True)
    ap.add_argument("--author", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--short-title")
    ap.add_argument("--edition", help='e.g. "Paris: Galliot du Pré, 1572"')
    ap.add_argument("--language", default="French")
    ap.add_argument("--target-language", default="English")
    ap.add_argument("--slug")
    a = ap.parse_args(argv)
    made = scaffold(a.dir, a)
    n = len(json.loads((pathlib.Path(a.dir) / "manifest.json").read_text())["pages"])
    print(f"scaffolded {pathlib.Path(a.dir).resolve()}: {len(made)} files, "
          f"{n} pages in the manifest")
    for m in made:
        print(f"  {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
