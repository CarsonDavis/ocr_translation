"""Which book the scripts work on, and the few facts about it that the code needs.

The scripts in this directory are the toolkit; a *book root* holds one book's data
(`manifest.json`, `pages/`, `transcription/`, `text/`, `translation/`, `docs/`,
`site/`) and a `book.json`. The book root is, in order:

1. `$BOOK_ROOT`, when set;
2. the current directory or the nearest parent that holds a `book.json`;
3. the directory above this one (`2026/`, the Coras book), as before.

A new book made by `new_book.py` has a `scripts` symlink to this directory, so every
runbook command (`uv run python scripts/wave.py status`, run from the book root) works
unchanged there.

`book.json` keys (all optional; missing keys take the DEFAULTS below):

  title, short_title, author, year, edition, language, target_language, description
      identity; `short_title` (else `title`) heads the review render.
  front_matter   {page id: section id} for pages that are a section by themselves.
  headings       {"texte": [words], "annotation": [words]}: the printed heading words
                 the stitch cuts sections at (one-letter misprints are tolerated).
  hyphen_keep    path (from the book root) of the hyphen keep-list for the stitch.
  site           the viewer's book record (split_pages.py writes it to site/data/book.json);
                 without it split_pages falls back to the Coras record.

A book may override any agent prompt by putting `prompts/<kind>.md` in its root;
otherwise the toolkit's `scripts/prompts/<kind>.md` is used. `{BOOK_ROOT}` in a prompt
is replaced with the book root when it is rendered.
"""
from __future__ import annotations

import copy
import json
import os
import pathlib
import tempfile

SCRIPTS = pathlib.Path(__file__).resolve().parent
TOOLKIT_PROMPTS = SCRIPTS / "prompts"

DEFAULTS = {
    "title": "",
    "short_title": "",
    "author": "",
    "year": None,
    "edition": "",
    "language": "French",
    "target_language": "English",
    "description": "",
    "front_matter": {"p000-title": "title", "p000-argument": "argument"},
    "headings": {"texte": ["TEXTE"],
                 "annotation": ["ANNOT", "ANNOTAT", "ANNOTATION", "ANNOTATIONS"]},
    "hyphen_keep": "hyphen_keep.txt",
    "site": None,
}


def find_root(cwd=None, env=None) -> pathlib.Path:
    """The book root (see the module docstring)."""
    env = os.environ if env is None else env
    if env.get("BOOK_ROOT"):
        return pathlib.Path(env["BOOK_ROOT"]).expanduser().resolve()
    here = pathlib.Path(cwd or os.getcwd()).resolve()
    for d in (here, *here.parents):
        if (d / "book.json").is_file():
            return d
    return SCRIPTS.parent


ROOT = find_root()


def load(root=None) -> dict:
    """DEFAULTS overlaid with `<root>/book.json` (top-level keys replace whole)."""
    cfg = copy.deepcopy(DEFAULTS)
    path = pathlib.Path(root or ROOT) / "book.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path}: top level must be a JSON object")
        cfg.update(data)
    return cfg


def prompt_path(kind: str, root=None) -> pathlib.Path:
    """`<root>/prompts/<kind>.md` if the book has its own, else the toolkit's."""
    own = pathlib.Path(root or ROOT) / "prompts" / f"{kind}.md"
    return own if own.is_file() else TOOLKIT_PROMPTS / f"{kind}.md"


def fill_root(text: str, root=None) -> str:
    return text.replace("{BOOK_ROOT}", str(pathlib.Path(root or ROOT)))


def scratch_dir() -> pathlib.Path:
    """Where rendered prompts go: $BOOK_SCRATCH, $CORAS_SCRATCH, else a temp dir."""
    env = os.environ.get("BOOK_SCRATCH") or os.environ.get("CORAS_SCRATCH")
    if env:
        return pathlib.Path(env)
    return pathlib.Path(tempfile.gettempdir()) / f"{ROOT.name}-scratch" / "waves"
