#!/usr/bin/env python3
"""Auto-defer every undecided arbitration item to the translator; same as `wave.py defer`.

    uv run python scripts/auto_defer.py [--dry-run] [PAGE ...]

Line items (body / note / unmatched / flagged) get "either", structure items (structural,
note-structure) get "unknown", each with "by": "auto" and "reason": "auto-deferred to
translator". Existing decisions are never overwritten. See wave.py for details.
"""
import importlib.util
import pathlib
import sys

# loaded by path: a plain `import wave` would find the standard library's wave module
_spec = importlib.util.spec_from_file_location(
    "coras_wave", pathlib.Path(__file__).resolve().parent / "wave.py")
wave = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wave)


def main(argv=None):
    wave.main(["defer", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    main()
