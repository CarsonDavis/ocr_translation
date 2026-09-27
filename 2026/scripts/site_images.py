#!/usr/bin/env python3
"""Convert the full-resolution page crops to the WebP images the viewer serves.

    uv run python scripts/site_images.py [--src pages/full] [--out site/img]
                                         [--quality 70] [--only p004,p005]
                                         [--upload --bucket NAME --profile PROFILE]
                                         [--slug martin-guerre]

Reads `<src>/<id>.jpg` (default `pages/full`, the deskewed full-resolution crops)
and writes `<out>/<id>.webp` (default `site/img`, which is gitignored) with
`cwebp -q <quality> -m 6`.  A page whose `.webp` is newer than its `.jpg` is left
alone, so a rerun after one recrop costs one conversion, not 162.

Nothing here writes to `manifest.json`, `transcription/`, `pages/` or `raw/`.
Without `--upload` the run only prints the `aws s3 sync` command for Carson to
run; with `--upload` it runs it.  See site/README.md.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import pathlib
import shlex
import subprocess
import sys

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPTS_DIR.parent

DEFAULT_SRC = DEFAULT_ROOT / "pages" / "full"
DEFAULT_OUT = DEFAULT_ROOT / "site" / "img"
DEFAULT_QUALITY = 70
DEFAULT_SLUG = "martin-guerre"

CWEBP = "cwebp"
# cwebp is single-threaded, so the pool is what keeps all the cores busy; -m 6 is
# the slowest/smallest of its compression methods and costs a few seconds a page.
CWEBP_METHOD = "6"

CACHE_CONTROL = "public, max-age=31536000, immutable"


# --- conversion -----------------------------------------------------------

def convert_one(src: pathlib.Path, dst: pathlib.Path, quality: int) -> bool:
    """Convert one JPEG to WebP.  True if cwebp ran, False if `dst` was fresh."""
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [CWEBP, "-q", str(quality), "-m", CWEBP_METHOD, "-quiet",
         str(src), "-o", str(dst)],
        check=True,
    )
    return True


def convert_all(
    src_dir: pathlib.Path,
    out_dir: pathlib.Path,
    quality: int = DEFAULT_QUALITY,
    only: list[str] | None = None,
) -> tuple[int, int, int]:
    """Convert every `<src_dir>/<id>.jpg`.  -> (n_images, n_converted, total_bytes).

    `total_bytes` is the size of all the output images the run is responsible for,
    the skipped ones included, so the summary reads the same on a rerun.
    """
    src_dir = pathlib.Path(src_dir)
    out_dir = pathlib.Path(out_dir)
    wanted = set(only) if only else None

    jobs = []
    for src in sorted(src_dir.glob("*.jpg")):
        if wanted is not None and src.stem not in wanted:
            continue
        jobs.append((src, out_dir / f"{src.stem}.webp"))

    if not jobs:
        return 0, 0, 0

    out_dir.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        converted = list(pool.map(
            lambda job: convert_one(job[0], job[1], quality), jobs
        ))

    total = sum(dst.stat().st_size for _, dst in jobs if dst.exists())
    return len(jobs), sum(converted), total


# --- upload ---------------------------------------------------------------

def upload_command(
    bucket: str, profile: str, slug: str, img_dir: pathlib.Path
) -> list[str]:
    """The `aws s3 sync` argv that puts `img_dir` under `<slug>/img/` in `bucket`.

    The images are content-addressed by page id and never change once a page is
    final, so they go up immutable with a one-year max-age; `--size-only` keeps a
    rerun from re-uploading 150 MB because the mtimes moved.
    """
    return [
        "aws", "s3", "sync",
        f"{pathlib.Path(img_dir)}/",
        f"s3://{bucket}/{slug}/img/",
        "--profile", profile,
        "--cache-control", CACHE_CONTROL,
        "--content-type", "image/webp",
        "--size-only",
    ]


# --- cli ------------------------------------------------------------------

def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert pages/full/*.jpg to site/img/*.webp for the viewer."
    )
    p.add_argument("--src", type=pathlib.Path, default=DEFAULT_SRC,
                   help="directory of full-resolution JPEG crops")
    p.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT,
                   help="directory to write the WebP images into")
    p.add_argument("--quality", type=int, default=DEFAULT_QUALITY,
                   help="cwebp -q value (default 70)")
    p.add_argument("--only", default=None,
                   help="comma-separated page ids to convert, e.g. p004,p005")
    p.add_argument("--upload", action="store_true",
                   help="run the aws s3 sync instead of only printing it")
    p.add_argument("--bucket", default=None, help="destination S3 bucket")
    p.add_argument("--profile", default=None, help="AWS profile to sync with")
    p.add_argument("--slug", default=DEFAULT_SLUG,
                   help="path prefix in the bucket (default martin-guerre)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None, runner=subprocess.run) -> int:
    args = parse_args(argv)

    if args.upload and not (args.bucket and args.profile):
        print("--upload needs both --bucket and --profile", file=sys.stderr)
        raise SystemExit(2)

    only = [s for s in (args.only or "").split(",") if s] or None
    n_images, n_converted, total_bytes = convert_all(
        args.src, args.out, args.quality, only
    )
    mb = total_bytes / 1e6  # MB as Finder and the AWS console count it
    print(f"{n_images} images, {n_converted} converted, {mb:.1f} MB")

    argv_aws = upload_command(
        args.bucket or "<bucket>", args.profile or "<profile>", args.slug, args.out
    )
    if args.upload:
        runner(argv_aws, check=True)
    else:
        print()
        print("To upload:")
        print(f"  {shlex.join(argv_aws)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
