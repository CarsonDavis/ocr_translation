# scripts/tests/test_site_images.py
"""Tests for scripts/site_images.py.

The images are 20x30 JPEGs made with Pillow, so cwebp does real work on real
files without the suite needing a 500 MB scan directory.  The upload path is
never run for real: `main` takes a `runner` and the tests pass a fake.

    uv run --with pytest,pillow pytest scripts/tests/test_site_images.py -q
"""
import pathlib
import sys

import pytest
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import site_images  # noqa: E402


def make_jpeg(path: pathlib.Path, size=(20, 30)) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (250, 248, 240)).save(path, "JPEG")
    return path


# --- conversion -----------------------------------------------------------

def test_converts_and_skips_fresh(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    make_jpeg(src / "p004.jpg")

    n_images, n_converted, total = site_images.convert_all(src, out)
    assert (out / "p004.webp").exists()
    assert (n_images, n_converted) == (1, 1)
    assert total > 0

    again = site_images.convert_all(src, out)
    assert again == (1, 0, total)


def test_only_filter(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    make_jpeg(src / "a.jpg")
    make_jpeg(src / "b.jpg")

    n_images, n_converted, _ = site_images.convert_all(src, out, only=["a"])
    assert (n_images, n_converted) == (1, 1)
    assert (out / "a.webp").exists()
    assert not (out / "b.webp").exists()


# --- upload ---------------------------------------------------------------

def test_upload_command_text():
    assert site_images.upload_command(
        "b", "prof", "martin-guerre", pathlib.Path("/x/img")
    ) == [
        "aws", "s3", "sync",
        "/x/img/",
        "s3://b/martin-guerre/img/",
        "--profile", "prof",
        "--cache-control", "public, max-age=31536000, immutable",
        "--content-type", "image/webp",
        "--size-only",
    ]


def test_upload_requires_args(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    calls = []

    with pytest.raises(SystemExit) as exc:
        site_images.main(
            ["--src", str(src), "--out", str(out), "--upload"],
            runner=lambda *a, **k: calls.append(a),
        )
    assert exc.value.code == 2
    assert calls == []
    assert not out.exists()


def test_upload_runs_command(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    make_jpeg(src / "p004.jpg")
    calls = []

    site_images.main(
        ["--src", str(src), "--out", str(out),
         "--upload", "--bucket", "b", "--profile", "p"],
        runner=lambda argv, **kwargs: calls.append((argv, kwargs)),
    )
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv == site_images.upload_command("b", "p", "martin-guerre", out)
    assert kwargs == {"check": True}


# --- output ---------------------------------------------------------------

def test_summary_output(tmp_path, capsys):
    src, out = tmp_path / "src", tmp_path / "out"
    make_jpeg(src / "p004.jpg")

    site_images.main(["--src", str(src), "--out", str(out)])
    printed = capsys.readouterr().out

    assert "1 images, 1 converted, 0.0 MB" in printed
    assert "To upload:" in printed
    # shlex.join quotes the angle brackets, so the line stays pasteable.
    assert "s3://<bucket>/martin-guerre/img/" in printed
    assert "--profile '<profile>'" in printed
