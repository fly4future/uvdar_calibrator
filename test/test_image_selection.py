"""Assert-based checks for calibration-photo discovery.

Run directly: python3 test/test_image_selection.py
No test framework on purpose -- these guard the folder-picking logic, where a
wrong answer looks like a plausible one: pointing the calibrator at its own
`detected_marker_previews` output used to "work" (no crash, some samples
accepted) while quietly producing a calibration set of a handful of images
whose previews show every marker drawn twice.
"""

from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uvdar_calibrator.apps import cli  # noqa: E402
from uvdar_calibrator.engine import detection  # noqa: E402

_EXTS = ("bmp", "png", "jpg", "jpeg", "tif", "tiff")


def _make_folder(root: Path, name: str) -> Path:
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(1, 4):
        for ext in _EXTS:
            (folder / f"img_{i:02d}.{ext}").write_bytes(b"")
    return folder


def test_preview_folder_is_recognized():
    """A preview folder is output, never input -- and nested ones count too."""
    assert detection.is_preview_dir("example_images/detected_marker_previews")
    assert detection.is_preview_dir(
        "example_images/detected_marker_previews/detected_marker_previews"
    )
    assert detection.is_preview_dir(Path("photos") / detection.PREVIEW_DIR_NAME)
    assert not detection.is_preview_dir("example_images")
    assert not detection.is_preview_dir("photos")
    print("ok: preview folders are recognized, photo folders are not")


def test_preview_output_is_never_offered_as_input():
    """find_image_files must skip its own output, including when pointed at it."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        photos = _make_folder(root, "photos")
        previews = _make_folder(root, f"photos/{detection.PREVIEW_DIR_NAME}")
        # A stray file elsewhere in the tree must not change the photo count,
        # and a preview folder nested under a real one must stay invisible.
        _make_folder(root, f"photos/sub/{detection.PREVIEW_DIR_NAME}")

        found = detection.find_image_files(str(photos))
        assert len(found) == 3 * len(_EXTS), f"expected the 12 photos, got {len(found)}"
        assert all(detection.PREVIEW_DIR_NAME not in Path(f).parts for f in found)

        # Pointed straight at the output folder: nothing, not its own previews.
        assert detection.find_image_files(str(previews)) == []
    print("ok: preview output is never returned as a calibration photo")


def test_cli_refuses_a_preview_folder():
    """The CLI must stop with an explanation, not a mysteriously tiny db."""
    with tempfile.TemporaryDirectory() as tmp:
        previews = _make_folder(Path(tmp), detection.PREVIEW_DIR_NAME)
        assert cli.run(str(previews)) is None, (
            "run() must refuse a preview folder instead of calibrating from it"
        )
    print("ok: CLI refuses a preview folder")


if __name__ == "__main__":
    test_preview_folder_is_recognized()
    test_preview_output_is_never_offered_as_input()
    test_cli_refuses_a_preview_folder()
    print("all checks passed")
