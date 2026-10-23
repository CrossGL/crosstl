"""Unpack a same-run host package that was downloaded without decompression.

``actions/download-artifact`` extracts archives while it streams them and aborts
when the blob stream stalls for 30 seconds. On Windows runners the extraction of
the large host package regularly stalls that long, so the host jobs download the
archive as one file and unpack it here instead.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


class ArtifactArchiveError(RuntimeError):
    """The downloaded directory does not hold exactly one safe archive."""


def _single_archive(directory: Path) -> Path:
    entries = sorted(directory.iterdir()) if directory.is_dir() else []
    if len(entries) != 1 or not entries[0].is_file():
        names = ", ".join(entry.name for entry in entries) or "nothing"
        raise ArtifactArchiveError(
            f"{directory} must hold exactly one downloaded archive, found {names}."
        )
    archive = entries[0]
    if not zipfile.is_zipfile(archive):
        raise ArtifactArchiveError(f"{archive} is not a zip archive.")
    return archive


def unpack(directory: Path) -> list[str]:
    """Extract the only archive in ``directory`` into it and remove the archive.

    Returns the extracted member names in archive order. Members that would
    land outside ``directory`` and archives that contain the archive's own name
    are refused before anything is written.
    """
    directory = Path(directory)
    archive = _single_archive(directory)
    root = directory.resolve()
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if not members:
            raise ArtifactArchiveError(f"{archive} is empty.")
        for member in members:
            target = (root / member.filename).resolve()
            if target == root or not target.is_relative_to(root):
                raise ArtifactArchiveError(
                    f"{archive} member {member.filename!r} escapes {directory}."
                )
            if target == archive.resolve():
                raise ArtifactArchiveError(
                    f"{archive} member {member.filename!r} overwrites the archive."
                )
        bundle.extractall(root)
    archive.unlink()
    return [member.filename for member in members]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    names = unpack(args.directory)
    files = sum(1 for name in names if not name.endswith("/"))
    print(f"Unpacked {files} files into {args.directory}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
