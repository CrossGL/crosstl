"""Source positions for offsets in preprocessed text.

Some project diagnostics, such as those of Metal template materialization, find
their construct in text that the preprocessor wrote and that later passes
rewrote: directive lines are gone, included files are inlined and declarations
are respelled. A line and column counted in that text name no position in any
file the user wrote.

The preprocessor records the origin of every output line: the file and the
physical line it came from, and the origin of the ``#include`` that read the
file (``HLSLPreprocessor.line_origins``). ``record_preprocessed_text`` keeps the
origins of a unit's recent preprocessor runs. ``source_span`` maps an offset in
one of those outputs, or in text rewritten from one, back to a file the user
wrote: the unit itself, or a header under the project root. A line from a
header outside the project root is reported at the ``#include`` that reads it.

Rewritten text is aligned to the preprocessor output line by line. A line that
reached the output unchanged and was not rewritten keeps its column. Any other
line points at the first non-blank column of the physical line it came from.
"""

from __future__ import annotations

import difflib
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Sequence

FILE_LIMIT = 32
TEXTS_PER_FILE = 4
ALIGNMENT_CACHE_LIMIT = 16
SOURCE_FILE_CACHE_LIMIT = 64

# (file, physical line, include site), where the include site is the origin of
# the ``#include`` that read ``file`` and ``None`` in the preprocessed file.
Origin = tuple


@dataclass(frozen=True)
class SourceSpan:
    """A span in a file the user wrote. ``file`` is relative to the project root."""

    file: str
    line: int
    column: int
    offset: int
    end_line: int
    end_column: int
    end_offset: int


@dataclass(frozen=True, eq=False)
class _SourceFile:
    text: str
    line_starts: tuple[int, ...]

    @classmethod
    def of(cls, text: str) -> _SourceFile:
        starts = [0]
        index = text.find("\n")
        while index >= 0:
            starts.append(index + 1)
            index = text.find("\n", index + 1)
        return cls(text=text, line_starts=tuple(starts))

    def line(self, number: int) -> tuple[int, str] | None:
        """The offset and text of 1-based line ``number``."""
        if not 1 <= number <= len(self.line_starts):
            return None
        start = self.line_starts[number - 1]
        end = self.text.find("\n", start)
        return start, self.text[start : len(self.text) if end < 0 else end]


@dataclass(frozen=True, eq=False)
class _Run:
    """One preprocessor run over a unit: its input, output and line origins."""

    path: str
    source: _SourceFile
    text: str
    lines: tuple[str, ...]
    origins: tuple[Origin | None, ...]
    alignments: OrderedDict[str, tuple[tuple[int, bool], ...]] = field(
        default_factory=OrderedDict
    )


@dataclass(frozen=True)
class _Position:
    path: str
    line: int
    column: int
    offset: int


_lock = Lock()
_runs: OrderedDict[str, list[_Run]] = OrderedDict()
_source_files: OrderedDict[tuple[str, int, int], _SourceFile | None] = OrderedDict()


def _key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def record_preprocessed_text(
    file_path: str, source: str, text: str, origins: Sequence[Origin | None]
) -> None:
    """Remember that preprocessing ``source``, the text of ``file_path``, wrote ``text`` with line ``origins``."""
    lines = tuple(text.split("\n"))
    if not file_path or len(lines) != len(origins):
        return
    key = _key(file_path)
    with _lock:
        runs = [
            run
            for run in _runs.pop(key, [])
            if run.text != text or run.source.text != source
        ]
        runs.append(
            _Run(
                path=file_path,
                source=_SourceFile.of(source),
                text=text,
                lines=lines,
                origins=tuple(origins),
            )
        )
        _runs[key] = runs[-TEXTS_PER_FILE:]
        while len(_runs) > FILE_LIMIT:
            _runs.popitem(last=False)


def clear() -> None:
    """Forget every recorded run and cached file."""
    with _lock:
        _runs.clear()
        _source_files.clear()


def source_span(
    unit_path: str, relative_path: str, text: str, offset: int, length: int
) -> SourceSpan | None:
    """The span the user wrote for ``text[offset : offset + length]``.

    ``text`` is the preprocessed text of the unit at ``unit_path``
    (``relative_path`` from the project root) or text rewritten from it. The
    result is ``None`` when no preprocessor run of the unit is recorded, when
    ``text`` is the unit's own source, or when the origin cannot be read.
    """
    with _lock:
        runs = list(_runs.get(_key(unit_path), ()))
    if not runs or any(run.source.text == text for run in runs):
        return None
    run = next((run for run in runs if run.text == text), runs[-1])
    root = _project_root(unit_path, relative_path)
    start = _position(run, text, offset, unit_path, root)
    if start is None:
        return None
    end = _position(run, text, offset + max(length, 0), unit_path, root)
    if end is None or end.path != start.path or end.offset < start.offset:
        end = _end_on_line(start, length, run)
    return SourceSpan(
        file=_report_name(start.path, unit_path, relative_path, root),
        line=start.line,
        column=start.column,
        offset=start.offset,
        end_line=end.line,
        end_column=end.column,
        end_offset=end.offset,
    )


def _position(
    run: _Run, text: str, offset: int, unit_path: str, root: str | None
) -> _Position | None:
    offset = min(max(offset, 0), len(text))
    line_index = text.count("\n", 0, offset)
    line_start = text.rfind("\n", 0, offset) + 1
    line_end = text.find("\n", offset)
    written = text[line_start : len(text) if line_end < 0 else line_end]
    if text == run.text:
        output_index, unchanged = line_index, True
    else:
        alignment = _alignment(run, text)
        if line_index >= len(alignment):
            return None
        output_index, unchanged = alignment[line_index]
    if output_index >= len(run.origins):
        return None
    origin = _reported_origin(run.origins[output_index], unit_path, root)
    if origin is None:
        return None
    column = offset - line_start + 1 if unchanged else None
    return _locate(origin, column, written, run, unit_path)


def _reported_origin(
    origin: Origin | None, unit_path: str, root: str | None
) -> Origin | None:
    """``origin``, or the ``#include`` that leads to it from the unit or a header under the project root."""
    while origin is not None and origin[0]:
        path = str(origin[0])
        if _key(path) == _key(unit_path) or _inside(path, root):
            return origin
        origin = origin[2]
    return None


def _locate(
    origin: Origin,
    column: int | None,
    written: str,
    run: _Run,
    unit_path: str,
) -> _Position | None:
    """The position of ``column`` on the physical line of ``origin``. ``column`` stands when the physical line is
    the ``written`` line; otherwise the first non-blank column of the physical line is used.
    """
    path, line_number = str(origin[0]), int(origin[1])
    source = run.source if _key(path) == _key(unit_path) else _source_file(path)
    line = source.line(line_number) if source is not None else None
    if line is None:
        return None
    line_start, physical = line
    if column is None or physical != written:
        column = len(physical) - len(physical.lstrip()) + 1
    column = min(column, len(physical) + 1)
    return _Position(path, line_number, column, line_start + column - 1)


def _end_on_line(start: _Position, length: int, run: _Run) -> _Position:
    """An end on the start's line, at most ``length`` characters on, for a span whose end maps elsewhere."""
    same_file = _key(start.path) == _key(run.path)
    source = run.source if same_file else _source_file(start.path)
    line = source.line(start.line) if source is not None else None
    line_end = start.offset if line is None else line[0] + len(line[1])
    end_offset = min(start.offset + max(length, 0), max(line_end, start.offset))
    return _Position(
        start.path,
        start.line,
        start.column + (end_offset - start.offset),
        end_offset,
    )


def _alignment(run: _Run, text: str) -> tuple[tuple[int, bool], ...]:
    """For every line of ``text``, the output line of ``run`` it comes from and whether it is unchanged."""
    with _lock:
        cached = run.alignments.get(text)
        if cached is not None:
            run.alignments.move_to_end(text)
            return cached
    lines = text.split("\n")
    matcher = difflib.SequenceMatcher(None, run.lines, lines)
    last = max(len(run.lines) - 1, 0)
    alignment: list[tuple[int, bool]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        for j in range(j1, j2):
            if tag == "equal":
                alignment.append((i1 + (j - j1), True))
            elif i2 > i1:
                alignment.append((i1 + min(j - j1, i2 - i1 - 1), False))
            else:
                alignment.append((min(i1, last), False))
    result = tuple(alignment)
    with _lock:
        run.alignments[text] = result
        while len(run.alignments) > ALIGNMENT_CACHE_LIMIT:
            run.alignments.popitem(last=False)
    return result


def _project_root(unit_path: str, relative_path: str) -> str | None:
    parts = PurePosixPath(relative_path.replace("\\", "/")).parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        return None
    root = Path(os.path.abspath(unit_path))
    for _ in parts:
        root = root.parent
    return str(root)


def _inside(path: str, root: str | None) -> bool:
    if root is None:
        return False
    relative = os.path.relpath(os.path.abspath(path), root)
    return relative != os.pardir and not relative.startswith(os.pardir + os.sep)


def _report_name(
    path: str, unit_path: str, relative_path: str, root: str | None
) -> str:
    if _key(path) == _key(unit_path) or root is None:
        return relative_path
    return PurePosixPath(
        *Path(os.path.relpath(os.path.abspath(path), root)).parts
    ).as_posix()


def _source_file(path: str) -> _SourceFile | None:
    """The text of a header, cached while its size and modification time stay the same."""
    try:
        status = os.stat(path)
    except OSError:
        return None
    key = (_key(path), status.st_mtime_ns, status.st_size)
    with _lock:
        if key in _source_files:
            _source_files.move_to_end(key)
            return _source_files[key]
    try:
        source: _SourceFile | None = _SourceFile.of(
            Path(path).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError):
        source = None
    with _lock:
        _source_files[key] = source
        while len(_source_files) > SOURCE_FILE_CACHE_LIMIT:
            _source_files.popitem(last=False)
    return source


__all__ = [
    "SourceSpan",
    "clear",
    "record_preprocessed_text",
    "source_span",
]
