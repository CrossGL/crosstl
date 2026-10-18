"""Classify what an importer did with one input, and reduce a crash to a stable signature.

- ``accepted``: the translation returned.
- ``refused``: it raised a typed diagnostic (an exception with a non-empty ``project_diagnostic_code``).
- ``rejected``: CrossTL itself raised ``SyntaxError``, ``ValueError``, ``TypeError`` or ``NotImplementedError``
  with an explicit ``raise`` statement: a deliberate but untyped rejection. These are counted, not failed; typing
  them is the fail-closed work, tracked by ``tests/conformance/fail_closed.py``.
- ``crash``: anything else. An exception raised implicitly (an ``IndexError`` from a subscript, a ``ValueError``
  from ``int()``, an ``AttributeError`` on ``None``), any exception raised outside CrossTL, ``RecursionError``,
  ``MemoryError`` and ``AssertionError``. The campaign adds ``timeout`` and ``worker-crash``.

A signature names the importer, the route, the exception type and the CrossTL function that raised it, without
line numbers, so it survives unrelated edits. A ``RecursionError`` is attributed to the function that repeats most
in its traceback.
"""

from __future__ import annotations

import ast
import collections
import functools
import linecache
import os
import traceback

import crosstl

PACKAGE_ROOT = os.path.dirname(os.path.abspath(crosstl.__file__))
REPOSITORY_ROOT = os.path.dirname(PACKAGE_ROOT)
REJECTION_TYPES = (SyntaxError, ValueError, TypeError, NotImplementedError)
CRASH_STATUSES = ("crash", "timeout", "worker-crash")


def _in_package(filename: str) -> bool:
    return os.path.abspath(filename).startswith(PACKAGE_ROOT + os.sep)


@functools.lru_cache(maxsize=None)
def _raise_lines(filename: str) -> frozenset:
    """Every line covered by a ``raise`` statement in ``filename``."""
    source = "".join(linecache.getlines(filename))
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return frozenset()
    lines = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise):
            end = getattr(node, "end_lineno", None) or node.lineno
            lines.update(range(node.lineno, end + 1))
    return frozenset(lines)


def _frames(error: BaseException):
    return traceback.extract_tb(error.__traceback__)


def _site(frame) -> str:
    relative = os.path.relpath(os.path.abspath(frame.filename), REPOSITORY_ROOT)
    return f"{relative.replace(os.sep, '/')}:{frame.name}"


def is_typed(error: BaseException) -> bool:
    code = getattr(error, "project_diagnostic_code", None)
    return isinstance(code, str) and bool(code)


def classify(error: BaseException | None) -> str:
    if error is None:
        return "accepted"
    if is_typed(error):
        return "refused"
    if isinstance(error, (RecursionError, MemoryError, AssertionError)):
        return "crash"
    frames = _frames(error)
    if not frames or not _in_package(frames[-1].filename):
        return "crash"
    explicit = frames[-1].lineno in _raise_lines(frames[-1].filename)
    if explicit and isinstance(error, REJECTION_TYPES):
        return "rejected"
    return "crash"


def raising_site(error: BaseException) -> str:
    """The CrossTL function an exception is attributed to (``<external>`` when none is on the stack)."""
    frames = [frame for frame in _frames(error) if _in_package(frame.filename)]
    if not frames:
        return "<external>"
    if isinstance(error, RecursionError):
        counts = collections.Counter(_site(frame) for frame in frames)
        best = max(counts.values())
        for frame in reversed(frames):
            if counts[_site(frame)] == best:
                return _site(frame)
    return _site(frames[-1])


def signature(importer: str, backend: str, status: str, error=None, detail="") -> str:
    route = importer if backend == "cgl" else f"{importer}->{backend}"
    if error is None:
        return f"{route}|{status}{'|' + detail if detail else ''}"
    return f"{route}|{status}|{type(error).__name__}|{raising_site(error)}"
