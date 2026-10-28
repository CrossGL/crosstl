"""Process-wide reuse of pure results computed from a source and its includes.

Repository-scale translation reads the same source file several times: the
project scan discovers its entry points, every report validation rescans it,
and the runtime manifest reflects its host interface. Each result is a
deterministic function of the source text, the files its preprocessor can
read and the call options. This module keys results on exactly those inputs.

The include closure is derived conservatively from the literal ``#include``,
``#import`` and ``__has_include`` operands of every reachable file, whether or
not the directive sits in an active conditional branch. Every candidate the
quote-then-include-path search could probe is recorded together with its
existence, and every existing candidate contributes its bytes. A hit is
therefore only possible while every file the preprocessor could read is
byte-identical and every absent candidate is still absent. Any operand that is
not a literal (a macro-computed include, for example) makes the source
uncacheable rather than approximated.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import re
from collections import OrderedDict
from threading import Lock
from typing import Any, Callable, Mapping, Sequence, TypeVar

T = TypeVar("T")

# Source backends whose preprocessor resolves includes by searching the
# including file's directory (quoted operands only) and then the configured
# include paths, which is the search the closure below models.
QUOTE_THEN_INCLUDE_PATH_BACKENDS = frozenset({"directx", "metal"})

CLOSURE_FILE_LIMIT = 4096
CLOSURE_BYTE_LIMIT = 64 * 1024 * 1024
MEMO_ENTRY_LIMIT = 64

_DIRECTIVE = re.compile(
    r"^[ \t]*#[ \t]*(include_next|include|import)\b(.*)$", re.MULTILINE
)
_HAS_INCLUDE = re.compile(r"__has_include(_next)?\b")
_LITERAL_OPERAND = re.compile(r"\s*([<\"])([^>\"]+)[>\"]")
_HAS_INCLUDE_OPERAND = re.compile(r"\s*\(\s*([<\"])([^>\"]+)[>\"]\s*\)")


class UncacheableSource(Exception):
    """The include closure of a source cannot be determined exactly."""


def _decoded(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    return data.decode("utf-8-sig", errors="replace")


def _logical_text(text: str) -> str:
    # Directive operands may continue onto the next physical line.
    return re.sub(r"\\[ \t]*\r?\n", "", text)


def _include_operands(text: str) -> list[tuple[str, str]]:
    logical = _logical_text(text)
    operands = []
    for match in _DIRECTIVE.finditer(logical):
        if match.group(1) == "include_next":
            # The search resumes after the including directory; not modelled.
            raise UncacheableSource("#include_next operand")
        operand = _LITERAL_OPERAND.match(match.group(2))
        if operand is None:
            raise UncacheableSource(f"non-literal #{match.group(1)} operand")
        operands.append((operand.group(1), operand.group(2)))
    for match in _HAS_INCLUDE.finditer(logical):
        if match.group(1):
            raise UncacheableSource("__has_include_next operand")
        operand = _HAS_INCLUDE_OPERAND.match(logical, match.end())
        if operand is None:
            raise UncacheableSource("non-literal __has_include operand")
        operands.append((operand.group(1), operand.group(2)))
    return operands


def _search_bases(
    delimiter: str, includer: str | None, include_paths: Sequence[str]
) -> list[str]:
    bases = []
    if delimiter == '"' and includer:
        bases.append(os.path.dirname(includer))
    bases.extend(os.fspath(path) for path in include_paths)
    return bases


def include_closure_fingerprint(
    source: str,
    file_path: str | None,
    include_paths: Sequence[str],
) -> str:
    """Return a digest of the source and of every file its includes can reach.

    Raises :class:`UncacheableSource` when an operand is not a literal path or
    the closure exceeds the file or byte limits.
    """

    records: list[list[Any]] = [
        ["source", hashlib.sha256(source.encode("utf-8", "surrogatepass")).hexdigest()],
        ["file", os.path.abspath(file_path) if file_path else None],
        ["includePaths", [os.path.abspath(os.fspath(p)) for p in include_paths]],
    ]
    pending: list[tuple[str, str | None]] = [(source, file_path)]
    visited: set[str] = set()
    total_bytes = 0
    while pending:
        text, includer = pending.pop()
        for delimiter, target in _include_operands(text):
            for base in _search_bases(delimiter, includer, include_paths):
                candidate = os.path.join(base, target)
                absolute = os.path.abspath(candidate)
                if not os.path.isfile(absolute):
                    records.append(["absent", absolute])
                    continue
                if absolute in visited:
                    continue
                visited.add(absolute)
                if len(visited) > CLOSURE_FILE_LIMIT:
                    raise UncacheableSource("include closure has too many files")
                try:
                    with open(absolute, "rb") as handle:
                        data = handle.read()
                except OSError as exc:
                    raise UncacheableSource(f"unreadable include: {exc}") from exc
                total_bytes += len(data)
                if total_bytes > CLOSURE_BYTE_LIMIT:
                    raise UncacheableSource("include closure is too large")
                records.append(["present", absolute, hashlib.sha256(data).hexdigest()])
                pending.append((_decoded(data), candidate))
    records[3:] = sorted(records[3:], key=lambda record: record[:2])
    encoded = json.dumps(records, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [type(value).__qualname__, _canonical(dataclasses.asdict(value))]
    if isinstance(value, Mapping):
        return [[str(key), _canonical(value[key])] for key in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical(item) for item in value), key=repr)
    raise TypeError(f"unsupported memo key input {type(value)!r}")


class SourceClosureMemo:
    """Bounded LRU of results keyed by call inputs and an include closure.

    ``guards`` are compared by identity, so a replaced implementation (for
    example a different registered frontend) never reuses an older result.
    Stored and returned values are deep copies, so callers may mutate them.
    """

    def __init__(self, limit: int = MEMO_ENTRY_LIMIT):
        self.limit = int(limit)
        self._entries: OrderedDict[str, tuple[tuple[Any, ...], Any]] = OrderedDict()
        self._lock = Lock()
        self.hits = 0
        self.misses = 0

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.hits = 0
            self.misses = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def compute(
        self,
        *,
        source: str,
        file_path: str | None,
        include_paths: Sequence[str],
        inputs: Any,
        guards: tuple[Any, ...],
        produce: Callable[[], T],
    ) -> T:
        """Return ``produce()``, reusing an earlier result for identical inputs."""

        try:
            closure = include_closure_fingerprint(source, file_path, include_paths)
            encoded = json.dumps(
                [closure, _canonical(inputs)],
                ensure_ascii=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (UncacheableSource, TypeError, ValueError, RecursionError):
            return produce()
        key = hashlib.sha256(encoded.encode("ascii")).hexdigest()
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and len(entry[0]) == len(guards):
                if all(old is new for old, new in zip(entry[0], guards)):
                    self._entries.move_to_end(key)
                    self.hits += 1
                    return copy.deepcopy(entry[1])
            self.misses += 1
        result = produce()
        retained = copy.deepcopy(result)
        with self._lock:
            self._entries[key] = (guards, retained)
            self._entries.move_to_end(key)
            while len(self._entries) > self.limit:
                self._entries.popitem(last=False)
        return result


ENTRY_DISCOVERY_MEMO = SourceClosureMemo()
SOURCE_HOST_INTERFACE_MEMO = SourceClosureMemo()
PREPROCESSOR_RUN_MEMO = SourceClosureMemo()


def memoized_entry_discovery(
    source_spec: Any,
    source: str,
    *,
    file_path: str,
    include_paths: Sequence[str],
    defines: Mapping[str, str],
    source_options: Mapping[str, Any],
) -> Any:
    """Discover entry points, reusing a result for an identical include closure."""

    def produce() -> Any:
        return source_spec.discover_entry_points(
            source,
            file_path=file_path,
            include_paths=include_paths,
            defines=defines,
            source_options=source_options,
        )

    if source_spec.name not in QUOTE_THEN_INCLUDE_PATH_BACKENDS:
        return produce()
    return ENTRY_DISCOVERY_MEMO.compute(
        source=source,
        file_path=file_path,
        include_paths=include_paths,
        inputs=["entry-discovery", source_spec.name, defines, source_options],
        guards=(
            source_spec,
            source_spec.entry_point_discoverer,
            type(source_spec).discover_entry_points,
        ),
        produce=produce,
    )


def memoized_source_host_interface(
    source_spec: Any,
    source: str,
    *,
    file_path: str,
    reflect: Callable[..., T],
    reflect_options: Mapping[str, Any],
) -> T:
    """Return ``reflect(source_spec.parse(source), **reflect_options)`` reused.

    The parse uses no configured include paths, so only quoted includes next
    to the source can contribute.
    """

    def produce() -> T:
        return reflect(
            source_spec.parse(source, file_path=file_path), **reflect_options
        )

    if source_spec.name not in QUOTE_THEN_INCLUDE_PATH_BACKENDS:
        return produce()
    return SOURCE_HOST_INTERFACE_MEMO.compute(
        source=source,
        file_path=file_path,
        include_paths=(),
        inputs=["source-host-interface", source_spec.name, reflect_options],
        guards=(
            source_spec,
            type(source_spec).parse,
            *source_spec.load_lexer_parser(),
            reflect,
        ),
        produce=produce,
    )


def memoized_preprocessor_run(
    preprocessor: Any,
    source: str,
    *,
    file_path: str,
    source_backend: str,
    run: Callable[..., str],
) -> str:
    """Return ``run(preprocessor, source, file_path=...)`` and its state change.

    The key covers the preprocessor's complete state before the run, so a hit
    restores the state (macro table, include bookkeeping) that the run would
    have left behind, and later passes over the same preprocessor see exactly
    what a fresh run produces.
    """

    state = vars(preprocessor)

    def produce() -> tuple[str, dict[str, Any]]:
        text = run(preprocessor, source, file_path=file_path)
        return text, dict(state)

    if source_backend not in QUOTE_THEN_INCLUDE_PATH_BACKENDS:
        return produce()[0]
    text, final_state = PREPROCESSOR_RUN_MEMO.compute(
        source=source,
        file_path=file_path,
        include_paths=list(getattr(preprocessor, "include_paths", ()) or ()),
        inputs=["preprocessor-run", source_backend, state],
        guards=(type(preprocessor), run),
        produce=produce,
    )
    state.clear()
    state.update(final_state)
    return text
