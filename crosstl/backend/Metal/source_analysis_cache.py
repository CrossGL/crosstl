"""Per-source analysis cache for the Metal importer.

The importer analyses whole translation units (namespaces, struct and template declarations, alias bindings) and
also small fragments such as a struct body or one template argument. Both used to share one small LRU, so a burst
of fragment lookups evicted the module's analysis and the next whole-module query rebuilt it from scratch; on a
module with many partial specializations that happened once per candidate specialization.

Fragments and translation units now live in separate LRUs, so fragments cannot evict a translation unit's
analysis. Each entry is keyed by the exact source text, so a cached analysis is only ever reused for identical
input.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Callable, Generic, Hashable, MutableMapping, TypeVar

#: Sources shorter than this many characters are fragments and use the fragment LRU.
FRAGMENT_LENGTH_LIMIT = 16384
#: Fragment analyses retained; fragments are small, so many of them stay cheap to keep.
FRAGMENT_CACHE_LIMIT = 256

AnalysisT = TypeVar("AnalysisT")
ValueT = TypeVar("ValueT")


class SourceAnalysisCache(Mapping[str, AnalysisT], Generic[AnalysisT]):
    """Two LRUs of analyses keyed by source text: one for translation units and one for fragments.

    It reads as a mapping from source text to analysis, so code that fingerprints or copies a preprocessor's state
    (``crosstl.project.source_closure_memo``) treats it like the plain dict it replaces.
    """

    __slots__ = ("_sources", "_fragments", "_source_limit")

    def __init__(self, source_limit: int) -> None:
        self._sources: dict[str, AnalysisT] = {}
        self._fragments: dict[str, AnalysisT] = {}
        self._source_limit = source_limit

    def get(self, code: str, factory: Callable[[], AnalysisT]) -> AnalysisT:
        """Return the analysis for ``code``, creating an empty one with ``factory`` on a miss."""
        if len(code) < FRAGMENT_LENGTH_LIMIT:
            cache, limit = self._fragments, FRAGMENT_CACHE_LIMIT
        else:
            cache, limit = self._sources, self._source_limit
        analysis = cache.pop(code, None)
        if analysis is None:
            analysis = factory()
        cache[code] = analysis
        if len(cache) > limit:
            del cache[next(iter(cache))]
        return analysis

    def clear(self) -> None:
        self._sources.clear()
        self._fragments.clear()

    def __contains__(self, code: object) -> bool:
        return code in self._sources or code in self._fragments

    def __getitem__(self, code: str) -> AnalysisT:
        try:
            return self._sources[code]
        except KeyError:
            return self._fragments[code]

    def __iter__(self) -> Iterator[str]:
        yield from self._sources
        yield from self._fragments

    def __len__(self) -> int:
        return len(self._sources) + len(self._fragments)


def memoized(
    values: MutableMapping[Hashable, object],
    key: Hashable,
    compute: Callable[[], ValueT],
) -> ValueT:
    """Return ``values[key]``, computing and storing it first when absent.

    ``compute`` must be a pure function of the source the mapping belongs to, and its result must not be mutated by
    any caller. An exception propagates and stores nothing, so the next call computes again.
    """
    try:
        return values[key]  # type: ignore[return-value]
    except KeyError:
        value = compute()
        values[key] = value
        return value
