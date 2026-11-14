"""Compiled regular expressions for the patterns the Metal importer builds from source names.

``re`` keeps 512 compiled patterns. The importer builds patterns from struct, alias and owner names, and a module
with many template instantiations cycles through a few thousand of them on every type it canonicalizes, so ``re``'s
own cache misses on nearly every call and each miss compiles the pattern again. This cache is sized for such
modules; a pattern compiles once per process.
"""

from __future__ import annotations

import functools
import re

PATTERN_CACHE_SIZE = 16384


@functools.lru_cache(maxsize=PATTERN_CACHE_SIZE)
def pattern(source: str) -> re.Pattern[str]:
    """``re.compile(source)``, cached beyond ``re``'s own limit."""
    return re.compile(source)
