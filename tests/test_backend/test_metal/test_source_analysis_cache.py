import collections
import collections.abc

import pytest

from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from crosstl.backend.Metal.source_analysis_cache import (
    FRAGMENT_CACHE_LIMIT,
    FRAGMENT_LENGTH_LIMIT,
    SourceAnalysisCache,
    memoized,
)

SCALARS = ["float", "uint", "int", "half", "short", "ushort", "char", "uchar"]


def test_fragment_lookups_do_not_evict_a_translation_unit():
    cache = SourceAnalysisCache(source_limit=2)
    unit = "x" * FRAGMENT_LENGTH_LIMIT
    analysis = cache.get(unit, dict)
    for index in range(FRAGMENT_CACHE_LIMIT * 2):
        cache.get(f"fragment {index}", dict)
    assert cache.get(unit, dict) is analysis


def test_each_tier_keeps_its_most_recent_entries():
    cache = SourceAnalysisCache(source_limit=2)
    units = [str(index) * FRAGMENT_LENGTH_LIMIT for index in range(3)]
    first, second, third = (cache.get(unit, dict) for unit in units)
    assert cache.get(units[2], dict) is third
    assert cache.get(units[1], dict) is second
    assert cache.get(units[0], dict) is not first
    fragments = [f"fragment {index}" for index in range(FRAGMENT_CACHE_LIMIT + 1)]
    oldest = cache.get(fragments[0], dict)
    for fragment in fragments[1:]:
        cache.get(fragment, dict)
    assert cache.get(fragments[0], dict) is not oldest
    cache.clear()
    assert len(cache) == 0


def test_cache_reads_as_a_mapping_of_both_tiers():
    # Project memoization fingerprints and copies a preprocessor's state, which holds this cache.
    cache = SourceAnalysisCache(source_limit=2)
    unit = "u" * FRAGMENT_LENGTH_LIMIT
    unit_analysis = cache.get(unit, dict)
    fragment_analysis = cache.get("fragment", dict)
    assert isinstance(cache, collections.abc.Mapping)
    assert list(cache) == [unit, "fragment"]
    assert cache[unit] is unit_analysis and cache["fragment"] is fragment_analysis
    assert dict(cache) == {unit: {}, "fragment": {}}
    with pytest.raises(KeyError):
        cache["absent"]


def test_memoized_computes_once_and_does_not_store_a_failure():
    values = {}
    calls = []

    def fail():
        calls.append("fail")
        raise ValueError("not cached")

    with pytest.raises(ValueError):
        memoized(values, "key", fail)
    assert "key" not in values
    assert memoized(values, "key", lambda: calls.append("ok") or 7) == 7
    assert memoized(values, "key", fail) == 7
    assert calls == ["fail", "ok"]


def test_constrained_partial_structs_scan_each_source_once(monkeypatch):
    # One whole-module scan per source text, however many specializations are tested against the partial.
    parameters = ", ".join(
        f"device Cell<{scalar}>* out{index} [[buffer({index})]]"
        for index, scalar in enumerate(SCALARS)
    )
    source = f"""
    template <typename T> constexpr constant bool accepts = is_same_v<T, float>;
    template <typename T, typename = void> struct Cell {{ uint value; }};
    template <typename T> struct Cell<T, enable_if_t<accepts<T>>> {{ float value; }};
    kernel void write_values({parameters}) {{}}
    """
    scans = collections.Counter()
    original = MetalPreprocessor._find_boolean_variable_templates

    def counted(self, code):
        scans[code] += 1
        return original(self, code)

    monkeypatch.setattr(MetalPreprocessor, "_find_boolean_variable_templates", counted)
    output = MetalPreprocessor()._materialize_explicit_template_struct_instantiations(
        source, allow_partial_object_specializations=True
    )
    assert "struct Cell_float_void { float value; }" in output
    for scalar in SCALARS[1:]:
        assert f"struct Cell_{scalar} {{ uint value; }}" in output
    assert scans and max(scans.values()) == 1
