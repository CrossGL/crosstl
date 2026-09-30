import textwrap

import pytest

from crosstl.backend.Metal.preprocessor import (
    MetalPreprocessor,
    _unresolved_metal_host_name_diagnostics,
    discover_metal_entry_points,
)
from crosstl.translator.entry_discovery import (
    ENTRY_DISCOVERY_AVAILABLE,
    ENTRY_DISCOVERY_UNAVAILABLE,
)
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources


def test_discovers_direct_included_and_host_named_metal_entries(tmp_path):
    include = tmp_path / "included.h"
    include.write_text(
        textwrap.dedent("""
            #if ENABLE_INCLUDED
            kernel void included_kernel(device float* output [[buffer(0)]]) {
              output[0] = 1.0f;
            }
            #endif
            """),
        encoding="utf-8",
    )
    source_path = tmp_path / "entries.metal"
    source = textwrap.dedent("""
        #include "included.h"

        // kernel void commented_kernel() {}
        #if 0
        kernel void inactive_kernel() {}
        #endif

        // This helper is used by a templated kernel.
        // An example may contain [kernel] in documentation.
        float helper(float value) {
          return value;
        }

        template <typename T>
        [[kernel]] void convert(device T* output [[buffer(0)]]) {
          output[0] = T(helper(1.0f));
        }

        template [[host_name("convert_float")]] [[kernel]]
        decltype(convert<float>) convert<float>;

        kernel void direct_kernel(device float* output [[buffer(0)]]) {
          output[0] = 2.0f;
        }

        [[host_name("renamed_kernel")]]
        kernel void source_name(device float* output [[buffer(0)]]) {
          output[0] = 3.0f;
        }
        """)
    source_path.write_text(source, encoding="utf-8")

    discovery = discover_metal_entry_points(
        source,
        file_path=str(source_path),
        include_paths=[str(tmp_path)],
        defines={"ENABLE_INCLUDED": "1"},
    )

    assert discovery.status == ENTRY_DISCOVERY_AVAILABLE
    assert [entry.name for entry in discovery.entries] == [
        "included_kernel",
        "convert_float",
        "direct_kernel",
        "renamed_kernel",
    ]
    assert [entry.stage for entry in discovery.entries] == ["compute"] * 4
    assert [entry.provenance.kind for entry in discovery.entries] == [
        "concrete",
        "host-named-materialization",
        "concrete",
        "concrete",
    ]
    assert discovery.entries[1].provenance.declared_name == "convert"
    assert discovery.entries[1].provenance.template_arguments == ("float",)
    assert discovery.entries[3].provenance.declared_name == "source_name"
    assert discovery.diagnostics == ()


def test_ignores_commented_entries_and_reports_dynamic_host_name():
    source = textwrap.dedent("""
        // instantiate_kernel("commented", convert, float)
        /*
        template [[host_name("also_commented")]] [[kernel]]
        decltype(convert<float>) convert<float>;
        */

        template <typename T>
        [[kernel]] void convert(device T* output [[buffer(0)]]) {
          output[0] = T(1);
        }

        template [[host_name(EXPORTED_NAME)]] [[kernel]]
        decltype(convert<float>) convert<float>;
        """)

    discovery = discover_metal_entry_points(source, file_path="entries.metal")

    assert discovery.entries == ()
    assert [diagnostic.code for diagnostic in discovery.diagnostics] == [
        "source.entry-discovery.unresolved-host-name"
    ]
    assert discovery.diagnostics[0].details == {"expression": "EXPORTED_NAME"}


def test_source_registry_reports_unavailable_entry_discovery():
    register_default_sources()
    cgl_spec = SOURCE_REGISTRY.get("cgl")
    assert cgl_spec is not None

    discovery = cgl_spec.discover_entry_points(
        "shader Empty {}",
        file_path="empty.cgl",
    )

    assert discovery.status == ENTRY_DISCOVERY_UNAVAILABLE
    assert discovery.source_backend == "cgl"
    assert discovery.source_path == "empty.cgl"
    assert discovery.entries == ()
    assert discovery.diagnostics == ()


@pytest.mark.parametrize(
    "spans",
    [
        [],
        [(0, 0)],
        [(0, 1), (2, 3)],
        [(9, 4), (4, 4)],
        [(0, 30), (5, 10)],
        [(5, 10), (0, 30)],
        [(0, 5), (5, 10), (5, 10)],
        [(10, 20), (0, 15), (30, 40)],
    ],
)
def test_discovery_exclusions_match_half_open_interval_membership(spans):
    preprocessor = MetalPreprocessor()
    for position in range(-1, 42):
        expected = any(start <= position < end for start, end in spans)
        assert (preprocessor._containing_span(position, spans) is not None) == expected


def test_unresolved_host_names_keep_locations_outside_unsorted_exclusions():
    source = "host_name(FIRST) host_name(SECOND) instantiate_kernel(THIRD, f, float)"
    first = source.index("host_name")
    second = source.index("host_name", first + 1)
    third = source.index("instantiate_kernel")
    spans = [(third + 1, len(source)), (first, second), (first, first + 1)]

    diagnostics = _unresolved_metal_host_name_diagnostics(
        MetalPreprocessor(), source, spans
    )

    assert [item.details for item in diagnostics] == [
        {"expression": "SECOND"},
        {"expression": "THIRD"},
    ]
    assert [item.location.offset for item in diagnostics] == [second, third]


def test_entry_discovery_indexes_large_exclusion_lists_once(monkeypatch):
    count = 256
    source = "template <typename T> [[kernel]] void convert(device T* out) {}\n"
    source += "\n".join(
        f"// host_name(COMMENT_{index})\n"
        f'template [[host_name("convert_{index}")]] [[kernel]] '
        "decltype(convert<float>) convert<float>;"
        for index in range(count)
    )
    large_builds = []
    large_queries = {}
    original = MetalPreprocessor._build_containing_span_lookup
    original_lookup = MetalPreprocessor._containing_span

    def counted_build(spans):
        if len(spans) >= count:
            large_builds.append(spans)
        return original(spans)

    def counted_lookup(self, position, spans):
        if len(spans) >= count:
            large_queries[id(spans)] = large_queries.get(id(spans), 0) + 1
        return original_lookup(self, position, spans)

    monkeypatch.setattr(
        MetalPreprocessor, "_build_containing_span_lookup", staticmethod(counted_build)
    )
    monkeypatch.setattr(MetalPreprocessor, "_containing_span", counted_lookup)
    discovery = discover_metal_entry_points(
        source, source_options={"preprocess": False}
    )

    assert [entry.name for entry in discovery.entries] == [
        f"convert_{index}" for index in range(count)
    ]
    assert discovery.diagnostics == ()
    assert large_builds
    assert len({id(spans) for spans in large_builds}) == len(large_builds)
    assert max(large_queries.values()) >= 2 * count


def test_discovery_exclusion_queries_do_not_rescan_span_collections():
    class CountedSpans(list):
        reads = 0

        def __getitem__(self, index):
            self.reads += 1
            return super().__getitem__(index)

        def __iter__(self):
            for index in range(len(self)):
                yield self[index]

    count = 4096
    spans = CountedSpans((index * 3, index * 3 + 2) for index in range(count))
    preprocessor = MetalPreprocessor()
    assert preprocessor._containing_span(0, spans) == (0, 2)
    spans.reads = 0

    for index in range(count):
        assert preprocessor._containing_span(index * 3, spans) == (
            index * 3,
            index * 3 + 2,
        )
    assert spans.reads <= 4 * count


def test_entry_discovery_rebuilds_indexes_for_changed_includes_and_defines(tmp_path):
    include = tmp_path / "entries.h"
    include.write_text("kernel void first() {}\n", encoding="utf-8")
    source = '#if ENABLE_ENTRIES\n#include "entries.h"\n#endif\n'
    path = tmp_path / "main.metal"

    def discover(enabled):
        return discover_metal_entry_points(
            source,
            file_path=str(path),
            include_paths=[str(tmp_path)],
            defines={"ENABLE_ENTRIES": str(enabled)},
        )

    first = discover(1)
    assert [entry.name for entry in first.entries] == ["first"]
    assert discover(1) == first
    assert discover(0).entries == ()
    include.write_text(
        "// kernel void first() {}\nkernel void next() {}\n", encoding="utf-8"
    )
    next_result = discover(1)
    assert [entry.name for entry in next_result.entries] == ["next"]
    assert next_result.diagnostics == ()
    assert discover(0).entries == ()


def test_entry_discovery_keeps_source_option_indexes_isolated():
    source = textwrap.dedent("""
        #define EXPORTED_NAME "converted"
        template <typename T> [[kernel]] void convert(device T* out) {}
        template [[host_name(EXPORTED_NAME)]] [[kernel]]
        decltype(convert<float>) convert<float>;
        """)

    expanded = discover_metal_entry_points(source)
    assert [entry.name for entry in expanded.entries] == ["converted"]
    assert expanded.diagnostics == ()
    raw = discover_metal_entry_points(source, source_options={"preprocess": False})
    assert raw.entries == ()
    assert [item.details for item in raw.diagnostics] == [
        {"expression": "EXPORTED_NAME"}
    ]
    assert discover_metal_entry_points(source) == expanded
