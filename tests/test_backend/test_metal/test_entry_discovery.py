import textwrap

from crosstl.backend.Metal.preprocessor import discover_metal_entry_points
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
