"""Metal runtime object types (indirect command buffers, rasterization rate maps) are refused for other targets."""

import pytest

import crosstl
from crosstl.translator.semantics import (
    MetalRuntimeTypeUnsupportedError,
    apply_rule_carrying,
)
from crosstl.translator.semantics.metal_runtime_types import metal_runtime_type_uses
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources

ENCODER = """#include <metal_stdlib>
using namespace metal;
struct Container {
    command_buffer commands [[id(0)]];
};
kernel void encode(device Container* container [[buffer(0)]],
                   device const uint* counts [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    render_command command(container->commands, i);
    if (counts[i] == 0) {
        command.reset();
    }
}
"""


def translate(tmp_path, source, backend):
    path = tmp_path / "shader.metal"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize("backend", ["directx", "opengl", "vulkan"])
def test_indirect_command_buffers_are_refused_for_other_targets(tmp_path, backend):
    with pytest.raises(MetalRuntimeTypeUnsupportedError) as raised:
        translate(tmp_path, ENCODER, backend)
    assert raised.value.type_name == "command_buffer"
    assert raised.value.code == (
        f"project.translate.{backend}-metal-runtime-type-unsupported"
    )
    assert "recorded on the host" in str(raised.value)


def test_metal_keeps_its_indirect_command_buffers(tmp_path):
    metal = translate(tmp_path, ENCODER, "metal")
    assert "command_buffer commands" in metal
    assert "render_command" in metal


def _program(source):
    register_default_sources()
    return SOURCE_REGISTRY.get("cgl").parse(source)


def test_only_reachable_uses_are_refused():
    # A structure holding a runtime object that no entry reaches (a header shared with the host) is harmless.
    program = _program(
        "shader s {\n"
        "    struct Unused { render_command command; };\n"
        "    struct Wrapper { Unused inner; };\n"
        "    void helper(Wrapper w) { }\n"
        "    compute { void main() { int x = 1; } }\n"
        "}\n"
    )
    assert metal_runtime_type_uses(program) == []
    apply_rule_carrying(program, "cgl", "vulkan")


def test_uses_through_nested_structures_and_rate_maps_are_found():
    program = _program(
        "shader s {\n"
        "    struct Inner { rasterization_rate_map_decoder decoder; };\n"
        "    struct Outer { Inner inner; };\n"
        "    void helper(Outer o) { }\n"
        "    fragment { vec4 main() @gl_FragColor { Outer o; helper(o); return vec4(1.0); } }\n"
        "}\n"
    )
    assert [name for name, _ in metal_runtime_type_uses(program)] == [
        "rasterization_rate_map_decoder",
        "rasterization_rate_map_decoder",
    ]
    with pytest.raises(MetalRuntimeTypeUnsupportedError, match="shading rates"):
        apply_rule_carrying(program, "cgl", "directx")
