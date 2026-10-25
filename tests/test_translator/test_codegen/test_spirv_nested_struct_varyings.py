"""Structure-typed members of vertex outputs and fragment inputs that carry no location.

Metal nests a mesh pipeline's vertex and primitive structures in the fragment's ``[[stage_in]]`` struct
(``struct { VertexOut v; PrimitiveOut p; }``), whose fields carry no location. The SPIR-V generator used to declare
each nested member as one ``Input`` variable of structure type at one planned location: a structure that needs two
locations then overlapped the next member's (spirv-val ``VUID-StandaloneSpirv-OpEntryPoint-08721``), and a
``[[position]]`` field became a located varying instead of ``FragCoord``. Such members are now flattened like located
ones: every leaf field takes its own location from the member's planned one, and a built-in field its built-in.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
import crosstl.translator
from crosstl.translator.codegen.SPIRV_codegen import VulkanSPIRVCodeGen

METAL = """#include <metal_stdlib>
using namespace metal;
struct VOut { float4 position [[position]]; float3 normal; float2 uv; };
struct POut { float3 color; };
struct FIn { VOut v; POut p; };
fragment float4 fs(FIn in [[stage_in]]) {
    return float4(in.v.normal + in.p.color, in.v.uv.x);
}
"""

CROSSGL = """
shader NestedVaryings {
    struct Inner {
        vec3 a;
        float b;
    }

    struct VOut {
        vec4 position @gl_Position;
        Inner inner;
        vec2 uv;
    }

    vertex {
        VOut main(vec3 position @POSITION) {
            VOut o;
            o.position = vec4(position, 1.0);
            o.inner.a = position;
            o.inner.b = 1.0;
            o.uv = position.xy;
            return o;
        }
    }

    fragment {
        vec4 main(VOut input) @gl_FragColor {
            return vec4(input.inner.a * input.inner.b, input.uv.x);
        }
    }
}
"""


def _validate(tmp_path, assembly):
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not (spirv_as and spirv_val):
        pytest.skip("spirv-as and spirv-val are required")
    source = tmp_path / "nested.spvasm"
    binary = tmp_path / "nested.spv"
    source.write_text(assembly, encoding="utf-8")
    result = subprocess.run(
        [spirv_as, "--target-env", "vulkan1.3", str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run(
        [spirv_val, "--target-env", "vulkan1.3", str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _locations(assembly):
    names = dict(re.findall(r'OpName (%\w+) "(\w+)"', assembly))
    return {
        names.get(target, target): int(location)
        for target, location in re.findall(
            r"OpDecorate (%\w+) Location (\d+)", assembly
        )
    }


def test_metal_nested_stage_in_structs_flatten_to_distinct_locations(tmp_path):
    path = tmp_path / "nested.metal"
    path.write_text(METAL, encoding="utf-8")
    assembly = crosstl.translate(str(path), backend="vulkan")

    locations = _locations(assembly)
    assert locations["CrossGL_fragment_input_in_v_normal"] == 0
    assert locations["CrossGL_fragment_input_in_v_uv"] == 1
    assert locations["CrossGL_fragment_input_in_p_color"] == 2
    assert "BuiltIn FragCoord" in assembly
    assert not any("position" in name for name in locations)
    _validate(tmp_path, assembly)


def test_nested_vertex_output_and_fragment_input_take_the_same_locations(tmp_path):
    shader = crosstl.translator.parse(CROSSGL)
    assembly = VulkanSPIRVCodeGen().generate(shader)
    _validate(tmp_path, assembly)

    path = tmp_path / "nested.cgl"
    path.write_text(CROSSGL, encoding="utf-8")
    locations = _locations(crosstl.translate(str(path), backend="vulkan"))
    outputs = {
        name.rsplit("_", 1)[-1]: location
        for name, location in locations.items()
        if "vertex_output" in name
    }
    inputs = {
        name.rsplit("_", 1)[-1]: location
        for name, location in locations.items()
        if "fragment_input" in name
    }
    assert outputs == inputs == {"a": 0, "b": 1, "uv": 2}
