"""Stage-interface locations: one value takes one location on every target.

The host binds a vertex input, and the next stage reads a varying, by location (Metal by ``[[attribute(n)]]`` and
``[[user(name)]]``). These tests translate one HLSL interface to GLSL, SPIR-V, WGSL and MSL and check that every
target gives each value the location :mod:`crosstl.translator.codegen.stage_io_semantic_locations` defines, then
validate the output with glslang, spirv-val, naga and the Metal compiler where they are installed.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator import parse
from crosstl.translator.codegen.stage_io_semantic_locations import (
    allocate_locations,
    semantic_location,
)
from crosstl.translator.semantics import (
    FragmentColourOutputUnsupportedError,
    name_fragment_colour_outputs,
)

_TARGETS = ("opengl", "vulkan", "wgsl", "metal")

_VERTEX_INPUTS = """
struct VSInput {
    float3 position : POSITION;
    float3 normal : NORMAL;
    float4 weights : BLENDWEIGHT;
    float2 uv : TEXCOORD;
    float4 color : COLOR;
    float2 detail : TEXCOORD1;
};
struct PSInput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float4 color : COLOR0;
    float4 extra : EXTRA;
};
PSInput VSMain(VSInput input) {
    PSInput output;
    output.position = float4(input.position + input.normal * input.weights.x, 1.0);
    output.uv = input.uv + input.detail;
    output.color = input.color;
    output.extra = input.weights;
    return output;
}
float4 PSMain(PSInput input) : SV_Target {
    return input.color * input.uv.x + input.extra;
}
"""

# POSITION, NORMAL, TEXCOORD (= TEXCOORD0), TEXCOORD1 and COLOR take fixed locations; BLENDWEIGHT takes the first
# location none of them holds.
_EXPECTED_VERTEX_INPUTS = {
    "position": 0,
    "normal": 1,
    "weights": 2,
    "uv": 5,
    "color": 13,
    "detail": 6,
}

_VERTEX_ID_MEMBER = """
struct VSInput {
    float3 position : POSITION;
    uint id : SV_VertexID;
};
float4 VSMain(VSInput input) : SV_Position {
    return float4(input.position * float(input.id), 1.0);
}
float4 PSMain() : SV_Target {
    return float4(1.0, 0.0, 0.0, 1.0);
}
"""

_UNINDEXED_LINK = """
struct VSOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD;
    float4 color : COLOR;
};
struct PSInput {
    float4 position : SV_Position;
    float4 color : COLOR0;
    float2 uv : TEXCOORD0;
};
VSOutput VSMain(float3 position : POSITION, float2 uv : TEXCOORD) {
    VSOutput output;
    output.position = float4(position, 1.0);
    output.uv = uv;
    output.color = float4(uv, 0.0, 1.0);
    return output;
}
float4 PSMain(PSInput input) : SV_Target {
    return input.color * input.uv.x;
}
"""

_COLOUR_TARGETS = """
struct PSOutput {
    float4 albedo : COLOR0;
    float4 normal : COLOR1;
    float4 emissive : SV_Target2;
};
float4 VSMain(float3 position : POSITION) : SV_Position {
    return float4(position, 1.0);
}
PSOutput PSMain(float4 position : SV_Position) {
    PSOutput output;
    output.albedo = float4(1.0, 0.0, 0.0, 1.0);
    output.normal = float4(0.0, 1.0, 0.0, 1.0);
    output.emissive = float4(0.0, 0.0, 1.0, 1.0);
    return output;
}
"""

_FRAG_DATA = """
shader FragData {
    struct FSOutput {
        vec4 base @ gl_FragColor;
        vec4 extra @ gl_FragData[1];
    };
    vertex {
        vec4 main() @ gl_Position {
            return vec4(0.0, 0.0, 0.0, 1.0);
        }
    }
    fragment {
        FSOutput main() {
            FSOutput output;
            output.base = vec4(1.0);
            output.extra = vec4(0.5);
            return output;
        }
    }
}
"""


def _run(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr


def _translate(tmp_path, source, suffix, backend):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def _validate(tmp_path, backend, code, glsl_stages=("vert", "frag")):
    if backend == "opengl" and shutil.which("glslangValidator"):
        # Both stages sit behind the GL_*_SHADER macros glslang predefines per stage.
        for stage in glsl_stages:
            _run(["glslangValidator", "--stdin", "-S", stage], input=code)
    elif backend == "vulkan" and shutil.which("spirv-as") and shutil.which("spirv-val"):
        asm, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
        asm.write_text(code)
        _run(["spirv-as", "--target-env", "vulkan1.1", str(asm), "-o", str(binary)])
        _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])
    elif backend == "wgsl" and shutil.which("naga"):
        source = tmp_path / "out.wgsl"
        source.write_text(code)
        _run(["naga", str(source)])
    elif backend == "metal" and shutil.which("xcrun"):
        found = subprocess.run(
            ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True
        )
        if found.returncode != 0:
            return
        source = tmp_path / "out.metal"
        source.write_text(code)
        _run(
            ["xcrun", "-sdk", "macosx", "metal", "-c", str(source)]
            + ["-o", str(tmp_path / "out.air")]
        )


def _glsl_stage(code, stage):
    return "".join(
        re.findall(rf"#ifdef GL_{stage}_SHADER\n(.*?)#endif", code, flags=re.S)
    )


def _struct_body(code, name):
    match = re.search(rf"struct {name} \{{(.*?)\}};", code, flags=re.S)
    assert match is not None, code
    return match.group(1)


def _spirv_locations(code):
    """``{OpName: location}`` of every decorated variable; the ids may be numeric when SPIRV-Tools is absent."""
    names = dict(re.findall(r"OpName (%\w+) \"(\w+)\"", code))
    return {
        names[ident]: int(location)
        for ident, location in re.findall(r"OpDecorate (%\w+) Location (\d+)", code)
        if ident in names
    }


def _named(locations, prefix, member):
    found = [
        location
        for name, location in locations.items()
        if name.startswith(prefix) and name.endswith(f"_{member}")
    ]
    assert len(found) == 1, (prefix, member, locations)
    return found[0]


def _vertex_input_locations(backend, code):
    if backend == "opengl":
        return {
            name: int(location)
            for location, name in re.findall(
                r"layout\(location = (\d+)\) in \w+ (\w+);", _glsl_stage(code, "VERTEX")
            )
        }
    if backend == "vulkan":
        locations = _spirv_locations(code)
        return {
            member: _named(locations, "CrossGL_vertex_input_", member)
            for member in _EXPECTED_VERTEX_INPUTS
        }
    pattern = {
        "wgsl": r"@location\((\d+)\) (\w+):",
        "metal": r"\w+ (\w+) \[\[attribute\((\d+)\)\]\];",
    }[backend]
    found = re.findall(pattern, _struct_body(code, "VSInput"))
    if backend == "metal":
        return {name: int(location) for name, location in found}
    return {name: int(location) for location, name in found}


@pytest.mark.parametrize("backend", _TARGETS)
def test_vertex_inputs_take_one_location_on_every_target(tmp_path, backend):
    code = _translate(tmp_path, _VERTEX_INPUTS, ".hlsl", backend)

    assert _vertex_input_locations(backend, code) == _EXPECTED_VERTEX_INPUTS
    _validate(tmp_path, backend, code)


def test_glsl_vertex_input_members_with_system_values_read_the_builtin(tmp_path):
    code = _translate(tmp_path, _VERTEX_ID_MEMBER, ".hlsl", "opengl")

    # The member used to be declared as a user input `in uint id;`, an unbound attribute.
    assert not re.search(r"\bin uint id\b", code)
    assert "uint(gl_VertexID)" in code
    assert "layout(location = 0) in vec3 position;" in code
    _validate(tmp_path, "opengl", code)


def _varying_locations(backend, code):
    """``(vertex output, fragment input)`` locations of ``uv`` and ``color``."""
    if backend == "opengl":
        vertex, fragment = _glsl_stage(code, "VERTEX"), _glsl_stage(code, "FRAGMENT")
        pattern = r"layout\(location = (\d+)\) {} vec\d \w*({})\b;"

        def find(text, direction, member):
            match = re.search(pattern.format(direction, member), text)
            assert match is not None, (direction, member, text)
            return int(match.group(1))

        return {
            member: (find(vertex, "out", member), find(fragment, "in", member))
            for member in ("uv", "color")
        }
    if backend == "vulkan":
        locations = _spirv_locations(code)
        return {
            member: (
                _named(locations, "CrossGL_vertex_output_", member),
                _named(locations, "CrossGL_fragment_input_", member),
            )
            for member in ("uv", "color")
        }
    pattern = {
        "wgsl": r"@location\((\w+)\) {}:",
        "metal": r" {} \[\[user\((\w+)\)\]\];",
    }[backend]

    def find(struct, member):
        match = re.search(pattern.format(member), _struct_body(code, struct))
        assert match is not None, (struct, member, code)
        return match.group(1)

    return {
        member: (find("VSOutput", member), find("PSInput", member))
        for member in ("uv", "color")
    }


@pytest.mark.parametrize("backend", _TARGETS)
def test_unindexed_semantics_link_to_index_zero(tmp_path, backend):
    """HLSL reads ``TEXCOORD`` as ``TEXCOORD0`` and ``COLOR`` as ``COLOR0``, so the stages link."""
    code = _translate(tmp_path, _UNINDEXED_LINK, ".hlsl", backend)

    varyings = _varying_locations(backend, code)
    for member, (written, read) in varyings.items():
        assert written == read, (member, varyings)
    if backend != "metal":
        assert {member: int(pair[0]) for member, pair in varyings.items()} == {
            "uv": 5,
            "color": 13,
        }
    _validate(tmp_path, backend, code)


_COLOUR_TARGET_LINES = {
    "opengl": (
        "layout(location = 0) out vec4 albedo;",
        "layout(location = 1) out vec4 normal;",
        "layout(location = 2) out vec4 emissive;",
    ),
    "wgsl": (
        "@location(0) albedo: vec4<f32>",
        "@location(1) normal: vec4<f32>",
        "@location(2) emissive: vec4<f32>",
    ),
    "metal": (
        "float4 albedo [[color(0)]];",
        "float4 normal [[color(1)]];",
        "float4 emissive [[color(2)]];",
    ),
    "directx": (
        "float4 albedo: SV_TARGET0;",
        "float4 normal: SV_TARGET1;",
        "float4 emissive: SV_TARGET2;",
    ),
}


@pytest.mark.parametrize("backend", [*_TARGETS, "directx"])
def test_fragment_colour_outputs_name_colour_attachments(tmp_path, backend):
    """A pixel-shader ``COLORn`` output is render target ``n``, as ``SV_Targetn`` is."""
    code = _translate(tmp_path, _COLOUR_TARGETS, ".hlsl", backend)

    if backend == "vulkan":
        locations = _spirv_locations(code)
        assert {
            member: _named(locations, "CrossGL_fragment_output_", member)
            for member in ("albedo", "normal", "emissive")
        } == {"albedo": 0, "normal": 1, "emissive": 2}
    else:
        # clang-format, when installed, spaces the HLSL semantic colon.
        code = re.sub(r"(\w) : (SV_)", r"\1: \2", code)
        for line in _COLOUR_TARGET_LINES[backend]:
            assert line in code
    assert "attribute(13)" not in code
    _validate(tmp_path, backend, code)


@pytest.mark.parametrize("backend", _TARGETS)
def test_frag_data_structure_members_name_colour_attachments(tmp_path, backend):
    code = _translate(tmp_path, _FRAG_DATA, ".cgl", backend)

    if backend == "vulkan":
        locations = _spirv_locations(code)
        assert _named(locations, "CrossGL_fragment_output_", "base") == 0
        assert _named(locations, "CrossGL_fragment_output_", "extra") == 1
    else:
        expected = {
            "opengl": (
                "layout(location = 0) out vec4 base;",
                "layout(location = 1) out vec4 extra;",
            ),
            "wgsl": ("@location(0) base: vec4<f32>", "@location(1) extra: vec4<f32>"),
            "metal": ("float4 base [[color(0)]];", "float4 extra [[color(1)]];"),
        }[backend]
        for line in expected:
            assert line in code
    _validate(tmp_path, backend, code)


def test_colour_outputs_on_a_structure_another_entry_reads_are_refused():
    program = parse("""
        shader Shared {
            struct Shade {
                vec4 position @ gl_Position;
                vec4 color @ Color0;
            };
            vertex {
                Shade main() {
                    Shade output;
                    output.position = vec4(0.0);
                    output.color = vec4(1.0);
                    return output;
                }
            }
            fragment {
                Shade main() {
                    Shade output;
                    output.position = vec4(0.0);
                    output.color = vec4(1.0);
                    return output;
                }
            }
        }
        """)

    with pytest.raises(FragmentColourOutputUnsupportedError, match="Shade"):
        name_fragment_colour_outputs(program, "opengl")


@pytest.mark.parametrize(
    "spellings, location",
    [
        (("POSITION", "POSITION0", "Position", "Position0"), 0),
        (("NORMAL", "NORMAL0", "Normal", "Normal0"), 1),
        (("TANGENT", "TANGENT0", "Tangent0"), 2),
        (("BINORMAL", "BINORMAL0", "Binormal0"), 3),
        (("TEXCOORD", "TEXCOORD0", "TexCoord", "TexCoord0"), 5),
        (("TEXCOORD7", "TexCoord7"), 12),
        (("COLOR", "COLOR0", "Color", "Color0"), 13),
        (("COLOR7", "Color7"), 20),
    ],
)
def test_semantic_spellings_share_one_location(spellings, location):
    assert {semantic_location(spelling) for spelling in spellings} == {location}


def test_semantics_without_a_fixed_location_take_the_first_free_one():
    assert semantic_location("BLENDWEIGHT") is None
    assert semantic_location(None) is None
    # (fixed location or None, span) in declaration order: the matrix needs four free locations in a row.
    entries = [(None, 1), (0, 1), (None, 4), (5, 1), (None, 1)]
    assert allocate_locations(entries) == [1, 0, 6, 5, 10]
    assert allocate_locations([(None, 1), (None, 1)], occupied=[0]) == [1, 2]


_GLSL_INPUT_ARRAY = """#version 450
layout(constant_id = 3) const int count = 3;
in vec4 weights[count];
in vec2 uv;
void main() {
    gl_Position = weights[0] + weights[2] + vec4(uv, 0.0, 0.0);
}
"""


@pytest.mark.parametrize("backend", _TARGETS)
def test_unlocated_glsl_inputs_take_locations_after_constant_sized_arrays(
    tmp_path, backend
):
    """A specialization-constant-sized input array spans its default size, so the next input does not overlap it."""
    code = _translate(tmp_path, _GLSL_INPUT_ARRAY, ".vert", backend)

    if backend == "opengl":
        assert "layout(location = 0) in vec4 weights[count];" in code
        assert "layout(location = 3) in vec2 uv;" in code
    elif backend == "vulkan":
        locations = _spirv_locations(code)
        assert _named(locations, "CrossGL_vertex_input_", "weights") == 0
        assert _named(locations, "CrossGL_vertex_input_", "uv") == 3
    elif backend == "wgsl":
        assert "@location(2) weights_2: vec4<f32>" in code
        assert "@location(3) uv: vec2<f32>" in code
    else:
        assert "float4 weights_2 [[attribute(2)]];" in code
        assert "float2 uv [[attribute(3)]];" in code
    _validate(tmp_path, backend, code, glsl_stages=("vert",))
