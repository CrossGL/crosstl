"""SPIR-V output orders rasterizer-ordered resources with SPV_EXT_fragment_shader_interlock."""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.spirv_refusals import SpirvLoweringRefused

BUFFER = """shader Probe {
    RWStructuredBuffer<vec4> accum @rasterizer_ordered;
    struct FragmentInput {
        vec4 coord @ gl_FragCoord;
        vec2 uv;
    }
    struct FragmentOutput {
        vec4 color;
    }
    fragment {
        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            uint k = uint(input.coord.y) * 64u + uint(input.coord.x);
            accum[k] = accum[k] + vec4(input.uv, 0.0, 1.0);
            output.color = vec4(input.uv, 0.0, 1.0);
            return output;
        }
    }
}
"""

IMAGE = """shader Probe {
    image2D accum @rasterizer_ordered @rgba32f;
    struct FragmentInput {
        vec4 coord @ gl_FragCoord;
        vec2 uv;
    }
    struct FragmentOutput {
        vec4 color;
    }
    fragment {
        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            ivec2 p = ivec2(input.coord.xy);
            imageStore(accum, p, imageLoad(accum, p) + vec4(input.uv, 0.0, 1.0));
            output.color = vec4(input.uv, 0.0, 1.0);
            return output;
        }
    }
}
"""

UNORDERED = BUFFER.replace(" @rasterizer_ordered", "")

EARLY_RETURN = """shader Probe {
    RWStructuredBuffer<vec4> accum @rasterizer_ordered;
    fragment {
        vec4 coord @input @gl_FragCoord;
        vec4 color @output @location(0);
        void main() {
            color = vec4(1.0);
            if (coord.x < 1.0) {
                return;
            }
            uint k = uint(coord.x);
            accum[k] = accum[k] + vec4(1.0);
        }
    }
}
"""

HLSL_SOURCE = """RasterizerOrderedBuffer<float4> accum : register(u1);
float4 main(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target {
    uint k = uint(pos.y) * 64u + uint(pos.x);
    accum[k] = accum[k] + float4(uv, 0.0, 1.0);
    return float4(uv, 0.0, 1.0);
}
"""

GLSL_EXPLICIT = """#version 450
#extension GL_ARB_fragment_shader_interlock : require
layout(LAYOUT) in;
layout(std430, binding = 0) buffer Accum { vec4 data[]; } accum;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
void main() {
    uint k = uint(gl_FragCoord.y) * 64u + uint(gl_FragCoord.x);
    color = vec4(uv, 0.0, 1.0);
    beginInvocationInterlockARB();
    accum.data[k] += vec4(uv, 0.0, 1.0);
    endInvocationInterlockARB();
}
"""

EXPLICIT_LAYOUTS = {
    "pixel_interlock_ordered": (
        "PixelInterlockOrderedEXT",
        "FragmentShaderPixelInterlockEXT",
    ),
    "pixel_interlock_unordered": (
        "PixelInterlockUnorderedEXT",
        "FragmentShaderPixelInterlockEXT",
    ),
    "sample_interlock_ordered": (
        "SampleInterlockOrderedEXT",
        "FragmentShaderSampleInterlockEXT",
    ),
    "sample_interlock_unordered": (
        "SampleInterlockUnorderedEXT",
        "FragmentShaderSampleInterlockEXT",
    ),
}
INTERLOCK_MODES = {mode for mode, _capability in EXPLICIT_LAYOUTS.values()}


def _translate(tmp_path, source, name="shader.cgl"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend="vulkan", format_output=False)


def _interlock_modes(code):
    return [
        mode
        for mode in re.findall(r"OpExecutionMode %\w+ (\w+)", code)
        if mode in INTERLOCK_MODES
    ]


def _instructions(code):
    return [line.strip() for line in code.splitlines() if line.strip()]


def _assert_whole_entry_section(code):
    lines = _instructions(code)
    assert lines.count("OpBeginInvocationInterlockEXT") == 1
    assert lines.count("OpEndInvocationInterlockEXT") == 1
    begin = lines.index("OpBeginInvocationInterlockEXT")
    end = lines.index("OpEndInvocationInterlockEXT")
    assert begin < end
    assert lines[end + 1] == "OpReturn"
    # Every access to the resource sits inside the section, and every output is written before it ends.
    body = lines[begin:end]
    assert any(
        re.search(r"\bOp(Load|Store|ImageRead|ImageWrite)\b", line) for line in body
    )
    assert not any(
        re.search(r"\bOp(ImageRead|ImageWrite)\b", line) for line in lines[end:]
    )


@pytest.mark.parametrize(
    "source, name",
    [(BUFFER, "shader.cgl"), (IMAGE, "shader.cgl"), (HLSL_SOURCE, "shader.hlsl")],
    ids=["buffer", "image", "hlsl-rov"],
)
def test_rasterizer_ordered_resource_holds_the_entry_section(tmp_path, source, name):
    code = _translate(tmp_path, source, name)
    assert "OpCapability FragmentShaderPixelInterlockEXT" in code
    assert 'OpExtension "SPV_EXT_fragment_shader_interlock"' in code
    assert _interlock_modes(code) == ["PixelInterlockOrderedEXT"]
    _assert_whole_entry_section(code)


def test_unordered_resource_takes_no_interlock(tmp_path):
    code = _translate(tmp_path, UNORDERED)
    assert "Interlock" not in code
    assert "SPV_EXT_fragment_shader_interlock" not in code


def test_early_return_is_refused(tmp_path):
    with pytest.raises(SpirvLoweringRefused, match="returns early"):
        _translate(tmp_path, EARLY_RETURN)


TWO_LAYOUTS = """shader Probe {
    RWStructuredBuffer<vec4> accum;
    fragment {
        layout(pixel_interlock_ordered) in;
        layout(sample_interlock_ordered) in;
        vec4 coord @input @gl_FragCoord;
        vec4 color @output @location(0);
        void main() {
            uint k = uint(coord.x);
            beginInvocationInterlockARB();
            accum[k] = accum[k] + vec4(1.0);
            endInvocationInterlockARB();
            color = vec4(1.0);
        }
    }
}
"""


def test_two_interlock_layouts_are_refused(tmp_path):
    with pytest.raises(SpirvLoweringRefused, match="one interlock execution mode"):
        _translate(tmp_path, TWO_LAYOUTS)


@pytest.mark.parametrize("layout", sorted(EXPLICIT_LAYOUTS))
def test_explicit_interlock_calls_keep_their_section(tmp_path, layout):
    mode, capability = EXPLICIT_LAYOUTS[layout]
    code = _translate(tmp_path, GLSL_EXPLICIT.replace("LAYOUT", layout), "shader.frag")
    assert f"OpCapability {capability}" in code
    assert _interlock_modes(code) == [mode]
    lines = _instructions(code)
    # The program's own section is kept as written: no second section is wrapped around the entry point.
    assert lines.count("OpBeginInvocationInterlockEXT") == 1
    assert lines.count("OpEndInvocationInterlockEXT") == 1
    begin = lines.index("OpBeginInvocationInterlockEXT")
    end = lines.index("OpEndInvocationInterlockEXT")
    assert begin < end
    assert any(line.startswith("OpStore ") for line in lines[begin:end])
    # The colour output is written after the program's section ends, so the generator added no section of its own.
    assert any(line.startswith("OpStore ") for line in lines[end:])


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "source, name",
    [
        (BUFFER, "shader.cgl"),
        (IMAGE, "shader.cgl"),
        (HLSL_SOURCE, "shader.hlsl"),
        (GLSL_EXPLICIT.replace("LAYOUT", "sample_interlock_ordered"), "shader.frag"),
    ],
    ids=["buffer", "image", "hlsl-rov", "explicit"],
)
def test_interlock_validates_for_vulkan(tmp_path, source, name):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("spirv-as and spirv-val are not on PATH")
    assembly = tmp_path / "shader.spvasm"
    binary = tmp_path / "shader.spv"
    assembly.write_text(_translate(tmp_path, source, name))
    _run(["spirv-as", "--target-env", "vulkan1.1", str(assembly), "-o", str(binary)])
    _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])
