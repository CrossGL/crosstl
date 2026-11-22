"""Slang output binds the multiview, shading-rate, stencil and intrinsic-read fragment built-ins.

``gl_ViewIndex`` is ``SV_ViewID``, ``gl_ShadingRateEXT`` and ``gl_PrimitiveShadingRateEXT`` are ``SV_ShadingRate``,
an ``int`` ``gl_FragStencilRefEXT`` (GLSL's ``gl_FragStencilRefARB``) is ``SV_StencilRef``, and a fragment input structure's ``gl_HelperInvocation`` and
subgroup members are read from ``IsHelperLane()`` and the ``WaveGetLane*`` functions in the entry body. Each is
refused in a stage, direction or type that lacks it. When slangc is installed, the SPIR-V it writes is checked for
the built-in each declaration means.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl.translator
from crosstl.translator.codegen.slang_codegen import SlangCodeGen
from crosstl.translator.codegen.slang_refusals import SlangLoweringRefused
from crosstl.translator.refusals import SlangRefusal

VIEW_AND_RATE = """
shader Probe {
    struct FragmentInput {
        vec2 uv;
        int view @ gl_ViewIndex;
        int rate @ gl_ShadingRateEXT;
    }
    struct FragmentOutput {
        vec4 color;
        int stencil @ gl_FragStencilRefEXT;
    }
    fragment {
        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            output.color = vec4(input.uv, float(input.view), float(input.rate));
            output.stencil = int(input.uv.x * 255.0);
            return output;
        }
    }
}
"""

PRIMITIVE_RATE = """
shader Probe {
    struct VertexOutput {
        vec4 position @ gl_Position;
        int rate @ gl_PrimitiveShadingRateEXT;
    }
    vertex {
        VertexOutput main(vec3 position @ POSITION, int view @ gl_ViewIndex) {
            VertexOutput output;
            output.position = vec4(position, float(view));
            output.rate = 5;
            return output;
        }
    }
}
"""

INTRINSIC_MEMBERS = """
shader Probe {
    struct FragmentInput {
        vec2 uv;
        bool helper @ gl_HelperInvocation;
        uint lanes @ gl_SubgroupSize;
        uvec4 below @ gl_SubgroupLtMask;
    }
    vec4 shade(FragmentInput input) {
        return vec4(input.uv, float(input.lanes + input.below.x), input.helper ? 1.0 : 0.0);
    }
    fragment {
        vec4 main(FragmentInput input) @ gl_FragColor {
            return shade(input);
        }
    }
}
"""


def generate(code):
    return SlangCodeGen().generate(crosstl.translator.parse(code))


def _compact(text):
    return " ".join(text.split())


def _spirv_assembly(tmp_path, slang, stage):
    slangc = shutil.which("slangc")
    if slangc is None:
        pytest.skip("slangc is not installed")
    source = tmp_path / f"{stage}.slang"
    source.write_text(slang, encoding="utf-8")
    output = tmp_path / f"{stage}.spvasm"
    completed = subprocess.run(
        [
            slangc,
            str(source),
            "-entry",
            "main",
            "-target",
            "spirv-asm",
            "-o",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return output.read_text(encoding="utf-8")


def test_slang_writes_view_shading_rate_and_int_stencil_system_values(tmp_path):
    slang = generate(VIEW_AND_RATE)

    assert "int view : SV_ViewID;" in slang
    assert "int rate : SV_ShadingRate;" in slang
    assert "int stencil : SV_StencilRef;" in slang
    assert "gl_" not in slang

    assembly = _spirv_assembly(tmp_path, slang, "fragment")
    for builtin in ("ViewIndex", "ShadingRateKHR", "FragStencilRefEXT"):
        assert f"BuiltIn {builtin}" in assembly


def test_slang_writes_primitive_shading_rate_output(tmp_path):
    slang = generate(PRIMITIVE_RATE)

    assert "int rate : SV_ShadingRate;" in slang
    assert "int view : SV_ViewID" in slang

    assembly = _spirv_assembly(tmp_path, slang, "vertex")
    assert "BuiltIn PrimitiveShadingRateKHR" in assembly
    assert "BuiltIn ViewIndex" in assembly


def test_slang_reads_intrinsic_members_of_a_fragment_input_in_the_entry(tmp_path):
    slang = _compact(generate(INTRINSIC_MEMBERS))

    # The program's structure is an ordinary value type; the entry takes the remaining members as its signature.
    assert (
        "struct FragmentInput { float2 uv; bool helper; uint lanes; uint4 below; };"
        in slang
    )
    assert "struct FragmentInputSignature { float2 uv : TEXCOORD0; };" in slang
    assert "input.helper = IsHelperLane();" in slang
    assert "input.lanes = WaveGetLaneCount();" in slang
    assert "input.below = WaveGetLaneLtMask();" in slang
    assert "gl_" not in slang

    assembly = _spirv_assembly(tmp_path, generate(INTRINSIC_MEMBERS), "fragment")
    assert "BuiltIn SubgroupSize" in assembly
    assert "BuiltIn SubgroupLtMask" in assembly


def test_hlsl_output_of_intrinsic_members_is_unchanged():
    # The shared pass takes Slang's table as an argument; HLSL keeps its own default, which also covers
    # gl_NumSamples (GetRenderTargetSampleCount()), a built-in Slang's table leaves out.
    from crosstl.translator.codegen import hlsl_builtin_input_members as shared
    from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
    from crosstl.translator.codegen.slang_builtin_input_members import (
        SLANG_INTRINSIC_INPUT_BUILTINS,
    )

    assert shared.INTRINSIC_INPUT_BUILTINS == frozenset(shared._BUILTIN_TYPES)
    assert "gl_NumSamples" in shared.INTRINSIC_INPUT_BUILTINS
    assert "gl_NumSamples" not in SLANG_INTRINSIC_INPUT_BUILTINS

    source = INTRINSIC_MEMBERS.replace(
        "uvec4 below @ gl_SubgroupLtMask;",
        "uvec4 below @ gl_SubgroupLtMask;\n        int samples @ gl_NumSamples;",
    ).replace("input.below.x)", "input.below.x + uint(input.samples))")
    default = shared.lower_hlsl_builtin_input_members(crosstl.translator.parse(source))
    explicit = shared.lower_hlsl_builtin_input_members(
        crosstl.translator.parse(source), shared._BUILTIN_TYPES
    )
    assert HLSLCodeGen().generate(default) == HLSLCodeGen().generate(explicit)

    hlsl = _compact(HLSLCodeGen().generate(crosstl.translator.parse(source)))
    assert "input.helper = IsHelperLane();" in hlsl
    assert "input.samples = int(GetRenderTargetSampleCount());" in hlsl
    assert "input.lanes = WaveGetLaneCount();" in hlsl
    assert re.search(r"\bgl_[A-Z]", hlsl) is None


@pytest.mark.parametrize(
    ("code", "message"),
    [
        (
            PRIMITIVE_RATE.replace(
                "int rate @ gl_PrimitiveShadingRateEXT;",
                "int rate @ gl_PrimitiveShadingRateEXT;\n        int view @ gl_ViewIndex;",
            ),
            "input-only builtin semantics cannot be used as outputs",
        ),
        (
            VIEW_AND_RATE.replace("gl_ShadingRateEXT", "gl_PrimitiveShadingRateEXT"),
            "output-only builtin semantics cannot be used as inputs",
        ),
        (
            VIEW_AND_RATE.replace("int rate @", "float rate @").replace(
                "float(input.rate)", "input.rate"
            ),
            "expected int or uint type",
        ),
        (
            """
            shader Probe {
                struct VertexInput {
                    vec3 position @ POSITION;
                    int rate @ gl_ShadingRateEXT;
                }
                vertex {
                    vec4 main(VertexInput input) @ gl_Position {
                        return vec4(input.position, float(input.rate));
                    }
                }
            }
            """,
            "valid stage is fragment",
        ),
        (
            """
            shader Probe {
                compute {
                    layout(local_size_x = 1) in;
                    buffer int* values;
                    void main(int view @ gl_ViewIndex) { values[0] = view; }
                }
            }
            """,
            "valid stage is domain, fragment, geometry, hull, vertex",
        ),
        (
            """
            shader Probe {
                struct FragmentOutput {
                    vec4 color;
                    int rate @ gl_PrimitiveShadingRateEXT;
                }
                fragment {
                    FragmentOutput main() {
                        FragmentOutput output;
                        output.color = vec4(1.0);
                        output.rate = 0;
                        return output;
                    }
                }
            }
            """,
            "valid stage is geometry, mesh, vertex",
        ),
        (
            VIEW_AND_RATE.replace("int stencil @", "float stencil @").replace(
                "int(input.uv.x * 255.0)", "input.uv.x"
            ),
            "expected uint type",
        ),
    ],
)
def test_slang_refuses_view_shading_rate_and_stencil_misuse(code, message):
    with pytest.raises((SlangRefusal, SlangLoweringRefused), match=message):
        generate(code)
