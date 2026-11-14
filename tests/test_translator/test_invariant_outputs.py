"""``@invariant`` outputs: built-in positions keep their invariance on every graphics target."""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.glsl_refusals import OpenGLLoweringRefused
from crosstl.translator.codegen.metal_codegen import UnsupportedMetalFeatureError

MEMBER = """shader Probe {
    struct VertexInput {
        vec4 position;
    }
    struct VertexOutput {
        vec4 position ATTRS;
    }
    vertex {
        VertexOutput main(VertexInput input) {
            VertexOutput output;
            output.position = input.position * 2.0;
            return output;
        }
    }
}
"""

STAGE_VARIABLE = """shader Probe {
    vertex {
        vec4 position @input @location(0);
        vec4 gl_Position @output ATTRS;
        void main() {
            gl_Position = position * 2.0;
        }
    }
}
"""

RETURN_VALUE = """shader Probe {
    vertex {
        vec4 main(vec4 position @ location(0)) ATTRS {
            return position * 2.0;
        }
    }
}
"""

ORDERS = ["@ gl_Position @ invariant", "@ invariant @ gl_Position"]
FORMS = {"member": MEMBER, "stage-variable": STAGE_VARIABLE, "return": RETURN_VALUE}


def _translate(tmp_path, source, backend, name="shader.cgl"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _spirv_position_is_invariant(code):
    """Whether the ``Position`` built-in variable is decorated ``Invariant``, exactly once.

    Unformatted SPIR-V names its ids by number, so the variable is found through its ``BuiltIn`` decoration.
    """
    builtins = re.findall(r"OpDecorate %(\w+) BuiltIn Position\b", code)
    assert len(builtins) == 1, code
    variable = re.escape(builtins[0])
    assert not re.search(rf"OpDecorate %{variable} Location\b", code)
    return len(re.findall(rf"OpDecorate %{variable} Invariant\b", code)) == 1


def _cases():
    return [
        pytest.param(form, order, id=f"{form}-{index}")
        for form in FORMS
        for index, order in enumerate(ORDERS)
    ]


@pytest.mark.parametrize("form, order", _cases())
@pytest.mark.parametrize("backend", ["opengl", "webgl"])
def test_glsl_redeclares_the_position_invariant(tmp_path, backend, form, order):
    code = _translate(tmp_path, FORMS[form].replace("ATTRS", order), backend)
    assert code.count("invariant gl_Position;") == 1
    assert "gl_Position = " in code


@pytest.mark.parametrize("form, order", _cases())
def test_metal_marks_the_position_invariant(tmp_path, form, order):
    code = _translate(tmp_path, FORMS[form].replace("ATTRS", order), "metal")
    assert "[[position]] [[invariant]]" in code
    assert "user(" not in code
    assert "__crossgl_position" not in code


@pytest.mark.parametrize("form, order", _cases())
def test_spirv_decorates_the_position_builtin_invariant(tmp_path, form, order):
    code = _translate(tmp_path, FORMS[form].replace("ATTRS", order), "vulkan")
    assert _spirv_position_is_invariant(code)


@pytest.mark.parametrize("form, order", _cases())
def test_hlsl_keeps_the_position_semantic_whatever_the_qualifier_order(
    tmp_path, form, order
):
    code = _translate(tmp_path, FORMS[form].replace("ATTRS", order), "directx")
    assert "SV_POSITION" in code
    assert "invariant" not in code.lower()


def test_glsl_keeps_invariant_user_outputs_and_fragment_depth(tmp_path):
    source = """shader Probe {
    struct VertexOutput {
        vec4 position @ gl_Position;
        vec4 color @ invariant @ location(0);
    }
    struct FragmentInput {
        vec4 color @ location(0);
    }
    struct FragmentOutput {
        vec4 color @ gl_FragColor;
        float depth @ invariant @ gl_FragDepth;
    }
    vertex {
        VertexOutput main() {
            VertexOutput output;
            output.position = vec4(0.0);
            output.color = vec4(1.0);
            return output;
        }
    }
    fragment {
        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            output.color = input.color;
            output.depth = 0.5;
            return output;
        }
    }
}
"""
    code = _translate(tmp_path, source, "opengl")
    assert "invariant out vec4 color;" in code
    assert "invariant gl_FragDepth;" in code
    assert "invariant gl_Position;" not in code


def test_metal_refuses_invariance_it_cannot_spell(tmp_path):
    source = MEMBER.replace(
        "vec4 position ATTRS;",
        "vec4 position @ gl_Position;\n        vec4 tint @ invariant;",
    ).replace("return output;", "output.tint = vec4(1.0);\n            return output;")
    with pytest.raises(UnsupportedMetalFeatureError, match="invariant"):
        _translate(tmp_path, source, "metal")


def test_glsl_refuses_an_invariant_builtin_without_a_redeclaration(tmp_path):
    source = """shader Probe {
    compute {
        uint index @ input @ invariant @ gl_LocalInvocationIndex;
        void main() @ numthreads(1, 1, 1) {
        }
    }
}
"""
    with pytest.raises(OpenGLLoweringRefused, match="invariant"):
        _translate(tmp_path, source, "opengl")


GLSL_SOURCE = """#version 450 core
layout(location = 0) in vec4 position;
invariant gl_Position;
void main() { gl_Position = position * 2.0; }
"""

METAL_SOURCE = """#include <metal_stdlib>
using namespace metal;
struct VOut { float4 position [[position, invariant]]; };
vertex VOut vmain(uint vid [[vertex_id]]) {
    VOut o;
    o.position = float4(float(vid));
    return o;
}
"""


def _is_invariant(backend, code):
    if backend == "vulkan":
        return _spirv_position_is_invariant(code)
    if backend == "metal":
        return "[[position]] [[invariant]]" in code
    return "invariant gl_Position;" in code


@pytest.mark.parametrize("backend", ["metal", "vulkan", "opengl"])
def test_glsl_source_invariance_survives_translation(tmp_path, backend):
    code = _translate(tmp_path, GLSL_SOURCE, backend, "shader.vert")
    assert _is_invariant(backend, code)


@pytest.mark.parametrize("backend", ["opengl", "vulkan", "metal"])
def test_metal_source_invariance_survives_translation(tmp_path, backend):
    code = _translate(tmp_path, METAL_SOURCE, backend, "shader.metal")
    assert _is_invariant(backend, code)


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("form", list(FORMS))
@pytest.mark.parametrize("backend", ["opengl", "webgl"])
def test_glsl_output_compiles(tmp_path, backend, form):
    if not shutil.which("glslangValidator"):
        pytest.skip("glslangValidator is not on PATH")
    code = _translate(tmp_path, FORMS[form].replace("ATTRS", ORDERS[1]), backend)
    path = tmp_path / "shader.vert"
    path.write_text(code)
    _run(["glslangValidator", "-S", "vert", str(path)])


@pytest.mark.parametrize("form", list(FORMS))
def test_spirv_output_validates(tmp_path, form):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("spirv-as and spirv-val are not on PATH")
    code = _translate(tmp_path, FORMS[form].replace("ATTRS", ORDERS[1]), "vulkan")
    source = tmp_path / "shader.spvasm"
    binary = tmp_path / "shader.spv"
    source.write_text(code)
    _run(["spirv-as", "--target-env", "vulkan1.1", str(source), "-o", str(binary)])
    _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])


@pytest.mark.parametrize("form", list(FORMS))
def test_metal_output_compiles(tmp_path, form):
    if not shutil.which("xcrun"):
        pytest.skip("xcrun is not on PATH")
    probe = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        pytest.skip("the Metal compiler is not installed")
    code = _translate(tmp_path, FORMS[form].replace("ATTRS", ORDERS[1]), "metal")
    path = tmp_path / "shader.metal"
    path.write_text(code)
    _run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-fpreserve-invariance",
            "-c",
            str(path),
            "-o",
            str(tmp_path / "shader.air"),
        ]
    )
