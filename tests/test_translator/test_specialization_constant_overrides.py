"""Specialization constants in WGSL and Slang output: ``@constant_id`` keeps its id instead of being frozen."""

import os
import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.specialization_constants import (
    SpecializationConstantError,
    specialization_constant_ids,
    specialization_dependent_constants,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

PROGRAM = """shader Probe {
    PRELUDE
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        SHARED
        void main() {
            uint i = gl_GlobalInvocationID.x;
            BODY
        }
    }
}
"""

SCALE = "const int scale @constant_id(0) = 3;"
SCALE_AND_NEGATE = (
    f"{SCALE}\n    const int doubled = scale * 2 + 1;\n"
    "    const bool negate @constant_id(5) = true;"
)
READ_SCALE = "outp[i] = negate ? -float(doubled) : float(scale);"


def _source(prelude, body, shared=""):
    return (
        PROGRAM.replace("PRELUDE", prelude)
        .replace("SHARED", shared)
        .replace("BODY", body)
    )


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _parse(source):
    return Parser(Lexer(source).get_tokens()).parse()


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr + result.stdout
    return result.stdout


def _naga(tmp_path, code):
    if shutil.which("naga"):
        shader = tmp_path / "out.wgsl"
        shader.write_text(code)
        _run(["naga", str(shader)])


def _slangc():
    configured = os.environ.get("SLANGC")
    return (
        configured
        if configured and os.path.exists(configured)
        else shutil.which("slangc")
    )


def _slang_spirv(tmp_path, code):
    """Compile Slang to SPIR-V when slangc and SPIRV-Tools are installed; return the disassembly or None."""
    slangc = _slangc()
    if not (slangc and shutil.which("spirv-dis") and shutil.which("spirv-val")):
        return None
    shader, binary = tmp_path / "out.slang", tmp_path / "out.spv"
    shader.write_text(code)
    _run(
        [
            slangc,
            str(shader),
            "-target",
            "spirv",
            "-entry",
            "main",
            "-stage",
            "compute",
            "-o",
            str(binary),
        ]
    )
    _run(["spirv-val", str(binary)])
    return _run(["spirv-dis", str(binary)])


# Target-neutral analysis.


def test_dependent_constants_follow_specialization_constants_in_order():
    ast = _parse(
        _source(
            f"{SCALE}\n    const int fixed = 4;\n    const int later = fixed + scale;",
            "outp[i] = float(later);",
        )
    )
    ids = specialization_constant_ids(ast.constants, "WGSL", 65535)
    assert ids == {"scale": 0}
    assert specialization_dependent_constants(
        ast.constants, lambda node: node.name in ids
    ) == {"scale", "later"}


def test_two_constants_with_one_id_are_refused():
    ast = _parse(
        _source(
            f"{SCALE}\n    const int other @constant_id(0) = 1;",
            "outp[i] = float(scale + other);",
        )
    )
    with pytest.raises(SpecializationConstantError) as error:
        specialization_constant_ids(ast.constants, "Slang", 2**31 - 1)
    assert error.value.kind == "duplicate-id"
    assert "'scale' and 'other' share id 0" in str(error.value)


# WGSL.


def test_wgsl_declares_specialization_constants_as_overrides_with_their_ids(
    tmp_path,
):
    code = _translate(tmp_path, _source(SCALE_AND_NEGATE, READ_SCALE), "wgsl")
    assert "@id(0)\noverride scale: i32 = 3;" in code
    assert "override doubled: i32 = ((scale * 2) + 1);" in code
    assert "@id(5)\noverride negate: bool = true;" in code
    assert not re.search(r"^const (scale|doubled|negate)\b", code, re.M)
    _naga(tmp_path, code)


def test_wgsl_keeps_an_independent_constant_as_const(tmp_path):
    code = _translate(
        tmp_path,
        _source(
            f"{SCALE}\n    const int fixed = 4;", "outp[i] = float(scale + fixed);"
        ),
        "wgsl",
    )
    assert "const fixed: i32 = 4;" in code
    assert "@id(0)\noverride scale: i32 = 3;" in code
    _naga(tmp_path, code)


def test_wgsl_declares_a_dependent_aggregate_as_a_private_variable(tmp_path):
    prelude = (
        "const float fs @constant_id(3) = 1.5;\n"
        "    const vec2 v = vec2(fs, 2.0);\n"
        "    const float w[2] = {fs, 2.0};\n"
        "    const float m = max(fs, 2.0);"
    )
    code = _translate(
        tmp_path, _source(prelude, "outp[i] = v.x + w[i % 2u] + m;"), "wgsl"
    )
    assert "@id(3)\noverride fs: f32 = 1.5;" in code
    assert "var<private> v: vec2<f32> = vec2<f32>(fs, 2.0);" in code
    assert re.search(
        r"var<private> w: array<f32, 2> = array(<f32, 2>)?\(fs, 2.0\);", code
    )
    assert "override m: f32 = max(fs, 2.0);" in code
    _naga(tmp_path, code)


def test_wgsl_sizes_a_workgroup_array_by_an_override(tmp_path):
    code = _translate(
        tmp_path,
        _source(
            SCALE,
            "tile[i] = 1.0;\n            outp[i] = tile[i];",
            shared="shared float tile[scale];",
        ),
        "wgsl",
    )
    assert "var<workgroup> tile: array<f32, scale>;" in code
    _naga(tmp_path, code)


@pytest.mark.parametrize(
    "prelude, body, kind",
    [
        (
            "const vec2 v @constant_id(0) = vec2(1.0, 2.0);",
            "outp[i] = v.x;",
            "type",
        ),
        (
            f"{SCALE}\n    const int doubled = scale * 2;",
            "float a[doubled];\n            a[i] = 1.0;\n            outp[i] = a[i];",
            "array-size",
        ),
        (
            SCALE,
            "switch (int(i)) {\n                case scale: outp[i] = 2.0; break;\n"
            "                default: break;\n            }",
            "case-selector",
        ),
        (
            "const float fs @constant_id(0) = 1.5;\n    const vec2 v = vec2(fs, 1.0);\n"
            "    const float x = v.x;",
            "outp[i] = x;",
            "dependent-initializer",
        ),
        (
            "const int big @constant_id(65536) = 1;",
            "outp[i] = float(big);",
            "id-out-of-range",
        ),
    ],
)
def test_wgsl_refuses_what_an_override_cannot_express(tmp_path, prelude, body, kind):
    with pytest.raises(SpecializationConstantError) as error:
        _translate(tmp_path, _source(prelude, body), "wgsl")
    assert error.value.kind == kind
    assert error.value.target == "WGSL"


def test_wgsl_does_not_freeze_a_nested_workgroup_array_size(tmp_path):
    with pytest.raises(SpecializationConstantError) as error:
        _translate(
            tmp_path,
            _source(
                SCALE,
                "tile[0][i] = 1.0;\n            outp[i] = tile[0][i];",
                shared="shared float tile[2][scale];",
            ),
            "wgsl",
        )
    assert error.value.kind == "array-size"


# Slang.


def test_slang_declares_specialization_constants_with_vk_constant_id(tmp_path):
    code = _translate(tmp_path, _source(SCALE_AND_NEGATE, READ_SCALE), "slang")
    assert "[vk::constant_id(0)] const int scale = 3;" in code
    assert "static const int doubled = scale * 2 + 1;" in code
    assert "[vk::constant_id(5)] const bool negate = true;" in code
    disassembly = _slang_spirv(tmp_path, code)
    if disassembly is not None:
        assert re.search(r"OpDecorate %scale SpecId 0\b", disassembly)
        assert re.search(r"OpDecorate %negate SpecId 5\b", disassembly)
        assert "OpSpecConstantOp %int IMul" in disassembly


def test_slang_sizes_an_array_by_a_folded_dependent_constant(tmp_path):
    code = _translate(
        tmp_path,
        _source(
            f"{SCALE}\n    const int doubled = scale * 2 + 1;",
            "float a[doubled];\n            a[i % 3u] = 1.0;\n            outp[i] = a[i % 2u];",
        ),
        "slang",
    )
    assert "static const int doubled = scale * 2 + 1;" in code
    disassembly = _slang_spirv(tmp_path, code)
    if disassembly is not None:
        assert re.search(r"OpTypeArray %float %\d+", disassembly)


def test_slang_evaluates_a_dependent_constant_slangc_cannot_fold(tmp_path):
    prelude = (
        f"{SCALE}\n    const float fs @constant_id(3) = 1.5;\n"
        "    const float f = float(scale) * fs;\n"
        "    const vec2 v = vec2(fs, 2.0);\n"
        "    const uint u = uint(scale) + 1u;"
    )
    code = _translate(
        tmp_path, _source(prelude, "outp[i] = f + v.y + float(u);"), "slang"
    )
    assert "[vk::constant_id(3)] const float fs = 1.5;" in code
    assert "static float f = float(scale) * fs;" in code
    assert re.search(r"^static float2 v = float2\(fs, 2\.0\);", code, re.M)
    assert re.search(r"^static const uint u = uint\(scale\) \+ 1u?;", code, re.M)
    _slang_spirv(tmp_path, code)


@pytest.mark.parametrize(
    "prelude, body, kind",
    [
        (
            "const vec2 v @constant_id(0) = vec2(1.0, 2.0);",
            "outp[i] = v.x;",
            "type",
        ),
        (
            f"{SCALE}\n    const int m = max(scale, 2);",
            "float a[m];\n            a[i] = 1.0;\n            outp[i] = a[i];",
            "array-size",
        ),
        (
            SCALE,
            "switch (int(i)) {\n                case scale: outp[i] = 2.0; break;\n"
            "                default: break;\n            }",
            "case-label",
        ),
    ],
)
def test_slang_refuses_what_slangc_cannot_specialize(tmp_path, prelude, body, kind):
    with pytest.raises(SpecializationConstantError) as error:
        _translate(tmp_path, _source(prelude, body), "slang")
    assert error.value.kind == kind
    assert error.value.target == "Slang"


_GLSL_INPUT_ARRAY = """#version 450
layout(constant_id = 3) const int count = 3;
in vec4 weights[count];
in vec2 uv;
void main() {
    gl_Position = weights[0] + weights[2] + vec4(uv, 0.0, 0.0);
}
"""


def test_slang_sizes_a_stage_input_array_by_its_default(tmp_path):
    """The interface spans the default size; slangc cannot lay out an input array sized by a constant id."""
    path = tmp_path / "shader.vert"
    path.write_text(_GLSL_INPUT_ARRAY)
    code = crosstl.translate(str(path), backend="slang", format_output=False)
    assert "[vk::constant_id(3)] const int count = 3;" in code
    assert re.search(r"float4 weights\[3\] : \w+;", code)
    slangc = _slangc()
    if slangc and shutil.which("spirv-dis") and shutil.which("spirv-val"):
        shader, binary = tmp_path / "out.slang", tmp_path / "out.spv"
        shader.write_text(code)
        _run([slangc, str(shader), "-target", "spirv", "-o", str(binary)])
        _run(["spirv-val", str(binary)])
        disassembly = _run(["spirv-dis", str(binary)])
        assert re.search(r"OpDecorate %\w*weights Location 0\b", disassembly)
        assert re.search(r"OpDecorate %\w*uv Location 3\b", disassembly)
