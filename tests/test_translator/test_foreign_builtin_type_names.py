"""A built-in type of another language is mapped to the target's spelling or refused, never copied unchanged.

The importers carry the built-in types of the languages they read into the IR as plain names. A generator that had
no mapping for one used to write it unchanged, so GLSL, HLSL and Metal output declared ``Float16 v;``, ``__half v;``
or ``float32_t v;``, which the target compilers reject.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.lowering_refusals import LoweringRefused
from crosstl.translator.codegen.written_type_names import (
    sized_float_spelling,
    sized_vector_parts,
)
from crosstl.translator.semantics.builtin_type_names import BUILTIN_TYPE_NAMES

TARGETS = ["opengl", "metal", "directx"]
LABELS = {"opengl": "OpenGL", "metal": "Metal", "directx": "DirectX"}


def _compute(declaration, helpers=""):
    return f"""
shader S {{
    {helpers}
    compute {{
        buffer float values[];
        @compute @workgroup_size(1, 1, 1)
        void main() {{
            {declaration}
            values[0] = 1.0;
        }}
    }}
}}
"""


def _translate(tmp_path, source, target):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=target, format_output=False)


# Refusals -----------------------------------------------------------------------------------------------------------
FOREIGN = {
    "opengl": ["Float16", "__half", "dim3", "HitObject", "int16_t2x2"],
    "metal": ["Float16", "__half", "dim3", "HitObject", "int16_t2x2"],
    "directx": ["Float16", "__half", "dim3", "HitObject", "atomic_uint"],
}


@pytest.mark.parametrize(
    "target,name",
    [(target, name) for target, names in FOREIGN.items() for name in names],
)
def test_a_foreign_built_in_type_is_refused(tmp_path, target, name):
    with pytest.raises(LoweringRefused) as raised:
        _translate(tmp_path, _compute(f"{name} v;"), target)
    assert f"the built-in type '{name}' has no {LABELS[target]} spelling" in str(
        raised.value
    )


def test_the_foreign_names_under_test_are_built_in_types():
    for names in FOREIGN.values():
        for name in names:
            assert name in BUILTIN_TYPE_NAMES


@pytest.mark.parametrize("target", TARGETS)
def test_a_program_type_named_like_a_foreign_built_in_is_written_unchanged(
    tmp_path, target
):
    output = _translate(
        tmp_path,
        _compute("dim3 v; v.x = 1;", helpers="struct dim3 { int x; };"),
        target,
    )
    assert "dim3 v" in output


# Sized floating-point spellings --------------------------------------------------------------------------------------
SIZED = {
    "opengl": {
        "float32_t": "float v",
        "float64_t": "double v",
        "float32_t4": "vec4 v",
        "float64_t2": "dvec2 v",
        "float32_t3x3": "mat3 v",
        "int16_t2": "ivec2 v",
        "uint64_t3": "u64vec3 v",
    },
    "metal": {
        "float32_t": "float v",
        "float32_t3x3": "float3x3 v",
        "float16_t2x2": "half2x2 v",
    },
}


@pytest.mark.parametrize(
    "target,name,expected",
    [
        (target, name, expected)
        for target, cases in SIZED.items()
        for name, expected in cases.items()
    ],
)
def test_a_sized_spelling_is_written_in_the_targets_spelling(
    tmp_path, target, name, expected
):
    output = _translate(tmp_path, _compute(f"{name} v;"), target)
    assert expected in output
    assert f"{name} v" not in output


def test_metal_refuses_a_sized_double_with_the_double_reason(tmp_path):
    with pytest.raises(LoweringRefused) as raised:
        _translate(tmp_path, _compute("float64_t2 v;"), "metal")
    reasons = raised.value.reasons
    assert any("double-precision value" in reason for reason in reasons)
    assert not any("built-in type" in reason for reason in reasons)


def test_sized_spelling_helpers():
    assert sized_float_spelling("float16_t") == "half"
    assert sized_float_spelling("float32_t4") == "float4"
    assert sized_float_spelling("float64_t2x3") == "double2x3"
    assert sized_float_spelling("float32_t1") == "float32_t1"
    assert sized_float_spelling("Float16") == "Float16"
    assert sized_vector_parts("int16_t2") == ("int16_t", 2)
    assert sized_vector_parts("bfloat16_t4") == ("bfloat16_t", 4)
    assert sized_vector_parts("int16_t") is None
    assert sized_vector_parts("int16_t2x2") is None


# Native validation -------------------------------------------------------------------------------------------------
def _glslang_accepts(tmp_path, code):
    path = tmp_path / "shader.comp"
    path.write_text(code)
    result = subprocess.run(
        ["glslangValidator", str(path)], capture_output=True, text=True, check=False
    )
    return result.returncode == 0, result.stdout[-400:]


@pytest.mark.skipif(
    shutil.which("glslangValidator") is None, reason="glslangValidator not installed"
)
@pytest.mark.parametrize("name", sorted(SIZED["opengl"]))
def test_glslang_accepts_the_sized_spellings(tmp_path, name):
    output = _translate(tmp_path, _compute(f"{name} v;"), "opengl")
    ok, log = _glslang_accepts(tmp_path, output)
    assert ok, log


@pytest.mark.skipif(shutil.which("xcrun") is None, reason="xcrun not installed")
@pytest.mark.parametrize("name", sorted(SIZED["metal"]))
def test_metal_compiler_accepts_the_sized_spellings(tmp_path, name):
    output = _translate(tmp_path, _compute(f"{name} v;"), "metal")
    path = tmp_path / "shader.metal"
    path.write_text(output)
    result = subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.2",
            "-c",
            str(path),
            "-o",
            str(tmp_path / "s.air"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if "unable to find utility" in result.stderr:
        pytest.skip("Metal toolchain not installed")
    assert result.returncode == 0, result.stderr[-400:]
