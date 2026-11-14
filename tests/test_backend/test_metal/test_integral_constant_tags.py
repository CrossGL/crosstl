"""Concrete Metal library integral constants used as tag types.

``metal::bool_constant<true>`` and friends have no CrossGL spelling. After
template materialization a concrete tag became the undeclared type
``bool_constant<true_>``. Concrete tags are now synthesized
structs with the same ``static constexpr`` ``value``.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.Metal.auto_placeholders import deduced_auto_type
from crosstl.backend.Metal.integral_constant_tags import (
    concretize_integral_constant_tags,
)
from crosstl.backend.Metal.preprocessor import MetalPreprocessor


def _concretize(code):
    return concretize_integral_constant_tags(MetalPreprocessor(), code)


def test_concrete_tags_become_one_struct_per_constant():
    code = (
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "float f(metal::bool_constant<true>, metal::true_type, bool_constant<1>);\n"
        "float g(metal::integral_constant<int, -3>, metal::integral_constant<uint, 4>);\n"
        "float h(metal::bool_constant<(4 > 2)>, metal::false_type);\n"
    )

    output = _concretize(code)

    assert output.count("struct metal_integral_constant_bool_true {") == 1
    assert "static constexpr constant bool value = true;" in output
    assert "static constexpr constant int value = -3;" in output
    assert "static constexpr constant uint value = 4u;" in output
    assert (
        "float f(metal_integral_constant_bool_true, metal_integral_constant_bool_true,"
        " metal_integral_constant_bool_true);" in output
    )
    assert (
        "float g(metal_integral_constant_int_m3, metal_integral_constant_uint_4);"
        in output
    )
    assert (
        "float h(metal_integral_constant_bool_true, metal_integral_constant_bool_false);"
        in output
    )
    # The definitions follow `using namespace metal;`, before any use.
    assert output.index("struct metal_integral_constant") > output.index(
        "using namespace metal;"
    )
    assert output.index("struct metal_integral_constant") < output.index("float f(")


@pytest.mark.parametrize(
    "code",
    [
        # Dependent arguments stay for template deduction.
        "using namespace metal;\ntemplate <bool B> float f(metal::bool_constant<B>);\n",
        "using namespace metal;\ntemplate <int N> float f(metal::integral_constant<int, N>);\n",
        # A base clause is read by the value-trait evaluator.
        "using namespace metal;\nstruct is_x : metal::true_type {};\n",
        "using namespace metal;\nstruct is_y : public metal::bool_constant<false> {};\n",
        # Unqualified names refer to library tags only through `using namespace metal`.
        "float f(bool_constant<true>);\n",
        # A user's own integral_constant shadows the unqualified spelling.
        "using namespace metal;\n"
        "template <typename T, T v> struct integral_constant { static constexpr constant T value = v; };\n"
        "float f(integral_constant<int, 2>);\n",
        # Not an integral value type, and comments or strings.
        "using namespace metal;\nfloat f(metal::integral_constant<float, 1>);\n",
        'using namespace metal;\n// metal::bool_constant<true>\nconstant char* s = "metal::true_type";\n',
    ],
)
def test_non_concrete_or_shadowed_tags_are_left_alone(code):
    assert _concretize(code) == code


def test_qualified_tag_is_rewritten_beside_a_user_integral_constant():
    code = (
        "using namespace metal;\n"
        "template <typename T, T v> struct integral_constant { static constexpr constant T value = v; };\n"
        "float f(integral_constant<int, 2>, metal::integral_constant<int, 2>);\n"
    )

    output = _concretize(code)

    assert (
        "float f(integral_constant<int, 2>, metal_integral_constant_int_2);" in output
    )


@pytest.mark.parametrize(
    "spelling, expected",
    [
        ("auto", "metal::bool_constant<true>"),
        ("constexpr auto", "const metal::bool_constant<true>"),
        ("const auto", "const metal::bool_constant<true>"),
        ("static constexpr auto", "const metal::bool_constant<true>"),
        ("auto const", "const metal::bool_constant<true>"),
        ("constexpr auto&", "auto"),
    ],
)
def test_const_and_constexpr_auto_locals_take_the_initializer_type(spelling, expected):
    assert deduced_auto_type(spelling, "metal::bool_constant<true>") == expected


def test_const_auto_pointer_locals_are_unchanged():
    assert deduced_auto_type("const auto*", "device float*") == "const device float*"
    assert deduced_auto_type("constexpr auto", "device float*") == "auto"


TAG_TEMPLATES = """
#include <metal_stdlib>
using namespace metal;

template <bool transpose_a>
float pick(const device float* src, metal::bool_constant<transpose_a>) {
  return transpose_a ? src[1] : src[0];
}

template <bool t>
float outer(const device float* src) {
  constexpr auto ta = metal::bool_constant<t>{};
  return pick(src, ta);
}

kernel void k(const device float* src [[buffer(0)]],
              device float* out [[buffer(1)]],
              uint tid [[thread_position_in_grid]]) {
  out[tid] = pick(src, metal::bool_constant<true>{}) + outer<false>(src) * 10.0f;
}
"""


def _translate(tmp_path, code, backend):
    source = tmp_path / "tags.metal"
    source.write_text(code)
    return crosstl.translate(
        str(source), backend=backend, source_backend="metal", format_output=False
    )


def _body(output, name):
    definition = list(re.finditer(rf"\b{name}\w*\([^)]*\)\s*\{{", output))[-1]
    return output[definition.end() : output.find("\n}", definition.end())]


@pytest.mark.parametrize("backend", ["directx", "opengl"])
def test_tag_parameters_specialize_through_temporaries_and_constexpr_locals(
    tmp_path, backend
):
    output = _translate(tmp_path, TAG_TEMPLATES, backend)

    assert not re.search(r"\bbool_constant\b|\btrue_\b", output)
    assert "struct metal_integral_constant_bool_true" in output
    assert "struct metal_integral_constant_bool_false" in output
    assert "true ?" in _body(output, "pick_true")
    assert "false ?" in _body(output, "pick_false")
    assert re.search(r"pick_false\w*\(", _body(output, "outer_false"))
    glslang = shutil.which("glslangValidator")
    if backend == "opengl" and glslang is not None:
        shader = tmp_path / "tags.comp"
        shader.write_text(output)
        result = subprocess.run(
            [glslang, "-S", "comp", str(shader)], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_hand_written_concrete_tag_parameter(tmp_path):
    code = """
    #include <metal_stdlib>
    using namespace metal;
    float pick_true(const device float* src, metal::bool_constant<true>) {
      return src[1];
    }
    kernel void k(const device float* src [[buffer(0)]],
                  device float* out [[buffer(1)]],
                  uint tid [[thread_position_in_grid]]) {
      out[tid] = pick_true(src, metal::bool_constant<true>{});
    }
    """

    for backend in ("directx", "metal"):
        output = _translate(tmp_path, code, backend)
        assert "metal_integral_constant_bool_true" in output
        assert not re.search(r"\btrue_\b", output)
