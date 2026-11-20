"""Operators with no class or enumeration operand skip free-operator overload ranking.

C++ [over.match.oper]/1: such an operator is the built-in one, so the importer must not rank declared free
operators for it (``builtin_operator_operands``). Ranking every candidate for each scalar or vector operand pair
was most of the importer's free-operator cost on large kernels. Class operands, and a structure whose name is
spelled like a scalar type, still resolve to their declared operators.
"""

import pytest

import crosstl
from crosstl.backend.Metal import MetalCrossGLCodeGen as importer
from crosstl.backend.Metal.builtin_operator_operands import (
    operands_have_no_class_type,
)

# `operator+` multiplies the imaginary parts, so a built-in addition cannot produce `a.imag * b.imag`.
SOURCE = """#include <metal_stdlib>
using namespace metal;

struct cplx {
  float real;
  float imag;
};

inline cplx operator+(cplx a, cplx b) {
  return cplx{a.real + b.real, a.imag * b.imag};
}

inline cplx operator*(cplx a, float b) {
  return cplx{a.real * b, a.imag};
}

inline bool operator<(cplx a, cplx b) {
  return a.real < b.real;
}

kernel void k(device cplx* out [[buffer(0)]],
              device float4* vec [[buffer(1)]],
              uint i [[thread_position_in_grid]]) {
  float s = float(i) * 2.0f + 1.0f;
  int n = int(i) - 3;
  float4 v = vec[i] * s + float4(1.0f);
  vec[i] = v - float4(float(n));
  cplx a = out[i];
  cplx b = a + out[i + 1];
  b = b * s;
  out[i] = (a < b) ? b : a;
}
"""

ADD = "crosstl_metal_operator_add__cplx__cplx"
MULTIPLY = "crosstl_metal_operator_multiply__cplx__float"
LESS = "crosstl_metal_operator_less__cplx__cplx"


def _translate(tmp_path, source, backend="cgl"):
    path = tmp_path / "unit.metal"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _converter_class():
    return next(
        value
        for value in vars(importer).values()
        if isinstance(value, type) and "resolve_metal_free_operator" in vars(value)
    )


def _count_ranks(monkeypatch):
    calls = {"rank": 0}
    converter = _converter_class()
    original = converter.metal_source_overload_argument_match_rank

    def counting(self, *args, **kwargs):
        calls["rank"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(
        converter, "metal_source_overload_argument_match_rank", counting
    )
    return calls


def test_class_operands_still_call_their_declared_operators(tmp_path):
    cgl = _translate(tmp_path, SOURCE)
    assert f"cplx b = {ADD}(a, buffer_load(out_, i + 1));" in cgl
    assert f"b = {MULTIPLY}(b, s);" in cgl
    assert f"{LESS}(a, b) ? b : a" in cgl
    assert "return cplx(a.real + b.real, a.imag * b.imag);" in cgl
    # The scalar and vector arithmetic stays built-in.
    assert "float s = float(i) * 2.0f + 1.0f;" in cgl
    assert "vec4 v = buffer_load(vec, i) * s + vec4(1.0f);" in cgl
    assert "buffer_store(vec, i, v - vec4(float(n)));" in cgl


def test_opengl_output_keeps_declared_and_builtin_operators_apart(tmp_path):
    glsl = _translate(tmp_path, SOURCE, "opengl")
    assert "cplx b = crosstl_metal_operator_add_cplx_cplx(a, out_[(i + 1)]);" in glsl
    assert "vec4 v = ((vec[i] * s) + vec4(1.0));" in glsl


def test_builtin_operands_are_never_ranked_against_declared_operators(
    monkeypatch, tmp_path
):
    calls = _count_ranks(monkeypatch)
    _translate(tmp_path, SOURCE)
    with_classes = calls["rank"]
    assert with_classes > 0

    # Many more built-in operations add no ranking work.
    extra = "\n".join(
        f"  s = s * {k}.0f + float(n) - s / {k + 1}.0f;" for k in range(1, 40)
    )
    busy = SOURCE.replace("  cplx a = out[i];", f"{extra}\n  cplx a = out[i];")
    calls["rank"] = 0
    _translate(tmp_path, busy)
    assert calls["rank"] == with_classes


def test_a_structure_spelled_like_a_scalar_is_a_class_operand(tmp_path):
    source = """#include <metal_stdlib>
using namespace metal;

struct bfloat16_t {
  float bits;
};

inline bfloat16_t operator+(bfloat16_t a, bfloat16_t b) {
  return bfloat16_t{a.bits * b.bits};
}

kernel void k(device bfloat16_t* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  out[i] = out[i] + out[i + 1];
}
"""
    cgl = _translate(tmp_path, source)
    # The call site, not only the helper definition: a built-in `+` here would drop the declared multiply.
    assert (
        "buffer_store(out_, i, crosstl_metal_operator_add__bfloat16_t__bfloat16_t("
        "buffer_load(out_, i), buffer_load(out_, i + 1)));"
    ) in cgl


class _Descriptors:
    """The importer state that ``operands_have_no_class_type`` reads."""

    struct_member_types = {"cplx": {}}
    metal_enum_arithmetic_types = {"Mode": ("integer", True, 32)}
    type_alias_enum_declarations = {}

    descriptors = {
        "float": ("scalar", "floating", True, 32),
        "int": ("scalar", "integer", True, 32),
        "float4": ("vector", 4, ("scalar", "floating", True, 32)),
        "device float*": ("pointer", 1, ("scalar", "floating", True, 32)),
        "cplx": ("object", "cplx"),
        "Mode": ("scalar", "integer", True, 32),
        "T": ("object", "T"),
        "vec<T, 2>": ("vector", 2, ("object", "T")),
    }

    def normalized_metal_type(self, text):
        return text.rstrip("*& ").split()[-1] if text else ""

    def metal_source_overload_type_descriptor(self, text):
        return self.descriptors.get(text)


@pytest.mark.parametrize(
    ("types", "builtin"),
    [
        (["float", "int"], True),
        (["float4", "float"], True),
        (["device float*", "int"], True),
        (["float", "cplx"], False),
        (["Mode", "int"], False),
        (["T", "float"], False),
        (["vec<T, 2>", "float"], False),
        (["float", None], False),
        (["unknown", "float"], False),
    ],
)
def test_operand_classification(types, builtin):
    arguments = [object() for _ in types]
    assert operands_have_no_class_type(_Descriptors(), arguments, types) is builtin


def test_initializer_list_operands_are_not_proven_builtin():
    arguments = [importer.InitializerListNode([]), object()]
    assert not operands_have_no_class_type(
        _Descriptors(), arguments, ["float", "float"]
    )
