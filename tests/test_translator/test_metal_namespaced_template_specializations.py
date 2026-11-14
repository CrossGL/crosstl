"""Concrete specializations of namespaced Metal function templates keep the template's namespace scope.

A function template looks up the names in its signature and body in the namespace that declares it (C++
[basic.lookup.unqual], [temp.res]). The Metal importer appended each concrete specialization at global scope, so a
type alias, alias template or constant declared beside a namespaced template was undeclared in the specialization:
translation refused ``real_t``, or emitted ``common_dtype<float, float>`` verbatim as invalid Metal. The
specialization is now declared in reopened definitions of the template's namespace and called by its qualified name.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.Metal.namespaced_specializations import (
    place_in_namespace,
    specialization_namespaces,
    specialization_reference,
)
from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from crosstl.project.namespace_alias_scopes import (
    alias_visible_at,
    alias_visible_in_scope,
    reopened_namespace_scopes,
)

TARGETS = ("metal", "directx", "opengl", "vulkan", "cgl")

EXPLICIT_SOURCE = """\
#include <metal_stdlib>
using namespace metal;
namespace lib {
namespace detail {
struct Pair { float a; float b; };
using real_t = float;
constant float kScale = 2.0f;
inline float twice(float v) { return v * kScale; }
template <typename T>
inline real_t combine(const T x) {
  Pair p = {x, x};
  real_t sum = twice(p.a) + p.b;
  return sum;
}
inline float forward(float x) { return combine<float>(x); }
}
}
kernel void run(device float* out [[buffer(0)]], device const float* in [[buffer(1)]],
                uint gid [[thread_position_in_grid]]) {
  out[gid] = lib::detail::combine<float>(in[gid]) + lib::detail::forward(in[gid]);
}
"""

CONSTRAINED_SOURCE = """\
#include <metal_stdlib>
using namespace metal;
namespace lib {
template <typename T>
constexpr constant bool is_pair_v = ::metal::is_same_v<T, float2>;
template <typename U, typename V>
using common_t = decltype(U(0) + V(0));
template <typename T, typename U, ::metal::enable_if_t<!is_pair_v<T>, bool> = true>
inline common_t<T, U> quotient(const T x, const U y) {
  return x / y;
}
template <typename T, typename U, ::metal::enable_if_t<is_pair_v<T> && is_pair_v<U>, bool> = true>
inline common_t<T, U> quotient(const T x, const U y) {
  const auto scale = y.x * y.x + y.y * y.y;
  return T(x.x * y.x + x.y * y.y, x.y * y.x - x.x * y.y) / scale;
}
}
kernel void run(device float* out [[buffer(0)]], device const float* in [[buffer(1)]],
                device float2* pairs [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  out[gid] = lib::quotient(in[gid], in[gid + 1]);
  pairs[gid] = lib::quotient(pairs[gid], pairs[gid + 1]);
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "kernel.metal"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _compile_metal(tmp_path, code):
    if not shutil.which("xcrun"):
        return
    source = tmp_path / "out.metal"
    source.write_text(code)
    result = subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(source),
            "-o",
            str(tmp_path / "out.air"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _compile_glsl(tmp_path, code):
    if not shutil.which("glslangValidator"):
        return
    source = tmp_path / "out.comp"
    source.write_text(code)
    result = subprocess.run(
        ["glslangValidator", str(source)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _validate_spirv(tmp_path, code):
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not (spirv_as and spirv_val):
        return
    source = tmp_path / "out.spvasm"
    source.write_text(code)
    binary = tmp_path / "out.spv"
    for command in (
        [spirv_as, "--target-env", "vulkan1.1", str(source), "-o", str(binary)],
        [spirv_val, "--target-env", "vulkan1.1", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr


def _check_output(tmp_path, backend, code, names):
    if backend == "vulkan":
        _validate_spirv(tmp_path, code)
        return
    for name in names:
        assert re.search(rf"\b{name}\b", code), name
    if backend == "metal":
        _compile_metal(tmp_path, code)
    elif backend == "opengl":
        _compile_glsl(tmp_path, code)


def test_explicit_specialization_is_declared_in_the_reopened_namespace():
    processed = MetalPreprocessor().preprocess(EXPLICIT_SOURCE)
    assert "using real_t = float;" in processed
    assert "::lib::detail::combine_float(in[gid])" in processed
    assert "return ::lib::detail::combine_float(x);" in processed
    tail = processed[processed.index("kernel void run") :]
    reopened = re.search(
        r"namespace lib \{\s*namespace detail \{\s*inline real_t combine_float\(", tail
    )
    assert reopened is not None, tail


def test_constrained_specializations_are_declared_in_the_reopened_namespace():
    processed = MetalPreprocessor().preprocess(CONSTRAINED_SOURCE)
    tail = processed[processed.index("kernel void run") :]
    assert "::lib::quotient_float_float(in[gid], in[gid + 1])" in tail
    assert "::lib::quotient_float2_float2(pairs[gid], pairs[gid + 1])" in tail
    # A concrete alias-template reference names its type ([temp.alias]/2):
    # common_t<float, float> is decltype(float(0) + float(0)), float.
    assert re.search(r"namespace lib \{\s*inline float quotient_float_float\(", tail)
    assert re.search(r"namespace lib \{\s*inline float2 quotient_float2_float2\(", tail)


@pytest.mark.parametrize("backend", TARGETS)
def test_namespace_scoped_alias_reaches_every_target(tmp_path, backend):
    code = _translate(tmp_path, EXPLICIT_SOURCE, backend)
    assert "real_t" not in code
    _check_output(tmp_path, backend, code, ("combine_float",))


@pytest.mark.parametrize("backend", TARGETS)
def test_namespace_scoped_alias_template_reaches_every_target(tmp_path, backend):
    code = _translate(tmp_path, CONSTRAINED_SOURCE, backend)
    assert "common_t" not in code
    assert "is_pair_v" not in code
    _check_output(
        tmp_path, backend, code, ("quotient_float_float", "quotient_float2_float2")
    )


def test_global_templates_keep_their_placement():
    source = (
        EXPLICIT_SOURCE.replace("namespace lib {\nnamespace detail {\n", "")
        .replace(
            "inline float forward(float x) { return combine<float>(x); }\n}\n}\n",
            "inline float forward(float x) { return combine<float>(x); }\n",
        )
        .replace("lib::detail::", "")
    )
    processed = MetalPreprocessor().preprocess(source)
    assert "combine_float(in[gid])" in processed
    assert "::combine_float" not in processed
    assert "namespace" not in processed.replace("using namespace metal;", "")


def test_place_in_namespace_reports_the_declaration_offset():
    text, offset = place_in_namespace("a::b", "inline float f() { return 1; }\n")
    assert (
        text == "namespace a {\nnamespace b {\ninline float f() { return 1; }\n}\n}\n"
    )
    assert text[offset:].startswith("inline float f()")
    assert place_in_namespace("", "int x;") == ("int x;", 0)
    assert specialization_reference("a::b", "f_float") == "::a::b::f_float"
    assert specialization_reference("", "f_float") == "f_float"


class _Template:
    def __init__(self, span, namespace):
        self.span = span
        self.namespace = namespace


def test_structure_member_templates_keep_the_global_placement():
    member = _Template((10, 20), "lib")
    free = _Template((40, 50), "lib")
    global_ = _Template((60, 70), "")
    placements = specialization_namespaces([member, free, global_], [(5, 30)])
    assert placements == {(10, 20): "", (40, 50): "lib", (60, 70): ""}


def test_reopened_namespace_blocks_extend_alias_visibility():
    source = "namespace a { using t = float; } int x; namespace a { t y; } namespace b { t z; }"
    first_open = source.index("{")
    first_close = source.index("}")
    second_open = source.index("{", first_close)
    second_close = source.index("}", second_open)
    third_open = source.index("{", second_close)
    third_close = source.index("}", third_open)
    namespace_spans = [
        (first_open + 1, first_close, "a"),
        (second_open + 1, second_close, "a"),
        (third_open + 1, third_close, "b"),
    ]
    declaration_end = source.index(";") + 1
    scopes = reopened_namespace_scopes(
        (first_open, first_close + 1), declaration_end, namespace_spans
    )
    assert scopes == ((second_open, second_close + 1),)
    alias = {
        "end": declaration_end,
        "scope_end": first_close + 1,
        "reopened_scopes": scopes,
    }
    assert alias_visible_at(alias, source.index("t y"))
    assert not alias_visible_at(alias, source.index("int x"))
    assert not alias_visible_at(alias, source.index("t z"))
    assert alias_visible_in_scope(alias, source.index("t y"), second_close + 1)
    assert not alias_visible_in_scope(alias, source.index("t z"), third_close + 1)
    assert reopened_namespace_scopes((0, len(source)), 0, namespace_spans) == ()
