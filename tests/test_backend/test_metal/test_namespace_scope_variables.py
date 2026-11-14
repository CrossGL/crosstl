"""Namespace-scope Metal variables: qualified references and same-named members of different namespaces."""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.Metal.namespace_scope_variables import MetalNamespaceLookupError

KERNEL_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""


def _translate(tmp_path, body, backend="opengl"):
    source = tmp_path / "unit.metal"
    source.write_text(KERNEL_HEADER + body, encoding="utf-8")
    return crosstl.translate(str(source), backend=backend, format_output=False)


def _compile_glsl(tmp_path, glsl):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")
    shader = tmp_path / "unit.comp"
    shader.write_text(glsl, encoding="utf-8")
    result = subprocess.run(
        [glslang, "-S", "comp", str(shader)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_qualified_reference_binds_to_namespace_constant(tmp_path):
    glsl = _translate(
        tmp_path,
        """
        namespace outer { namespace inner {
        constant constexpr uint COUNT = 4;
        } }
        struct Params { uint sizes[outer::inner::COUNT]; };
        kernel void k(device uint* o [[buffer(0)]], constant Params& p [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
            uint total = 0;
            for (uint d = 0; d < outer::inner::COUNT; ++d) total += p.sizes[d];
            o[i] = total + ::outer::inner::COUNT;
        }
        """,
    )
    assert "u3a" not in glsl
    assert "uint sizes[4];" in glsl
    assert "(d < COUNT)" in glsl
    _compile_glsl(tmp_path, glsl)


def test_namespace_alias_and_relative_qualification_bind(tmp_path):
    glsl = _translate(
        tmp_path,
        """
        namespace outer { namespace inner { constant constexpr int COUNT = 4; } }
        namespace short_name = outer::inner;
        namespace outer { inline int twice() { return 2 * inner::COUNT; } }
        kernel void k(device int* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
            o[i] = short_name::COUNT + outer::twice();
        }
        """,
    )
    assert "u3a" not in glsl
    assert "return (2 * COUNT);" in glsl
    assert "(COUNT + twice())" in glsl


def test_same_named_constants_of_different_namespaces_stay_distinct(tmp_path):
    glsl = _translate(
        tmp_path,
        """
        namespace first { constant constexpr int COUNT = 2; inline int scaled(int x) { return x * COUNT; } }
        namespace second { constant constexpr int COUNT = 3; constant constexpr int TWICE = 2 * COUNT; }
        constant constexpr int COUNT = 5;
        kernel void k(device int* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
            o[i] = first::scaled(1) + first::COUNT + second::COUNT + second::TWICE + COUNT;
        }
        """,
    )
    assert "const int first_COUNT = 2;" in glsl
    assert "const int second_COUNT = 3;" in glsl
    assert "const int COUNT = 5;" in glsl
    assert "const int TWICE = (2 * second_COUNT);" in glsl
    assert "return (x * first_COUNT);" in glsl
    assert "((((scaled(1) + first_COUNT) + second_COUNT) + TWICE) + COUNT)" in glsl
    _compile_glsl(tmp_path, glsl)


def test_local_declaration_hides_same_named_namespace_constants(tmp_path):
    glsl = _translate(
        tmp_path,
        """
        namespace first { constant constexpr int COUNT = 2; }
        namespace second { constant constexpr int COUNT = 3; }
        inline int pick(int COUNT) { return COUNT + 1; }
        kernel void k(device int* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
            int COUNT = first::COUNT;
            for (int COUNT_2 = 0; COUNT_2 < 1; ++COUNT_2) { COUNT += second::COUNT; }
            o[i] = pick(COUNT);
        }
        """,
    )
    assert "return (COUNT + 1);" in glsl
    assert "int COUNT = first_COUNT;" in glsl
    assert "COUNT += second_COUNT;" in glsl
    _compile_glsl(tmp_path, glsl)


def test_using_directive_selects_namespace_for_unqualified_use(tmp_path):
    glsl = _translate(
        tmp_path,
        """
        namespace first { constant constexpr int COUNT = 2; }
        namespace second { constant constexpr int COUNT = 3; }
        using namespace second;
        kernel void k(device int* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
            o[i] = COUNT;
        }
        """,
    )
    assert "o[i] = second_COUNT;" in glsl


def test_generated_name_avoids_existing_identifier(tmp_path):
    glsl = _translate(
        tmp_path,
        """
        namespace first { constant constexpr int COUNT = 2; }
        constant constexpr int COUNT = 5;
        constant constexpr int first_COUNT = 7;
        kernel void k(device int* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
            o[i] = first::COUNT + first_COUNT + COUNT;
        }
        """,
    )
    assert "const int first_COUNT_2 = 2;" in glsl
    assert "((first_COUNT_2 + first_COUNT) + COUNT)" in glsl


def test_ambiguous_unqualified_use_is_refused(tmp_path):
    with pytest.raises(MetalNamespaceLookupError) as exc_info:
        _translate(
            tmp_path,
            """
            namespace first { constant constexpr int COUNT = 2; }
            namespace second { constant constexpr int COUNT = 3; }
            using namespace first;
            using namespace second;
            kernel void k(device int* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
                o[i] = COUNT;
            }
            """,
            backend="cgl",
        )
    assert exc_info.value.candidates == ("first::COUNT", "second::COUNT")
