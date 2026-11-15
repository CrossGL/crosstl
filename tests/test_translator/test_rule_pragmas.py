"""CrossGL programs declare the rule families of their source language with ``#pragma cgl_rules(...)``.

Without the pragma, translating a program's saved CrossGL dropped the rules its source language defines: Metal's
``int(x)`` converts NaN to 0 (MSL 4.1 section 8.6), but the same CrossGL text translated as plain CrossGL left it
undefined on GLSL. Each case below fails without the pragma.
"""

import textwrap

import pytest

import crosstl
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics.rule_pragmas import (
    UnknownRuleFamilyError,
    rule_pragma,
    take_declared_families,
)

METAL = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;
    kernel void truncate(device const float* values [[buffer(0)]],
                         device int* results [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) {
        results[i] = int(values[i]);
    }
""")

CROSSGL = textwrap.dedent("""
    PRAGMA
    shader main {
        compute {
            RWStructuredBuffer<float> values @buffer(0);
            RWStructuredBuffer<int> results @buffer(1);
            @numthreads(1, 1, 1)
            void main(uvec3 id @ gl_GlobalInvocationID) {
                results[id.x] = int(values[id.x]);
            }
        }
    }
""")


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_imported_programs_declare_their_source_families():
    assert rule_pragma("metal") == "#pragma cgl_rules(ftoi_nan)"
    assert rule_pragma("directx") == "#pragma cgl_rules(udiv_zero, wrap)"
    assert rule_pragma("opengl") == "#pragma cgl_rules(wrap)"
    assert rule_pragma("cuda") == "#pragma cgl_rules(fp_ieee, libm, round_away)"
    assert rule_pragma("slang") is None


def test_the_crossgl_of_a_metal_program_carries_its_rules(tmp_path):
    crossgl = crosstl.translate(_write(tmp_path, "a.metal", METAL), backend="cgl")

    assert crossgl.startswith("#pragma cgl_rules(ftoi_nan)\n")
    saved = _write(tmp_path, "a.cgl", crossgl)
    source = _write(tmp_path, "b.metal", METAL)
    for backend in ("opengl", "directx", "vulkan", "metal"):
        assert crosstl.translate(saved, backend=backend) == crosstl.translate(
            source, backend=backend
        ), backend


@pytest.mark.parametrize(
    ("pragma", "carried"),
    [("#pragma cgl_rules(ftoi_nan)", True), ("", False)],
)
def test_declared_families_apply_to_crossgl_sources(tmp_path, pragma, carried):
    path = _write(tmp_path, "a.cgl", CROSSGL.replace("PRAGMA", pragma))

    glsl = crosstl.translate(path, backend="opengl")

    assert ("cgl_rule_sat_ftoi(" in glsl) is carried
    assert "cgl_rules" not in glsl


def test_the_pragma_is_removed_from_the_program():
    program = Parser(
        Lexer(
            CROSSGL.replace("PRAGMA", "#pragma cgl_rules(ftoi_nan, wrap)")
        ).get_tokens()
    ).parse()

    assert take_declared_families(program) == {"ftoi_nan", "wrap"}
    assert not any(node.directive == "pragma" for node in program.preprocessors)


@pytest.mark.parametrize(
    "pragma", ["#pragma cgl_rules(saturate)", "#pragma cgl_rules ftoi_nan"]
)
def test_unknown_families_are_refused(tmp_path, pragma):
    path = _write(tmp_path, "a.cgl", CROSSGL.replace("PRAGMA", pragma))

    with pytest.raises(UnknownRuleFamilyError) as raised:
        crosstl.translate(path, backend="opengl")

    assert raised.value.code == "project.translate.unknown-rule-family"
