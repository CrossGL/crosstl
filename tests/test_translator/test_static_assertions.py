"""CrossGL ``static_assert`` declarations are checked at translation time for every target, then removed.

The parser skipped a shader-scope ``static_assert`` token by token and read a block-scope one as a call of an
undeclared function, so a false assertion translated silently and GLSL output called ``static_assert``.
"""

import textwrap

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics import (
    SizeofUnsupportedError,
    StaticAssertionFailedError,
    StaticAssertionUnresolvedError,
    resolve_static_assertions,
)

TARGETS = (
    "metal",
    "directx",
    "opengl",
    "vulkan",
    "wgsl",
    "cuda",
    "hip",
    "slang",
    "rust",
    "mojo",
)


def _shader(declarations="", struct_body="", stage_scope="", body=""):
    return textwrap.dedent("""
        shader main {
            const int WIDTH = 8;
            const uint LANES = 4u;
            enum Mode { Off, Low, High };
        %s
            struct Pair {
                int a;
                int b;
        %s
            };
            compute {
        %s
                RWStructuredBuffer<int> out_ @buffer(0);
                @numthreads(1, 1, 1)
                void main(uint3 gid @gl_GlobalInvocationID) {
        %s
                    out_[gid.x] = WIDTH;
                }
            }
        }
        """) % (declarations, struct_body, stage_scope, body)


def _parse(source):
    return Parser(Lexer(source).get_tokens()).parse()


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


HOLDING = _shader(
    declarations='    static_assert(WIDTH * 2 == 16 && LANES <= 4u, "eight wide");',
    struct_body="        static_assert(High == 2);",
    stage_scope="        static_assert(WIDTH % 3 != 0 ? true : false);",
    body='            static_assert(!(WIDTH < LANES), "lanes fit");',
)


def test_parser_keeps_static_assertions_with_their_messages():
    program = _parse(HOLDING)
    declared = program.static_assertions
    assert [type(node) for node in declared] == [A.StaticAssertNode] * 3
    assert [node.message for node in declared] == ["eight wide", None, None]
    assert isinstance(declared[0].condition, A.BinaryOpNode)
    assert all(
        not isinstance(member, A.StaticAssertNode)
        for struct in program.structs
        for member in getattr(struct, "members", None) or []
    )
    stage = next(iter(program.stages.values()))
    statements = stage.entry_point.body
    statements = getattr(statements, "statements", statements)
    first = statements[0]
    assert isinstance(first, A.StaticAssertNode)
    assert first.message == "lanes fit"


@pytest.mark.parametrize("backend", TARGETS)
def test_holding_assertions_are_removed_for_every_target(tmp_path, backend):
    code = _translate(tmp_path, HOLDING, backend)
    assert "static_assert" not in code


def test_resolution_reports_how_many_assertions_held():
    program = _parse(HOLDING)
    assert resolve_static_assertions(program, "opengl") == 4
    assert program.static_assertions == []


@pytest.mark.parametrize("backend", TARGETS)
def test_false_assertion_is_refused_with_its_message(tmp_path, backend):
    source = _shader(declarations='    static_assert(WIDTH == 4, "four wide");')
    with pytest.raises(StaticAssertionFailedError, match="four wide") as error:
        _translate(tmp_path, source, backend)
    assert error.value.code == "project.translate.static-assertion-failed"


@pytest.mark.parametrize(
    "placement",
    ["declarations", "struct_body", "stage_scope", "body"],
)
def test_false_assertion_is_refused_in_every_scope(tmp_path, placement):
    source = _shader(**{placement: "    static_assert(Low > High);"})
    with pytest.raises(StaticAssertionFailedError, match="Low > High"):
        _translate(tmp_path, source, "opengl")


@pytest.mark.parametrize(
    "condition, reason",
    [
        ("1.5 > 1.0", "not an integer"),
        ("WIDTH - 9 < 0", "outside"),
        ("-1 < 0", "outside"),
        ("(1 << 31) != 0", "outside"),
        ("WIDTH / 0 == 1", "divides by zero"),
        ("UNKNOWN == 1", "UNKNOWN"),
    ],
)
def test_unevaluable_assertion_is_refused(tmp_path, condition, reason):
    source = _shader(declarations=f"    static_assert({condition});")
    with pytest.raises(StaticAssertionUnresolvedError, match=reason) as error:
        _translate(tmp_path, source, "opengl")
    assert error.value.code == "project.translate.static-assertion-unresolved"


def test_sizeof_of_a_structure_keeps_its_own_refusal(tmp_path):
    source = _shader(declarations="    static_assert(sizeof(Pair) == 8);")
    with pytest.raises(SizeofUnsupportedError):
        _translate(tmp_path, source, "opengl")


def test_sizeof_folded_for_the_target_is_evaluated(tmp_path):
    source = _shader(declarations="    static_assert(sizeof(float4) == 16);")
    assert "static_assert" not in _translate(tmp_path, source, "opengl")


def test_block_assertion_naming_a_local_is_refused(tmp_path):
    source = _shader(
        body="            int WIDTH2 = 3;\n            static_assert(WIDTH2 == 3);"
    )
    with pytest.raises(StaticAssertionUnresolvedError, match="WIDTH2"):
        _translate(tmp_path, source, "opengl")


def test_crossgl_output_keeps_the_source_text(tmp_path):
    assert "static_assert(WIDTH * 2 == 16" in _translate(tmp_path, HOLDING, "cgl")


METAL = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;
    constant constexpr int N = 4;
    static_assert(N * 2 == 8, "eight");
    struct P { int a; static_assert(sizeof(int) == 4); };
    kernel void k(device int* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
        static_assert(N > 1);
        out[i] = N;
    }
    """)


@pytest.mark.parametrize("backend", ["opengl", "directx", "metal", "vulkan"])
def test_metal_assertions_survive_crossgl_and_are_checked(tmp_path, backend):
    metal = tmp_path / "kernel.metal"
    metal.write_text(METAL, encoding="utf-8")
    cgl = crosstl.translate(str(metal), backend="cgl", format_output=False)
    assert cgl.count("static_assert(") == 3
    failing = tmp_path / "failing.cgl"
    failing.write_text(cgl.replace("N > 1", "N > 4"), encoding="utf-8")
    with pytest.raises(StaticAssertionFailedError, match="N > 4"):
        crosstl.translate(str(failing), backend=backend, format_output=False)
    holding = tmp_path / "holding.cgl"
    holding.write_text(cgl, encoding="utf-8")
    code = crosstl.translate(str(holding), backend=backend, format_output=False)
    assert "static_assert" not in code
