"""A generic structure's dependent ``static_assert`` is checked in its specializations, never in the template.

The parser hoisted a generic structure's assertions into ``ShaderNode.static_assertions`` without the structure's
type parameters, so ``sizeof(T)`` was folded as if ``T`` were a type of the program and refused. A Metal template
structure with ``static_assert(sizeof(T) <= 16)`` failed to translate back to Metal, although the importer had
already materialized every specialization with its own, substituted assertion.
"""

import textwrap

import pytest

import crosstl
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics import (
    StaticAssertionFailedError,
    StaticAssertionUnresolvedError,
)
from crosstl.translator.semantics.template_assertions import (
    resolve_template_assertions,
)

TARGETS = ("metal", "directx", "opengl", "vulkan", "wgsl", "cuda", "hip", "slang")


def _shader(template_body, use="Box_float box;", helpers=""):
    return textwrap.dedent("""
        shader main {
            generic<T> struct Box {
                T value;
        %s
            };
            struct Box_float {
                float value;
                static_assert(4 <= 16, "fits");
            };
        %s
            compute {
                RWStructuredBuffer<float> out_ @buffer(0);
                @numthreads(1, 1, 1)
                void main(uint3 gid @gl_GlobalInvocationID) {
                    %s
                    out_[gid.x] = 1.0;
                }
            }
        }
        """) % (template_body, helpers, use)


def _parse(source):
    return Parser(Lexer(source).get_tokens()).parse()


def _translate(tmp_path, source, backend, name="shader.cgl"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


DEPENDENT = (
    '        static_assert(sizeof(T) <= 16 && 16 % metal_sizeof(T) == 0, "fits");'
)


def test_parser_records_the_template_of_each_assertion():
    source = _shader(DEPENDENT).replace(
        "generic<T> struct Box {",
        "struct Pair<A, B> { A a; B b; static_assert(sizeof(A) == sizeof(B)); };\n"
        "    generic<T> struct Box {",
    )
    declared = _parse(source).static_assertions
    assert [(node.template, node.generic_params) for node in declared] == [
        ("Pair", ("A", "B")),
        ("Box", ("T",)),
        (None, ()),
    ]


@pytest.mark.parametrize("backend", TARGETS)
def test_dependent_assertion_of_an_unused_template_is_removed(tmp_path, backend):
    code = _translate(tmp_path, _shader(DEPENDENT), backend)
    assert "static_assert" not in code


def test_removal_reports_the_dependent_assertions_only():
    program = _parse(_shader(DEPENDENT + "\n        static_assert(2 > 1);"))
    assert resolve_template_assertions(program, "metal") == 1
    assert [node.template for node in program.static_assertions] == ["Box", None]


def test_specialization_assertion_is_still_checked(tmp_path):
    source = _shader(DEPENDENT).replace("4 <= 16", "4 <= 2")
    with pytest.raises(StaticAssertionFailedError, match="fits"):
        _translate(tmp_path, source, "metal")


def test_independent_assertion_of_an_unused_template_is_still_checked(tmp_path):
    source = _shader('        static_assert(1 > 2, "never");')
    with pytest.raises(StaticAssertionFailedError, match="never"):
        _translate(tmp_path, source, "opengl")


@pytest.mark.parametrize("backend", ["metal", "directx", "opengl"])
def test_dependent_assertion_of_an_instantiated_template_is_refused(tmp_path, backend):
    source = _shader(DEPENDENT, use="Box<float> box;")
    with pytest.raises(StaticAssertionUnresolvedError, match="generic structure Box"):
        _translate(tmp_path, source, backend)


def test_template_named_only_by_unreachable_code_is_unused(tmp_path):
    helpers = textwrap.dedent("""
            float unused_helper() {
                Box<float> box;
                return box.value;
            }
        """)
    code = _translate(tmp_path, _shader(DEPENDENT, helpers=helpers), "opengl")
    assert "static_assert" not in code


def test_template_named_only_by_another_dependent_assertion_is_unused(tmp_path):
    source = _shader(DEPENDENT).replace(
        "generic<T> struct Box {",
        "generic<U> struct Probe { U value; static_assert(sizeof(Box<U>) > 0); };\n"
        "    generic<T> struct Box {",
    )
    assert "static_assert" not in _translate(tmp_path, source, "metal")


# metal::is_same_v<T, float>, which the Metal importer keeps as one escaped identifier.
ESCAPED = "metal_u3a_u3ais_same_v_u3cT_u2c_u20float_u3e"


def test_escaped_source_spelling_naming_a_parameter_is_dependent(tmp_path):
    source = _shader(f"        static_assert({ESCAPED});")
    assert "static_assert" not in _translate(tmp_path, source, "metal")


def test_escaped_source_spelling_outside_a_template_is_refused(tmp_path):
    source = _shader("").replace(
        'static_assert(4 <= 16, "fits");', f"static_assert({ESCAPED});"
    )
    with pytest.raises(StaticAssertionUnresolvedError):
        _translate(tmp_path, source, "metal")


METAL = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;

    struct pair_t {
      float real;
      float imag;
    };

    template <typename in_T>
    struct Traits {
      static_assert(sizeof(in_T) <= 16 && 16 % sizeof(in_T) == 0, "width");
    };

    template <typename T>
    [[kernel]] void copy_pairs(
        device const T* in [[buffer(0)]],
        device T* out [[buffer(1)]],
        uint i [[thread_position_in_grid]]) {
      Traits<T> traits;
      (void)traits;
      out[i] = in[i];
    }

    template [[host_name("copy_pairs_pair")]] [[kernel]] void copy_pairs<pair_t>(
        device const pair_t*, device pair_t*, uint);
    """)


@pytest.mark.parametrize("backend", ["metal", "directx", "opengl"])
def test_metal_template_assertion_round_trips(tmp_path, backend):
    metal = tmp_path / "kernel.metal"
    metal.write_text(METAL, encoding="utf-8")
    code = crosstl.translate(str(metal), backend=backend, format_output=False)
    assert "Traits_pair_t traits;" in code
    assert "static_assert" not in code
    cgl = crosstl.translate(str(metal), backend="cgl", format_output=False)
    assert "metal_sizeof(in_T)" in cgl
    assert "static_assert(8 <= 16" in cgl
    failing = cgl.replace("static_assert(8 <= 16", "static_assert(8 <= 4")
    with pytest.raises(StaticAssertionFailedError, match="width"):
        _translate(tmp_path, failing, backend, name="failing.cgl")
    holding = _translate(tmp_path, cgl, backend, name="holding.cgl")
    assert "static_assert" not in holding
