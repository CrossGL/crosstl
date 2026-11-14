"""Integer types in translation-time constant expressions: conversions wrap, and type aliases name their types.

The evaluators of binding indices, array extents, ``sizeof`` and ``static_assert`` kept the source value of a
conversion to a narrow integer type (``uint8_t(300)`` was 300, not 44) and refused every constant, conversion and
``sizeof`` operand that named a program-scope ``typedef``, which the MSL, HLSL and GLSL generators keep.
"""

import textwrap

import pytest

import crosstl
from crosstl.translator.semantics import (
    StaticAssertionFailedError,
    StaticAssertionUnresolvedError,
)
from crosstl.translator.semantics.binding_constants import INTEGER_TYPES
from crosstl.translator.semantics.integral_types import (
    INTEGER_REPRESENTATIONS,
    convert_to_integer_type,
)

TARGETS = ("metal", "directx", "opengl", "vulkan", "wgsl", "cuda", "slang")


def _shader(declarations="", binding="0"):
    return textwrap.dedent("""
        shader main {
            typedef uint word;
            typedef word index_t;
        %s
            compute {
                RWStructuredBuffer<uint> out_ @buffer(%s);
                @numthreads(1, 1, 1)
                void main(uint3 gid @gl_GlobalInvocationID) {
                    out_[gid.x] = 1u;
                }
            }
        }
        """) % (declarations, binding)


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize(
    "type_name, value, expected",
    [
        ("uint8_t", 300, 44),
        ("uchar", 256, 0),
        ("int8_t", 200, -56),
        ("char", 127, 127),
        ("ushort", 65537, 1),
        ("short", 40000, -25536),
        ("uint", -1, 2**32 - 1),
        ("int", 2**31, -(2**31)),
        ("ulong", -1, 2**64 - 1),
        ("size_t", 2**64 + 3, 3),
    ],
)
def test_integer_conversion_is_modular(type_name, value, expected):
    assert convert_to_integer_type(value, type_name) == expected


def test_every_integer_type_has_a_width():
    assert INTEGER_TYPES == frozenset(INTEGER_REPRESENTATIONS)


ALIASED = _shader(
    declarations=textwrap.indent(
        textwrap.dedent("""
            const index_t SLOT = 3;
            const word LANES = 4;
            static_assert(LANES * 2 == 8, "aliased constant");
            static_assert(index_t(LANES) == 4, "aliased conversion");
            static_assert(sizeof(index_t) == 4, "aliased sizeof");
            static_assert(sizeof(word[LANES]) == 16, "aliased array");
            """),
        "    ",
    ),
    binding="SLOT",
)


@pytest.mark.parametrize("backend", TARGETS)
def test_alias_typed_constants_are_integer_constants(tmp_path, backend):
    code = _translate(tmp_path, ALIASED, backend)
    assert "static_assert" not in code


@pytest.mark.parametrize(
    "backend, binding",
    [
        ("metal", "[[buffer(3)]]"),
        ("directx", "register(u3)"),
        ("opengl", "binding = 3"),
    ],
)
def test_alias_typed_constant_binds_its_index(tmp_path, backend, binding):
    code = _translate(tmp_path, ALIASED, backend)
    assert binding in code.replace("binding=", "binding = ")


@pytest.mark.parametrize(
    "condition",
    [
        "uint8_t(300) == 44",
        "uchar(255 + 1) == 0",
        "ushort(65537) == 1",
        "(uint8_t)258 == 2",
        "NARROW == 2",
    ],
)
def test_conversions_to_narrow_types_wrap(tmp_path, condition):
    source = _shader(
        declarations=f"    const uint8_t NARROW = 258;\n    static_assert({condition});"
    )
    code = _translate(tmp_path, source, "opengl")
    assert "static_assert" not in code


@pytest.mark.parametrize(
    "condition", ["uint8_t(300) == 300", "ushort(65536) != 0", "NARROW == 258"]
)
def test_assertions_that_overlook_the_wrap_fail(tmp_path, condition):
    source = _shader(
        declarations=f"    const uint8_t NARROW = 258;\n    static_assert({condition});"
    )
    with pytest.raises(StaticAssertionFailedError):
        _translate(tmp_path, source, "metal")


def test_negative_result_of_a_signed_narrow_conversion_is_unresolved(tmp_path):
    source = _shader(declarations="    static_assert(int8_t(200) < 0);")
    with pytest.raises(StaticAssertionUnresolvedError, match="outside"):
        _translate(tmp_path, source, "directx")


@pytest.mark.parametrize(
    "backend, binding",
    [
        ("metal", "[[buffer(2)]]"),
        ("directx", "register(u2)"),
        ("opengl", "binding = 2"),
    ],
)
def test_narrow_binding_constant_wraps(tmp_path, backend, binding):
    source = _shader(declarations="    const uint8_t SLOT = 258;", binding="SLOT")
    code = _translate(tmp_path, source, backend)
    assert binding in code.replace("binding=", "binding = ")
