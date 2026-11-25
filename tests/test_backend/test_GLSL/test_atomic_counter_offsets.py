"""GLSL input writes the buffer offset of every atomic counter that does not start its binding's buffer.

GLSL 4.60 section 4.4.6.3: a counter without ``offset`` takes the next offset of its binding, each counter moves that
offset 4 bytes per element past its start, and a type-only ``layout(binding = B, offset = O) uniform atomic_uint;``
moves it to ``O``. CrossGL keeps neither the defaults nor the declaration order across stages, so the importer
resolves them (:mod:`crosstl.backend.GLSL.atomic_counter_offsets`).
"""

from __future__ import annotations

import textwrap

import pytest

from crosstl.backend.GLSL.atomic_counter_offsets import literal
from crosstl.backend.GLSL.openglCrossglCodegen import GLSLToCrossGLConverter
from crosstl.backend.GLSL.OpenglLexer import GLSLLexer
from crosstl.backend.GLSL.OpenglParser import GLSLParser


def generate_crossgl(code, shader_type="vertex"):
    tokens = GLSLLexer(textwrap.dedent(code).strip()).tokenize()
    ast = GLSLParser(tokens, shader_type).parse()
    return GLSLToCrossGLConverter(shader_type=shader_type).generate(ast)


def counter_lines(crossgl):
    return [line.strip() for line in crossgl.splitlines() if "atomic_uint" in line]


def test_implicit_offsets_follow_defaults_and_arrays():
    crossgl = generate_crossgl("""
        #version 430
        layout (binding = 0, offset = 4) uniform atomic_uint;
        layout (binding = 0) uniform atomic_uint bar, q;
        layout (binding = 0) uniform atomic_uint baz[2][3];
        layout (binding = 0, offset = 0x40) uniform atomic_uint z;
        layout (binding = 0) uniform atomic_uint w;
        layout (binding = 1) uniform atomic_uint other;
        void main() {
            gl_Position = vec4(float(atomicCounterIncrement(bar) + atomicCounter(baz[1][2])
                + atomicCounter(q) + atomicCounter(z) + atomicCounter(w) + atomicCounter(other)));
        }
    """)

    assert counter_lines(crossgl) == [
        "atomic_uint bar @binding(0) @offset(4);",
        "atomic_uint q @binding(0) @offset(8);",
        "atomic_uint baz[2][3] @binding(0) @offset(12);",
        "atomic_uint z @binding(0) @offset(64);",
        "atomic_uint w @binding(0) @offset(68);",
        "atomic_uint other @binding(1);",
    ]


def test_each_binding_keeps_its_own_next_offset():
    crossgl = generate_crossgl("""
        #version 430
        layout (binding = 1) uniform atomic_uint a;
        layout (binding = 2) uniform atomic_uint b;
        layout (binding = 1) uniform atomic_uint c;
        layout (binding = 2, offset = 8) uniform atomic_uint;
        layout (binding = 2) uniform atomic_uint d;
        void main() {
            gl_Position = vec4(float(atomicCounter(a) + atomicCounter(b) + atomicCounter(c) + atomicCounter(d)));
        }
    """)

    assert counter_lines(crossgl) == [
        "atomic_uint a @binding(1);",
        "atomic_uint b @binding(2);",
        "atomic_uint c @binding(1) @offset(4);",
        "atomic_uint d @binding(2) @offset(8);",
    ]
    assert "uniform atomic_uint;" not in crossgl


def test_explicit_zero_offset_is_kept():
    crossgl = generate_crossgl("""
        #version 430
        layout (binding = 0, offset = 0) uniform atomic_uint a;
        void main() { gl_Position = vec4(float(atomicCounter(a))); }
    """)

    assert counter_lines(crossgl) == ["atomic_uint a @binding(0) @offset(0);"]


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("4", 4),
        ("0x10", 16),
        ("0X1f", 31),
        ("010", 8),
        ("0", 0),
        ("12u", 12),
        ("8U", 8),
    ],
)
def test_literal_reads_glsl_integer_literals(text, value):
    assert literal(text) == value


@pytest.mark.parametrize("text", ["K", "4.0", "", "1 + 2", None])
def test_literal_rejects_other_values(text):
    assert literal(text) is None
