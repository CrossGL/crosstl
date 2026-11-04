"""A type alias used as a vector's element type names the aliased scalar (#3748).

``vec2<real>`` for ``typedef float real;`` used to be written ``real2`` by the Metal, HLSL and GLSL generators and
``vec2<real>`` by the WGSL generator, which no compiler accepts.
"""

import textwrap

import pytest

import crosstl
from crosstl.translator.semantics import TypeAliasUnsupportedError

PROGRAM_ALIAS = textwrap.dedent("""
    shader P {
        typedef float real;
        compute {
            buffer float* outp;
            void main() {
                vec2<real> d = vec2<real>(outp[1]);
                outp[0] = d.x;
            }
        }
    }
""")

BLOCK_ALIAS = textwrap.dedent("""
    shader P {
        compute {
            buffer float* outp;
            void main() {
                typedef int count_t;
                vec3<count_t> c = vec3<count_t>(int(outp[1]));
                outp[0] = float(c.x);
            }
        }
    }
""")

NOT_A_SCALAR = textwrap.dedent("""
    shader P {
        typedef vec2 pair;
        compute {
            buffer float* outp;
            void main() {
                vec2<pair> d;
                outp[0] = 1.0;
            }
        }
    }
""")

EXPECTED = {
    "metal": ["float2 d = float2(outp[1]);", "int3 c = int3(int(outp[1]));"],
    "directx": ["float2 d = float2(", "int3 c = int3("],
    "opengl": ["vec2 d = vec2(outp[1]);", "ivec3 c = ivec3(int(outp[1]));"],
    "wgsl": [
        "var d: vec2<f32> = vec2<f32>(outp[1]);",
        "var c: vec3<i32> = vec3<i32>(i32(outp[1]));",
    ],
}


def translate(tmp_path, source, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize("backend", sorted(EXPECTED))
def test_program_alias_as_vector_element_type_is_resolved(tmp_path, backend):
    output = translate(tmp_path, PROGRAM_ALIAS, backend)
    assert EXPECTED[backend][0] in output
    assert "real2" not in output
    assert "vec2<real>" not in output


@pytest.mark.parametrize("backend", sorted(EXPECTED))
def test_block_alias_as_vector_element_type_is_resolved(tmp_path, backend):
    output = translate(tmp_path, BLOCK_ALIAS, backend)
    assert EXPECTED[backend][1] in output
    assert "count_t" not in output


@pytest.mark.parametrize("source", [PROGRAM_ALIAS, BLOCK_ALIAS])
def test_alias_as_vector_element_type_translates_to_spirv(tmp_path, source):
    translate(tmp_path, source, "spirv")


@pytest.mark.parametrize("backend", sorted(EXPECTED) + ["spirv"])
def test_alias_of_a_vector_as_vector_element_type_is_refused(tmp_path, backend):
    with pytest.raises(TypeAliasUnsupportedError, match="'pair' is used as a vector"):
        translate(tmp_path, NOT_A_SCALAR, backend)


SIZED_ALIASES = textwrap.dedent("""
    shader P {
        typedef i64 long_t;
        typedef u16 short_t;
        compute {
            buffer float* outp;
            void main() {
                vec2<long_t> a = vec2<long_t>(1);
                vec3<short_t> b;
                b.x = 1u;
                outp[0] = float(a.x) + float(b.x);
            }
        }
    }
""")


@pytest.mark.parametrize("backend", ["directx", "metal", "opengl", "spirv"])
def test_alias_of_a_sized_integer_as_vector_element_type_is_resolved(tmp_path, backend):
    # ``vec2<long_t>`` for ``typedef i64 long_t;`` is ``vec2<i64>``, as written by hand.
    by_hand = (
        SIZED_ALIASES.replace("    typedef i64 long_t;\n", "")
        .replace("    typedef u16 short_t;\n", "")
        .replace("long_t", "i64")
        .replace("short_t", "u16")
    )
    assert translate(tmp_path, SIZED_ALIASES, backend) == translate(
        tmp_path, by_hand, backend
    )


WGSL_SPELLING_ALIASES = textwrap.dedent("""
    shader P {
        typedef f32 r32;
        typedef i32 s32;
        compute {
            buffer float* outp;
            void main() {
                vec2<r32> d = vec2<r32>(outp[1]);
                vec3<s32> c;
                c.x = 1;
                outp[0] = d.x + float(c.x);
            }
        }
    }
""")


@pytest.mark.parametrize(
    "backend", ["cuda", "directx", "hip", "metal", "opengl", "spirv", "wgsl"]
)
def test_alias_of_a_wgsl_scalar_spelling_as_vector_element_type_is_resolved(
    tmp_path, backend
):
    # The parser reads ``vec2<f32>`` as a vector of ``float``; CUDA output wrote ``f322`` for a vector of ``f32``.
    by_hand = (
        WGSL_SPELLING_ALIASES.replace("    typedef f32 r32;\n", "")
        .replace("    typedef i32 s32;\n", "")
        .replace("r32", "f32")
        .replace("s32", "i32")
    )
    assert translate(tmp_path, WGSL_SPELLING_ALIASES, backend) == translate(
        tmp_path, by_hand, backend
    )
