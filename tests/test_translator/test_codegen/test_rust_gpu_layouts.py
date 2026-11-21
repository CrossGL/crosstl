"""rust-gpu buffer bindings read the std430 (storage) and std140 (uniform) layouts the CrossGL program declares.

rustc_codegen_spirv decorates each member of a buffer structure with its ``#[repr(C)]`` Rust offset, and glam's
types are aligned to their component except ``Vec4``, ``Mat2`` and ``Mat4`` (16 bytes). A structure whose Rust
offsets differ from the std430/std140 offsets would read the wrong bytes, so the binding is refused by member.
"""

import pytest

from crosstl.translator.codegen.rust_refusals import RustLoweringRefused

from .test_rust_codegen import (
    assert_generated_rust_smoke_compiles,
    generate_code,
    parse_code,
    tokenize_code,
)


def rust(code):
    return generate_code(parse_code(tokenize_code(code)))


def test_matching_storage_and_uniform_structures_bind_and_compile(tmp_path):
    code = rust("""
    shader Layouts {
        struct Particle {
            vec4 position;
            vec3 velocity;
            float life;
            vec2 uv;
            uint id;
            float spare;
            mat4 transform;
        };
        struct Params {
            vec4 tint;
            vec3 direction;
            float scale;
        };
        RWStructuredBuffer<Particle> particles;
        uniform Params params;
        compute {
            layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
            void main() {
                uint i = gl_GlobalInvocationID.x;
                Particle p = particles[i];
                p.position = p.position + params.tint * params.scale;
                p.life = p.life - 1.0;
                particles[i] = p;
            }
        }
    }
    """)
    assert "particles: &mut [Particle]" in code
    assert "params: &Params" in code
    assert_generated_rust_smoke_compiles(code, tmp_path)


@pytest.mark.parametrize(
    "declarations, body, reason",
    [
        (
            "struct Q { float a; vec2 b; };\nbuffer Q* qs;",
            "qs[0].a = qs[1].b.x;",
            "storage buffer 'qs': 'Q.b' is at std430 offset 8, rust-gpu lays it out at offset 4",
        ),
        (
            "struct V { vec3 p; vec3 v; };\nRWStructuredBuffer<V> vs;",
            "vs[0] = vs[1];",
            "storage buffer 'vs': 'V.v' is at std430 offset 16, rust-gpu lays it out at offset 12",
        ),
        (
            "struct W { float a; uvec4 b; };\nbuffer W* ws;",
            "ws[0].a = 1.0;",
            "storage buffer 'ws': 'W.b' is at std430 offset 16, rust-gpu lays it out at offset 4",
        ),
        (
            "buffer mat3* ms;",
            "ms[0] = ms[1];",
            "storage buffer 'ms': 'mat3 column 1' is at std430 offset 16, rust-gpu lays it out at offset 12",
        ),
        (
            "uniform float weights[4];\nbuffer float* outp;",
            "outp[0] = weights[1];",
            "uniform 'weights': 'float[4]' has the std140 array stride 16, rust-gpu lays it out with stride 4",
        ),
        (
            "struct B { bool flag; };\nbuffer B* bs;",
            "bs[0] = bs[1];",
            "storage buffer 'bs': 'B.flag' is a bool; a rust-gpu buffer cannot hold a Boolean",
        ),
    ],
)
def test_mismatched_buffer_layouts_are_refused_by_member(declarations, body, reason):
    with pytest.raises(RustLoweringRefused) as refused:
        rust(f"""
        shader Layouts {{
            {declarations}
            compute {{
                void main() {{
                    {body}
                }}
            }}
        }}
        """)
    assert f"resource: {reason}" in refused.value.reasons


def test_unreferenced_buffer_with_a_mismatched_layout_is_not_bound():
    code = rust("""
    shader Layouts {
        struct V { vec3 p; vec3 v; };
        RWStructuredBuffer<V> vs;
        compute {
            void main() {}
        }
    }
    """)
    assert "pub struct V {" in code
    assert "pub fn main() -> ()" in code


def test_vertex_vec4_result_without_a_semantic_is_the_position(tmp_path):
    code = rust("""
    shader Positions {
        vertex {
            vec4 main(vec3 position @ POSITION) {
                return vec4(position, 1.0);
            }
        }
    }
    """)
    assert "#[spirv(position)] out_value: &mut Vec4" in code
    assert_generated_rust_smoke_compiles(code, tmp_path)


def test_vertex_result_of_another_type_without_a_semantic_is_refused():
    with pytest.raises(RustLoweringRefused, match="returns a vec3 without a semantic"):
        rust("""
        shader Positions {
            vertex {
                vec3 main(vec3 position @ POSITION) {
                    return position;
                }
            }
        }
        """)
