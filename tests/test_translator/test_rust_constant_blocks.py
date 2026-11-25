"""CrossGL constant blocks in the rust-gpu output (:mod:`crosstl.translator.codegen.rust_gpu_constant_blocks`).

A ``cbuffer`` becomes a ``#[repr(C)]`` struct that places every member at the byte its block layout (std140, std430
for a push constant, or the ``@buffer_layout`` it names) gives it, and an entry-point parameter bound as a uniform
or push constant; every read of a member reads through that parameter. Each generated crate compiles with rust-gpu.
"""

from __future__ import annotations

import pytest

from crosstl.translator.codegen.rust_refusals import RustLoweringRefused

from .test_codegen.test_rust_codegen import (
    assert_generated_rust_smoke_compiles,
    generate_code,
    parse_code,
    tokenize_code,
)


def _rust(source):
    return generate_code(parse_code(tokenize_code(source)))


def _kernel(block, body, helpers=""):
    return f"""
        shader Probe {{
            {block}
            compute {{
                layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
                buffer float* outp;
                {helpers}
                void main() {{
                    uint i = gl_GlobalInvocationID.x;
                    {body}
                }}
            }}
        }}
        """


def _flat(code):
    return " ".join(code.split())


LOWERINGS = [
    pytest.param(
        "cbuffer Params @binding(3) { float scale; vec3 tint; };",
        "outp[i] = scale + tint.y;",
        [
            "pub struct Params { pub scale: f32,",
            "#[spirv(uniform, descriptor_set = 0, binding = 3)] params: &Params",
            "(params.scale + params.tint.y)",
        ],
        id="explicit-binding",
    ),
    pytest.param(
        # std140 puts the vec4 after the vec3 at offset 16; the vec3 ends at 12, so a u32 fills bytes 12..16.
        "cbuffer Params { vec3 a; vec4 b; };",
        "outp[i] = a.x + b.w;",
        ["pub a: Vec3,", "pub b: Vec4,", "(params.a.x + params.b.w)"],
        id="std140-vec4-after-vec3",
    ),
    pytest.param(
        "cbuffer Params { bool enabled; bvec2 lanes; };",
        "outp[i] = (enabled && lanes.y) ? 1.0 : 0.0;",
        [
            "pub enabled: u32,",
            "pub lanes: UVec2,",
            "(params.enabled != 0)",
            "(params.lanes).cmpne(UVec2::new(0, 0)).y",
        ],
        id="bool-as-u32",
    ),
    pytest.param(
        # std140 gives every element of a scalar array its own 16-byte register.
        "cbuffer Params { float weights[4]; int counts[2]; };",
        "outp[i] = weights[i % 4u] + float(counts[1]);",
        [
            "pub weights: [Vec4; 4],",
            "pub counts: [IVec4; 2],",
            "params.weights[(i % 4) as usize].x",
            "params.counts[1].x",
        ],
        id="std140-scalar-arrays",
    ),
    pytest.param(
        "cbuffer Params { bool flags[3]; };",
        "outp[i] = flags[i % 3u] ? 1.0 : 0.0;",
        ["pub flags: [UVec4; 3],", "(params.flags[(i % 3) as usize].x != 0)"],
        id="std140-bool-array",
    ),
    pytest.param(
        # A std140 mat3 holds each 12-byte column in a 16-byte register.
        "cbuffer Params { mat3 m; };",
        "vec3 r = m * vec3(1.0, 2.0, 3.0); outp[i] = r.z + m[1].y;",
        [
            "pub m: [Vec4; 3],",
            "Mat3::from_cols_array(&[params.m[0].x, params.m[0].y, params.m[0].z, params.m[1].x,",
            "Vec3::new(params.m[1].x, params.m[1].y, params.m[1].z).y",
        ],
        id="std140-mat3-registers",
    ),
    pytest.param(
        "cbuffer Params { mat4 model; vec4 tint; };",
        "vec4 r = model * tint; outp[i] = r.w;",
        ["pub model: Mat4,", "(params.model * params.tint)"],
        id="mat4",
    ),
    pytest.param(
        "cbuffer Params @push_constant { float scale; vec3 tint; };",
        "outp[i] = scale * tint.z;",
        [
            "#[spirv(push_constant)] params: &Params",
            "(params.scale * params.tint.z)",
        ],
        id="push-constant",
    ),
]


@pytest.mark.parametrize("block, body, expected", LOWERINGS)
def test_constant_blocks_lower_to_uniform_structs(block, body, expected, tmp_path):
    code = _rust(_kernel(block, body))
    flat = _flat(code)
    for text in expected:
        assert _flat(text) in flat, text
    assert_generated_rust_smoke_compiles(code, tmp_path)


def test_helper_reads_the_block_through_a_parameter(tmp_path):
    code = _rust(
        _kernel(
            "cbuffer Params { float scale; };",
            "outp[i] = scaled(float(i));",
            helpers="float scaled(float x) { return x * scale; }",
        )
    )
    assert "params.scale" in code
    assert_generated_rust_smoke_compiles(code, tmp_path)


def test_storage_buffers_take_the_lowest_free_bindings(tmp_path):
    code = _rust(
        _kernel("cbuffer Params @binding(0) { float scale; };", "outp[i] = scale;")
    )
    assert "#[spirv(uniform, descriptor_set = 0, binding = 0)] params: &Params" in code
    assert "#[spirv(storage_buffer, descriptor_set = 0, binding = 1)]" in code
    assert_generated_rust_smoke_compiles(code, tmp_path)


@pytest.mark.parametrize(
    "block, body, message",
    [
        pytest.param(
            "cbuffer Params { float weights[]; };",
            "outp[i] = weights[0];",
            "unsized arrays in uniform blocks are not supported",
            id="unsized-array",
        ),
        pytest.param(
            "cbuffer Params { float weights[4]; };",
            "float w[4] = weights; outp[i] = w[0];",
            "is read as a whole array",
            id="whole-array-read",
        ),
        pytest.param(
            "struct Flags { bool on; }; cbuffer Params { Flags flags; };",
            "outp[i] = flags.on ? 1.0 : 0.0;",
            "holds bool values inside an array or struct",
            id="bool-in-struct",
        ),
    ],
)
def test_blocks_the_struct_cannot_hold_are_refused(block, body, message):
    with pytest.raises(RustLoweringRefused, match=message):
        _rust(_kernel(block, body))


def test_declared_compute_builtin_input_is_an_entry_parameter(tmp_path):
    """A stage variable ``@input`` with a compute built-in reads the built-in, not a zero-valued static."""
    code = _rust("""
        shader Probe {
            compute {
                layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
                buffer float* outp;
                uint3 gl_GlobalInvocationID @input @gl_GlobalInvocationID;
                void main() {
                    outp[gl_GlobalInvocationID[0]] = 1.0;
                }
            }
        }
        """)
    assert "#[spirv(global_invocation_id)] gl_GlobalInvocationID: UVec3" in code
    assert "GL_GLOBAL_INVOCATION_ID" not in code
    assert_generated_rust_smoke_compiles(code, tmp_path)


def test_module_scope_stage_input_left_by_the_shared_passes_is_refused():
    """A module-scope ``@input`` that is not a compute built-in has no entry parameter to read, so it is refused."""
    source = """
        shader Probe {
            compute {
                layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
                buffer float* outp;
                float level @input @location(0);
                void main() {
                    outp[0] = level;
                }
            }
        }
        """
    with pytest.raises(RustLoweringRefused, match="module-scope input 'level'"):
        _rust(source)
