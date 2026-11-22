"""CrossGL built-ins outside GLSL's core set in the rust-gpu output (:mod:`crosstl.translator.codegen.rust_gpu_math_calls`).

C and Metal math functions, the matrix, geometric and packing functions, float constants, the subgroup barriers and
the helper-invocation test are written as num-traits, glam and spirv_std calls that compute what CrossGL defines, and
every generated crate compiles with rust-gpu. Operand types they have no Rust form for are refused by name.
"""

from __future__ import annotations

import pytest

from crosstl.translator.codegen.rust_refusals import RustLoweringRefused

from .test_rust_codegen import (
    assert_generated_rust_smoke_compiles,
    generate_code,
    parse_code,
    tokenize_code,
)


def _rust(source):
    return generate_code(parse_code(tokenize_code(source)))


def _kernel(body, declarations=""):
    return f"""
        shader Probe {{
            compute {{
                layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
                buffer float* fin;
                buffer int* iin;
                buffer uint* uin;
                buffer float* outp;
                {declarations}
                void main() {{
                    uint i = gl_GlobalInvocationID.x;
                    float x = fin[i];
                    float y = fin[i + 1u];
                    vec3 v = vec3(fin[i], fin[i + 1u], fin[i + 2u]);
                    vec3 w = vec3(fin[i + 3u], fin[i + 4u], fin[i + 5u]);
                    int a = iin[i];
                    int b = iin[i + 1u];
                    uint p = uin[i];
                    {body}
                }}
            }}
        }}
        """


MATH_KERNEL = _kernel("""
    float sum = expm1(x) + log1p(x) + log10(x) + rcp(x) + cbrt(x) + copysign(x, y);
    vec3 vsum = expm1(v) + rcp(v) + cbrt(v) + copysign(v, w) + pow(v, y);
    bool negative = signbit(x);
    bvec3 negatives = signbit(v);
    float squares = length_squared(v) + distance_squared(v, w) + length_squared(x) + distance_squared(x, y);
    uint gap = absdiff(a, b) + absdiff(p, 3u);
    uvec3 gaps = absdiff(ivec3(a), ivec3(b));
    vec3 bent = refract(v, w, y);
    float bentScalar = refract(x, y, 0.5);
    vec3 picked = select(negatives, v, w);
    vec3 pickedLast = select(w, v, negatives);
    float pickedScalar = select(negative, x, y);
    float s;
    float c;
    sincos(x, s, c);
    vec3 vs;
    vec3 vc;
    sincos(v, vs, vc);
    mat3 m = mat3(v, w, vec3(1.0, 0.0, 0.0));
    mat3 inv = inverse(m);
    float det = determinant(m);
    uint half2 = packHalf2x16(vec2(x, y));
    vec2 unpacked = unpackHalf2x16(half2) + unpackSnorm2x16(p) + unpackUnorm2x16(p);
    vec4 bytes = unpackSnorm4x8(p) + unpackUnorm4x8(p);
    uint packed = packSnorm2x16(unpacked) + packUnorm2x16(unpacked) + packSnorm4x8(bytes) + packUnorm4x8(bytes);
    float limits = nan() + infinity() + epsilon();
    subgroupBarrier();
    subgroupMemoryBarrier();
    subgroupMemoryBarrierBuffer();
    subgroupMemoryBarrierShared();
    subgroupMemoryBarrierImage();
    outp[i] = sum + vsum.x + squares + float(gap + gaps.x + half2 + packed) + bent.x + bentScalar + picked.y
        + pickedLast.z + pickedScalar + s + c + vs.x + vc.y + inv[0].x + det + unpacked.x + bytes.w + limits
        + (negative ? 1.0 : 0.0) + (negatives.y ? 1.0 : 0.0);
""")


@pytest.mark.parametrize(
    "spelling",
    [
        "(x).exp_m1()",
        "(x).ln_1p()",
        "(x).log10()",
        "(x).recip()",
        "{ let c: f32 = x; c.abs().powf(1.0 / 3.0).copysign(c) }",
        "(x).copysign(y)",
        "(v).map(|c| c.exp_m1())",
        "(v).recip()",
        "(v).map(|c| c.abs().powf(1.0 / 3.0).copysign(c))",
        "(v).copysign(w)",
        "(v).powf(y)",
        "let negative: bool = (x).is_sign_negative();",
        "BVec3::new(__v.x.is_sign_negative(), __v.y.is_sign_negative(), __v.z.is_sign_negative())",
        "(v).length_squared()",
        "(v).distance_squared(w)",
        "{ let __v = x; __v * __v }",
        "{ let __d = (x) - (y); __d * __d }",
        "(a).abs_diff(b)",
        "(p).abs_diff(3)",
        "UVec3::new(__a.x.abs_diff(__b.x), __a.y.abs_diff(__b.y), __a.z.abs_diff(__b.z))",
        "if __k < 0.0 { Vec3::ZERO } else { __e * __i - (__e * __d + __k.sqrt()) * __n }",
        "let __d = __n.dot(__i);",
        "let __d = __n * __i;",
        "{ let __c = negatives; let __t = v; let __f = w; Vec3::select(__c, __t, __f) }",
        "{ let __c = negatives; let __t = v; let __f = w; Vec3::select(__c, __t, __f) }",
        "{ let __c = negative; let __t = x; let __f = y; if __c { __t } else { __f } }",
        "{ let (__s, __c) = (x).sin_cos(); s = __s; c = __c; }",
        "{ let __x = v; vs = __x.map(|c| c.sin()); vc = __x.map(|c| c.cos()); }",
        "(m).inverse()",
        "(m).determinant()",
        "spirv_std::float::vec2_to_f16x2(",
        "spirv_std::float::f16x2_to_vec2(half2)",
        "spirv_std::float::u16x2_to_vec2_snorm(p)",
        "spirv_std::float::u16x2_to_vec2_unorm(p)",
        "spirv_std::float::u8x4_to_vec4_snorm(p)",
        "spirv_std::float::u8x4_to_vec4_unorm(p)",
        "spirv_std::float::vec2_to_u16x2_snorm(unpacked)",
        "spirv_std::float::vec2_to_u16x2_unorm(unpacked)",
        "spirv_std::float::vec4_to_u8x4_snorm(bytes)",
        "spirv_std::float::vec4_to_u8x4_unorm(bytes)",
        "((f32::NAN + f32::INFINITY) + f32::EPSILON)",
        "spirv_std::arch::subgroup_barrier();",
        "spirv_std::arch::subgroup_memory_barrier();",
        "spirv_std::arch::subgroup_memory_barrier_buffer();",
        "spirv_std::arch::subgroup_memory_barrier_shared();",
        "spirv_std::arch::subgroup_memory_barrier_image();",
    ],
)
def test_math_calls_are_written_as_rust_gpu_calls(spelling):
    code = _rust(MATH_KERNEL)
    assert spelling in code, code


def test_math_calls_compile_with_rust_gpu(tmp_path):
    code = _rust(MATH_KERNEL)
    # sincos assigns its outputs, which are therefore mutable bindings.
    assert "let mut s: f32" in code and "let mut vc: Vec3" in code, code
    assert_generated_rust_smoke_compiles(code, tmp_path)


def test_select_with_the_condition_last_picks_the_middle_operand():
    code = _rust(_kernel("outp[i] = select(x, y, x > y);"))
    assert (
        "{ let __c = (x > y); let __t = y; let __f = x; if __c { __t } else { __f } }"
        in code
    ), code


def test_literal_receivers_are_typed():
    code = _rust(_kernel("outp[i] = rcp(2.0) + copysign(1.0, x);"))
    assert "((2.0 as f32)).recip()" in code, code
    assert "((1.0 as f32)).copysign(x)" in code, code


def test_helper_lane_test_in_a_fragment_shader(tmp_path):
    code = _rust("""
        shader Helper {
            struct FragmentInput { vec2 uv; };
            struct FragmentOutput { vec4 color; };
            fragment {
                buffer int* outp;
                FragmentOutput main(FragmentInput input) {
                    FragmentOutput output;
                    outp[0] = IsHelperLane() ? 1 : 0;
                    output.color = vec4(input.uv, 0.0, 1.0);
                    return output;
                }
            }
        }
        """)
    assert "spirv_std::arch::is_helper_invocation()" in code, code
    assert_generated_rust_smoke_compiles(code, tmp_path)


@pytest.mark.parametrize(
    ("body", "function"),
    [
        # Three bool operands do not say which of select's orders is meant.
        (
            "bool t = select(x > 0.0, y > 0.0, x > y); outp[i] = t ? 1.0 : 0.0;",
            "select",
        ),
        ("outp[i] = absdiff(x, y);", "absdiff"),
        ("outp[i] = cbrt(a);", "cbrt"),
        ("outp[i] = determinant(mat2x3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0));", "determinant"),
        ("outp[i] = refract(v, w, v).x;", "refract"),
        ("outp[i] = float(packHalf2x16(v.xy, 1.0));", "packHalf2x16"),
        ("outp[i] = IsHelperLane() ? 1.0 : 0.0;", "IsHelperLane"),
    ],
)
def test_operands_without_a_rust_form_are_refused(body, function):
    with pytest.raises(RustLoweringRefused) as refused:
        _rust(_kernel(body))
    assert refused.value.code == "project.translate.rust-lowering-unsupported"
    assert (
        f"function: '{function}' has no rust-gpu lowering for these operands"
        in refused.value.reasons
    ), refused.value.reasons
