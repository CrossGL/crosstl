"""Metal passes array arguments outside thread storage to array parameters as thread copies.

A by-value array parameter is a ``thread`` pointer in Metal. Constant-buffer members, program-scope constants,
threadgroup arrays and buffer arrays used to be passed as they are and failed with "no matching function for
call", and constant-buffer arrays stored in another shape (``float4`` slots for std140 ``float[3]``, register
words for row-major matrices) could not be passed at all. They are now copied element by element into a
temporary ``crosstl_array_value<T, N>`` whose array member the call receives.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused

CROSSGL = textwrap.dedent("""
    shader S {
        const float K[3] = {2.0, 3.0, 5.0};
        cbuffer U @binding(0) @buffer_layout(std140) {
            vec4 v[2];
            float w[3];
            mat2x3 m[2];
            mat2x3 q[2] @row_major;
        };
        struct O {
            float o[8];
        };
        O outp @glsl_buffer_block(std430) @binding(1);
        float pickv(vec4 a[2]) { return a[0].x + a[1].y; }
        float pickw(float a[3]) { return a[0] + a[2]; }
        float pickm(mat2x3 a[2]) { return a[0][1][1] + a[1][0][2]; }
        float forward(vec4 a[2]) { return pickv(a); }
        float viaconstant(constant float a[3]) { return pickc(a); }
        float pickc(const float a[3]) { return a[1]; }
        compute {
            layout(local_size_x = 1) in;
            shared vec4 sh[2];
            void main() {
                vec4 t[2] = {vec4(1.0), vec4(2.0)};
                constant float c[3] = {1.0, 2.0, 3.0};
                sh[0] = v[0];
                sh[1] = v[1];
                outp.o[0] = pickv(v);
                outp.o[1] = pickw(w);
                outp.o[2] = pickm(m);
                outp.o[3] = pickm(q);
                outp.o[4] = pickv(sh);
                outp.o[5] = pickw(K);
                outp.o[6] = pickv(t) + forward(t) + pickc(c);
                outp.o[7] = pickw({1.0, 2.0, 3.0});
            }
        }
    }
    """)

HLSL = textwrap.dedent("""
    struct L { float3 p; float r; };
    struct E { float4 a[2]; };
    cbuffer U : register(b0) {
        L lights[2];
        float w[3];
    };
    StructuredBuffer<E> sb : register(t0);
    RWStructuredBuffer<float> outp : register(u1);
    groupshared float gs[4];
    float lsum(L a[2]) { return a[0].p.y + a[1].r; }
    float wsum(float a[3]) { return a[0] + a[2]; }
    float esum(float4 a[2]) { return a[0].x + a[1].z; }
    float gsum(float a[4]) { return a[0] + a[3]; }
    [numthreads(1, 1, 1)]
    void CSMain(uint3 id : SV_DispatchThreadID) {
        gs[id.x] = 1.0;
        GroupMemoryBarrierWithGroupSync();
        outp[0] = lsum(lights) + wsum(w) + esum(sb[0].a) + gsum(gs);
    }
    """)


def _translate(tmp_path, source, suffix=".cgl"):
    path = tmp_path / f"arguments{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="metal")


def _compact(text):
    """``text`` without whitespace, so assertions do not depend on line breaking."""
    return "".join(text.split())


def _contains(code, expected):
    return _compact(expected) in _compact(code)


def _compile(tmp_path, code):
    if shutil.which("xcrun") is None:
        pytest.skip("xcrun is not installed")
    (tmp_path / "arguments.metal").write_text(code, encoding="utf-8")
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", "arguments.metal", "-o", "a.air"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=tmp_path,
    )
    if "unable to find utility" in result.stderr:
        pytest.skip("the Metal toolchain is not installed")
    assert result.returncode == 0, result.stderr[-2000:]


def test_constant_buffer_arrays_are_passed_as_thread_copies(tmp_path):
    code = _translate(tmp_path, CROSSGL)
    assert _contains(code, "pickv(crosstl_array_value<float4, 2>{{u.v[0], u.v[1]}}.v)")
    # std140 stores each float in a float4 slot; the copy reads the first component.
    assert _contains(
        code, "pickw(crosstl_array_value<float, 3>{{u.w[0].x, u.w[1].x, u.w[2].x}}.v)"
    )
    assert _contains(
        code, "pickm(crosstl_array_value<float2x3, 2>{{u.m[0], u.m[1]}}.v)"
    )
    assert code.count("struct crosstl_array_value") == 1


def test_row_major_constant_buffer_arrays_are_loaded_element_by_element(tmp_path):
    code = _translate(tmp_path, CROSSGL)
    load = "crosstl_hlsl_cbuffer_float2x3_row_major_padded_load"
    assert _contains(
        code,
        f"pickm(crosstl_array_value<float2x3, 2>{{{{{load}(u.q[0]), {load}(u.q[1])}}}}.v)",
    )


def test_threadgroup_constant_and_literal_arrays_are_passed_as_thread_copies(
    tmp_path,
):
    code = _translate(tmp_path, CROSSGL)
    assert _contains(code, "pickv(crosstl_array_value<float4, 2>{{sh[0], sh[1]}}.v)")
    assert _contains(code, "pickw(crosstl_array_value<float, 3>{{K[0], K[1], K[2]}}.v)")
    assert _contains(code, "pickw(crosstl_array_value<float, 3>{{1.0, 2.0, 3.0}}.v)")
    # A constant-qualified parameter is a constant pointer, so forwarding it copies too.
    assert _contains(code, "pickc(crosstl_array_value<float, 3>{{a[0], a[1], a[2]}}.v)")


def test_thread_arrays_keep_pointer_passing(tmp_path):
    code = _translate(tmp_path, CROSSGL)
    assert _contains(code, "pickv(t)")
    assert _contains(code, "forward(t)")
    assert _contains(code, "return pickv(a);")
    # A function-scope constant array is a const array in thread storage.
    assert _contains(code, "pickc(c)")


def test_crossgl_output_compiles(tmp_path):
    _compile(tmp_path, _translate(tmp_path, CROSSGL))


def test_hlsl_constant_structured_and_groupshared_arrays_are_copied(tmp_path):
    code = _translate(tmp_path, HLSL, ".hlsl")
    assert _contains(
        code, "lsum(crosstl_array_value<L, 2>{{u.lights[0], u.lights[1]}}.v)"
    )
    assert _contains(
        code, "wsum(crosstl_array_value<float, 3>{{u.w[0].x, u.w[1].x, u.w[2].x}}.v)"
    )
    assert _contains(
        code,
        "esum(crosstl_array_value<float4, 2>{{float4(sb[0].a[0]), float4(sb[0].a[1])}}.v)",
    )
    assert _contains(
        code, "gsum(crosstl_array_value<float, 4>{{gs[0], gs[1], gs[2], gs[3]}}.v)"
    )
    _compile(tmp_path, code)


def test_programs_without_copies_omit_the_template(tmp_path):
    code = _translate(
        tmp_path,
        textwrap.dedent("""
            shader S {
                float pick(float a[2]) { return a[1]; }
                compute {
                    buffer float data[];
                    layout(local_size_x = 1) in;
                    void main() {
                        float t[2] = {1.0, 2.0};
                        data[0] = pick(t);
                    }
                }
            }
            """),
    )
    assert "crosstl_array_value" not in code


def test_arguments_with_effects_are_refused(tmp_path):
    source = textwrap.dedent("""
        shader S {
            float pick(float a[3]) { return a[1]; }
            compute {
                buffer float data[];
                layout(local_size_x = 1) in;
                shared float g[2][3];
                void main() {
                    int k = 0;
                    data[0] = pick(g[k++]);
                }
            }
        }
        """)
    with pytest.raises(MetalLoweringRefused, match="array argument to 'pick'"):
        _translate(tmp_path, source)
