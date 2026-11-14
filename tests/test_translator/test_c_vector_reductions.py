"""CUDA and HIP output reduce vectors with ``all`` and ``any`` through generated helpers.

Neither runtime has a device ``all`` or ``any`` for vectors, and nvcc's legacy ``all``/``any`` are warp votes on a
``bool``, so a call left as written either fails to compile or votes across the warp.
"""

import re

import pytest

import crosstl

BACKENDS = ("cuda", "hip")

CROSSGL = """shader P {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;

        void main() {
            uint i = gl_GlobalInvocationID.x;
            vec2 a = vec2(outp[i], 1.0);
            vec2 b = vec2(2.0, 1.0);
            float r = 0.0;
            if (all(a == b)) {
                r += 1.0;
            }
            if (any(a != b)) {
                r += 2.0;
            }
            bvec3 m = bvec3(a.x > 0.0, true, false);
            if (any(m) || all(m)) {
                r += 4.0;
            }
            if (any(bvec3(a.y > 0.0, false, false))) {
                r += 8.0;
            }
            ivec3 iv = ivec3(1, 0, 2);
            if (any(iv)) {
                r += 16.0;
            }
            outp[i] = r;
        }
    }
}
"""

HLSL = """RWStructuredBuffer<float> outp : register(u0);
RWStructuredBuffer<int> iin : register(u1);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    float3 c = float3(outp[i], 0.5, 1.0);
    float r = 0.0;
    if (all(c)) { r += 1.0; }
    if (any(iin[i])) { r += 2.0; }
    if (all(outp[i] > 0.0)) { r += 4.0; }
    half2 h = half2(outp[i], 0.0);
    half3 h3 = half3(h.x, h.y, 1.0);
    if (any(h) && all(h3)) { r += 8.0; }
    if (all(h.x)) { r += 16.0; }
    outp[i] = r;
}
"""


def translate(tmp_path, source, suffix, backend):
    path = tmp_path / f"kernel{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def bare_reductions(code):
    return re.findall(r"(?<![\w.])(?:all|any)\s*\(", code)


@pytest.mark.parametrize("backend", BACKENDS)
def test_boolean_and_integer_vectors_use_component_helpers(tmp_path, backend):
    code = translate(tmp_path, CROSSGL, ".cgl", backend)
    assert bare_reductions(code) == []
    assert "cgl_uchar2_all(make_uchar2((a.x == b.x), (a.y == b.y)))" in code
    assert "cgl_uchar2_any(make_uchar2((a.x != b.x), (a.y != b.y)))" in code
    assert "(cgl_uchar3_any(m) || cgl_uchar3_all(m))" in code
    assert "cgl_int3_any(iv)" in code
    assert (
        "__device__ inline bool cgl_uchar3_all(uchar3 value)\n{\n"
        "    return (value.x != 0) && (value.y != 0) && (value.z != 0);\n}"
    ) in code
    assert (
        "__device__ inline bool cgl_uchar2_any(uchar2 value)\n{\n"
        "    return (value.x != 0) || (value.y != 0);\n}"
    ) in code
    # One helper per vector type and reduction, however many calls use it.
    assert code.count("inline bool cgl_uchar3_any(") == 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_float_half_and_scalar_operands(tmp_path, backend):
    code = translate(tmp_path, HLSL, ".hlsl", backend)
    assert bare_reductions(code) == []
    assert "cgl_float3_all(c)" in code
    assert "((iin[i]) != 0)" in code
    assert "cgl_half2_any(h) && cgl_half3_all(h3)" in code
    # __half compares with __half only, so each lane is tested as float.
    assert (
        "__device__ inline bool cgl_half2_any(half2 value)\n{\n"
        "    return (static_cast<float>(value.x) != 0.0f) || "
        "(static_cast<float>(value.y) != 0.0f);\n}"
    ) in code
    assert "(static_cast<float>((h.x)) != 0.0f)" in code


@pytest.mark.parametrize("backend", BACKENDS)
def test_operand_without_a_component_test_is_refused(tmp_path, backend):
    source = """shader P {
    struct Pair {
        float a;
        float b;
    }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            Pair p = Pair(outp[i], 1.0);
            if (any(p)) {
                outp[i] = 1.0;
            }
        }
    }
}
"""
    with pytest.raises(
        ValueError, match=r"cannot lower any\(\) on an operand of type Pair"
    ):
        translate(tmp_path, source, ".cgl", backend)
