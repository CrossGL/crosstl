"""HLSL output writes CrossGL brace lists for structures and matrices in forms DXC accepts.

- A structure that a brace list leaves partly or wholly zero, as ``S s = {};``, zeroes its array members element by
  element: ``s.w = {float(0), ...};`` is no HLSL statement (#3734).
- A short brace list for a structure constant, ``const W C = {{1.0, 2.0}};``, is an initializer list with the
  omitted members zero instead of a constructor call that takes a brace list (#3716).
- A matrix brace list of column vectors, ``mat2 m = {vec2(1.0, 2.0), vec2(3.0, 4.0)}``, gives one column per vector
  (GLSL 4.60 section 4.1.11) instead of counting each vector as one component and padding with zeros (#3718).

Each program stores constants, so DXC folds the stored value and the DXIL listing shows the result.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.directx_values import DirectXAggregateInitializerError

ZERO_FILLED = """shader Probe {
    buffer float* outp;
    struct Inner { vec4 v; };
    struct S { float w[3]; int n; Inner parts[2]; float big[6]; float grid[2][3]; };
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            S s = {};
            S t = {{1.0, 2.0, 3.0}, 4};
            s.w[1] = 5.0;
            outp[i] = s.w[0] + s.w[1] + float(s.n) + s.parts[1].v.y + s.big[5] + s.grid[1][2]
                + t.w[2] * 10.0 + float(t.n) * 100.0 + t.parts[1].v.y + t.big[5] + t.grid[1][2];
        }
    }
}
"""

CUDA_ZERO_FILLED = """struct S { float w[3]; int n; };

extern "C" __global__ void first(float* out) {
    S s = {};
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = s.w[i % 3] + (float)s.n;
}
"""

SHORT_CONSTANT = """shader Probe {
    buffer float* outp;
    struct W { float m[2]; float s; };
    const W C = {{1.0, 2.0}};
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = C.m[1] + C.s * 10.0 + C.m[0] * 100.0;
        }
    }
}
"""

MATRIX_COLUMNS = """shader Probe {
    buffer float* outp;
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            mat2 m = {vec2(1.0, 2.0), vec2(3.0, 4.0)};
            mat2x3 n = {vec3(5.0, 6.0, 7.0), vec3(8.0)};
            mat3 p = {vec3(1.0, 2.0, 3.0)};
            mat2 a[2] = {mat2(1.0), {vec2(9.0, 10.0), vec2(11.0, 12.0)}};
            mat2 f = {{1.0, 2.0}, {3.0, 4.0}};
            mat2 g = {1.0, 2.0, 3.0, 4.0};
            outp[i] = m[1][0] + m[0][1] * 10.0 + n[1][2] * 100.0 + n[0][2] * 1000.0
                + p[0][2] * 10000.0 + p[1][1] + p[2][2] + a[1][1][0] * 100000.0
                + (f[1][0] - g[1][0]) + (f[0][1] - g[0][1]);
        }
    }
}
"""

MIXED_TYPES = """shader Probe {
    buffer float* outp;
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            int k = int(i) + 2;
            float x = outp[i];
            mat2 q = {ivec2(k), vec2(x)};
            mat4 r = {vec4(x), vec4(1.0), vec4(2.0), vec4(3.0)};
            outp[i] = q[0][1] + q[1][0] + r[3][2];
        }
    }
}
"""


def _hlsl(tmp_path, program, name="probe.cgl"):
    path = tmp_path / name
    path.write_text(program)
    return crosstl.translate(str(path), backend="directx")


def _dxil(tmp_path, code):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not on PATH")
    source = tmp_path / "translated.hlsl"
    source.write_text(code)
    listing = tmp_path / "translated.ll"
    result = subprocess.run(
        [dxc, "-T", "cs_6_6", "-E", "CSMain", "-WX", str(source), "-Fc", str(listing)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    return listing.read_text()


def _stored_constant(listing):
    stored = re.findall(r"rawBufferStore\.f32\(.*?, float ([-+.e0-9]+),", listing)
    assert len(stored) == 1, stored
    return float(stored[0])


def test_omitted_array_members_are_zeroed_element_by_element(tmp_path):
    code = _hlsl(tmp_path, ZERO_FILLED)

    assert not re.search(r"\.\w+ = \{", code)
    assert "s.w[0] = float(0);" in code
    assert "s.parts[1] = (Inner)0;" in code
    assert "s.grid[1][2] = float(0);" in code
    # Six elements are more than the unrolled limit, so a loop zeroes them.
    assert re.search(r"for \(uint (\w+) = 0u; \1 < 6u; \+\+\1\)", code)


def test_zero_filled_structures_reach_dxil(tmp_path):
    listing = _dxil(tmp_path, _hlsl(tmp_path, ZERO_FILLED))

    # s: w[1] = 5, every other member zero; t: w[2] = 3 and n = 4, every omitted member zero.
    assert _stored_constant(listing) == 5.0 + 30.0 + 400.0


def test_cuda_zero_initialized_structure_compiles(tmp_path):
    code = _hlsl(tmp_path, CUDA_ZERO_FILLED, name="probe.cu")

    assert "s.w = {" not in code
    _dxil(tmp_path, code)


def test_short_structure_constant_is_an_initializer_list(tmp_path):
    code = _hlsl(tmp_path, SHORT_CONSTANT)

    assert "__crossgl_construct_W({" not in code
    assert "static const W C = {{1.0, 2.0}, float(0)};" in code
    # C.m[1] + C.s * 10 + C.m[0] * 100 = 2 + 0 + 100.
    assert _stored_constant(_dxil(tmp_path, code)) == 102.0


def test_matrix_vector_columns_are_columns(tmp_path):
    code = _hlsl(tmp_path, MATRIX_COLUMNS)

    assert "transpose(float2x2(float2(1.0, 2.0), float2(3.0, 4.0)))" in code
    assert "transpose(float3x3(float3(1.0, 2.0, 3.0), float(0)," in code
    # m[1][0] = 3, m[0][1] = 2, n[1][2] = 8, n[0][2] = 7, p[0][2] = 3, p's other columns are zero,
    # a[1][1][0] = 11, and the nested and flat lists f and g agree.
    expected = 3.0 + 20.0 + 800.0 + 7000.0 + 30000.0 + 1100000.0
    assert _stored_constant(_dxil(tmp_path, code)) == expected


def test_matrix_columns_of_other_component_types_compile(tmp_path):
    code = _hlsl(tmp_path, MIXED_TYPES)

    assert "float(0)" not in code
    _dxil(tmp_path, code)


@pytest.mark.parametrize(
    "initializer, reason",
    [
        ("{vec2(1.0, 2.0), 3.0}", "matrix-shape-mismatch"),
        ("{vec3(1.0), vec3(2.0)}", "matrix-shape-mismatch"),
        ("{vec2(1.0), vec2(2.0), vec2(3.0)}", "element-count-mismatch"),
    ],
)
def test_malformed_matrix_columns_are_refused(tmp_path, initializer, reason):
    program = MATRIX_COLUMNS.replace(
        "mat2 m = {vec2(1.0, 2.0), vec2(3.0, 4.0)};", f"mat2 m = {initializer};"
    )
    with pytest.raises(DirectXAggregateInitializerError) as raised:
        _hlsl(tmp_path, program)

    assert raised.value.reason == reason
