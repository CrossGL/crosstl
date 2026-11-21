"""Matrix and vector subscripts in CUDA and HIP output (#3193).

CrossGL, like GLSL, reads column ``i`` of a matrix with ``m[i]`` and element ``j`` of that column with ``m[i][j]``,
and component ``i`` of a vector with ``v[i]`` (GLSL 4.60 sections 5.5 and 5.6). CUDA and HIP output copied both
unchanged before: the generated matrix structures subscripted their flat storage, so ``m[1]`` was one ``float``, and
CUDA's vector types have no subscript operator. A host C++ compiler runs the generated kernels here on one thread
(the CUDA keywords are defined away), so the values the subscripts read and write are checked, not only the text.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.lowering_refusals import LoweringRefused

BACKENDS = ["cuda", "hip"]

RUN_PROBE = """
shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* data;
        uniform int k;

        vec3 pick(mat3 a, int c) { return a[c]; }
        void bump(inout float x) { x = x + 1.0; }

        void main() {
            mat3 m = mat3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
            vec3 c = m[1];
            data[0] = c.x + c.y * 10.0 + c.z * 100.0;
            m[0] = vec3(10.0, 11.0, 12.0);
            m[2][1] = 42.0;
            m[k][2] += 1.0;
            m[1] *= 2.0;
            m[2].x = 70.0;
            data[1] = m[0][0] + m[0][1] + m[0][2];
            data[2] = m[1][0] + m[1][1] * 10.0 + m[1][2] * 100.0;
            data[3] = m[2][0] + m[2][1] + m[2][2];
            data[4] = pick(m, k).z + m[k].y + m[0][k];
            vec4 v = vec4(1.0, 2.0, 3.0, 4.0);
            v[k] = 5.0;
            v[2] += 1.0;
            bump(v[k]);
            bump(m[1][1]);
            data[5] = v.x + v.y * 10.0 + v.z * 100.0 + v.w * 1000.0;
            data[6] = m[1][1] + dot(m[1], vec3(1.0, 1.0, 1.0));
            mat3x2 r = mat3x2(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
            r[1].xy = r[2].yx;
            data[7] = r[1].x * 10.0 + r[1].y + r[2][k];
            mat3 t = transpose(m);
            data[8] = t[0][1] + t[2][0];
            m[0] = m[2];
            data[9] = m[0][0] + m[0][1] + m[0][2];
            dmat2 d = dmat2(1.0, 2.0, 3.0, 4.0);
            d[k][0] = d[0][1] + 10.0;
            data[10] = float(d[1][0] + d[1][1]);
        }
    }
}
"""

# GLSL semantics of RUN_PROBE with k = 1, worked by hand: columns are written and read in place.
RUN_EXPECTED = [654.0, 33.0, 1508.0, 121.0, 35.0, 4461.0, 44.0, 71.0, 20.0, 121.0, 16.0]

ARRAYS = """
shader Probe {
    compute {
        layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;
        buffer float* data;
        buffer mat4* mats;
        uniform int k;
        shared vec4 tile[4];

        void main() {
            uint id = gl_LocalInvocationID.x;
            tile[id] = vec4(float(id));
            barrier();
            vec3 arr[3];
            arr[0] = vec3(1.0, 2.0, 3.0);
            arr[k][1] = 9.0;
            float fs[4];
            fs[k] = 3.0;
            data[id] = tile[id][k] + arr[k][2] + fs[k] + mats[id][2][3];
        }
    }
}
"""

OUT_OF_RANGE = """
shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* data;
        void main() {
            mat3 m = mat3(1.0);
            vec4 v = vec4(1.0);
            data[0] = %s;
        }
    }
}
"""

COLUMN_OUT_ARGUMENT = """
shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* data;
        void grow(inout vec3 c) { c = c * 2.0; }
        void main() {
            mat3 m = mat3(1.0);
            grow(m[1]);
            data[0] = m[1][1];
        }
    }
}
"""

_HOST_SHIM = """
#include <cstdio>
#define __host__
#define __device__
#define __global__
#define __constant__
#define __shared__ static
struct float2 { float x, y; };
struct float3 { float x, y, z; };
struct float4 { float x, y, z, w; };
struct double2 { double x, y; };
struct double3 { double x, y, z; };
struct double4 { double x, y, z, w; };
struct uint3 { unsigned int x, y, z; };
inline float2 make_float2(float x, float y) { return {x, y}; }
inline float3 make_float3(float x, float y, float z) { return {x, y, z}; }
inline float4 make_float4(float x, float y, float z, float w) { return {x, y, z, w}; }
inline double2 make_double2(double x, double y) { return {x, y}; }
inline double3 make_double3(double x, double y, double z) { return {x, y, z}; }
inline double4 make_double4(double x, double y, double z, double w) { return {x, y, z, w}; }
static uint3 threadIdx, blockIdx, blockDim, gridDim;
"""

_HOST_MAIN = """
int main() {
    float out[%d] = {0};
    data = out;
    k = 1;
    blockDim = {1, 1, 1};
    compute_main();
    for (float value : out) {
        std::printf("%%.6f\\n", value);
    }
    return 0;
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "subscripts.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _host_compiler():
    for name in ("c++", "clang++", "g++"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _target_compile(tmp_path, code, backend):
    """Compile ``code`` with the target's own compiler when it is installed."""
    if backend == "cuda" and shutil.which("nvcc"):
        (tmp_path / "out.cu").write_text(code)
        command = [shutil.which("nvcc"), "-std=c++17", "--ptx", "out.cu"]
    elif backend == "hip" and shutil.which("hipcc"):
        (tmp_path / "out.hip").write_text(code)
        command = [
            shutil.which("hipcc"),
            "-std=c++17",
            "--offload-device-only",
            "-c",
            "out.hip",
        ]
    else:
        return
    result = subprocess.run(
        command, capture_output=True, text=True, cwd=tmp_path, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("backend", BACKENDS)
def test_subscripts_compute_the_glsl_values_on_the_host(tmp_path, backend):
    compiler = _host_compiler()
    if compiler is None:
        pytest.skip("no host C++ compiler")
    code = _translate(tmp_path, RUN_PROBE, backend)
    _target_compile(tmp_path, code, backend)
    body = "\n".join(
        line for line in code.splitlines() if not line.startswith("#include <")
    )
    source = tmp_path / "host.cpp"
    source.write_text(_HOST_SHIM + "\n" + body + "\n" + _HOST_MAIN % len(RUN_EXPECTED))
    built = subprocess.run(
        [compiler, "-std=c++17", "-o", "host", "host.cpp"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    ran = subprocess.run(
        [str(tmp_path / "host")], capture_output=True, text=True, check=False
    )
    assert ran.returncode == 0, ran.stderr
    assert [float(value) for value in ran.stdout.split()] == RUN_EXPECTED


@pytest.mark.parametrize("backend", BACKENDS)
def test_matrix_columns_read_as_vectors_and_write_through_views(tmp_path, backend):
    code = _translate(tmp_path, RUN_PROBE, backend)
    assert "float3 c = m.column(1);" in code
    assert "return a.column(c);" in code
    assert "m[0] = make_float3(10.0f, 11.0f, 12.0f);" in code
    assert "m[2].y = 42.0f;" in code
    assert "m[k].z += 1.0f;" in code
    assert "m[2].x = 70.0f;" in code
    assert "bump(m[1].y);" in code
    assert "cgl_float3_dot(m.column(1)" in code
    assert (
        "__host__ __device__ cgl_float3_column operator[](int index) "
        "{ return cgl_float3_column(m + (static_cast<unsigned int>(index) < 3u ? index : 2) * 3); }"
    ) in code
    assert "struct cgl_double2_const_column {" in code
    assert "return m[index]; }" not in code


@pytest.mark.parametrize("backend", BACKENDS)
def test_vector_subscripts_name_components(tmp_path, backend):
    code = _translate(tmp_path, RUN_PROBE, backend)
    assert "cgl_vector_component4(v, k) = 5.0f;" in code
    assert "v.z += 1.0f;" in code
    assert "bump(cgl_vector_component4(v, k));" in code
    assert "cgl_vector_component2(r[2], k)" in code
    assert re.search(
        r"template <typename V>\s*__host__ __device__ inline auto cgl_vector_component4\(V&& v, int index\)"
        r"\s*-> decltype\(\(v\.x\)\)",
        code,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_arrays_keep_their_subscripts(tmp_path, backend):
    code = _translate(tmp_path, ARRAYS, backend)
    assert "tile[id] = cgl_float4_splat(" in code
    assert "arr[k].y = 9.0f;" in code
    assert "fs[k] = 3.0f;" in code
    assert "cgl_vector_component4(tile[id], k)" in code
    assert "cgl_vector_component3(arr[k], 2)" not in code
    assert "arr[k].z" in code
    assert "mats[id][2].w" in code
    _target_compile(tmp_path, code, backend)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "expression, message",
    [
        ("m[3][0]", r"the \w+ matrix with index 3 in \w+: its columns are 0 to 2"),
        ("v[4]", r"the \w+ vector with index 4 in \w+: its components are 0 to 3"),
        ("m[0][3]", r"the \w+ vector with index 3 in \w+: its components are 0 to 2"),
    ],
)
def test_constant_subscripts_past_the_end_are_refused(
    tmp_path, backend, expression, message
):
    with pytest.raises(LoweringRefused, match=message):
        _translate(tmp_path, OUT_OF_RANGE % expression, backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_matrix_column_out_argument_is_refused(tmp_path, backend):
    with pytest.raises(
        LoweringRefused, match="matrix column passed to an out/inout parameter"
    ):
        _translate(tmp_path, COLUMN_OUT_ARGUMENT, backend)


POINTER_BUFFERS = """
shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* data;
        buffer mat4* mats;
        buffer dmat2x4* dmats;
        buffer bvec2* flags;
        buffer u8vec4* bytes;
        buffer f16vec2* halves;
        void main() {
            data[0] = mats[0][3][2] + float(dmats[0][1][3]);
        }
    }
}
"""


@pytest.mark.parametrize("backend", BACKENDS)
def test_buffer_pointers_name_the_generated_element_types(tmp_path, backend):
    """HIP output declared ``buffer mat4* m`` as ``mat4x4* m`` before, a type HIP does not define."""
    code = _translate(tmp_path, POINTER_BUFFERS, backend)
    for declaration in (
        "__device__ float4x4* mats;",
        "__device__ double2x4* dmats;",
        "__device__ uchar2* flags;",
        "__device__ uchar4* bytes;",
        "__device__ half2* halves;",
    ):
        assert declaration in code
    assert "(mats[0][3].z + float(dmats[0][1].w))" in code
    _target_compile(tmp_path, code, backend)
