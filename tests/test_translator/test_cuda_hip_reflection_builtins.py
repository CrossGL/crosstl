"""CUDA and HIP output lowers GLSL's ``reflect``, ``refract`` and ``faceforward`` to generated helpers.

Neither the CUDA Math API nor the HIP runtime declares these three functions, for any scalar or vector
type, so the emitters wrote calls nvcc and hipcc rejected as undeclared identifiers. Both emitters now
generate a helper per type that follows GLSL 4.60 section 8.5, as they already do for ``normalize``.
"""

import shutil
import subprocess

import pytest

import crosstl

KERNEL = """shader Reflection {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer vec4* outv;
        void main() {
            vec3 i = vec3(1.0, -1.0, 0.0);
            vec3 n = vec3(0.0, 1.0, 0.0);
            vec3 r = reflect(i, n);
            vec3 t = refract(normalize(i), n, 0.5);
            vec3 f = faceforward(n, i, n);
            vec2 r2 = reflect(vec2(1.0, -1.0), vec2(0.0, 1.0));
            float rs = reflect(1.0, 1.0);
            outv[0] = vec4(r, 0.0);
            outv[1] = vec4(t, rs);
            outv[2] = vec4(f, r2.x);
        }
    }
}
"""

TYPED_KERNEL = """shader TypedReflection {
    compute {
        layout(local_size_x = 1) in;
        buffer double* outd;
        buffer float* outf;
        void main() {
            dvec3 di = dvec3(1.0, -1.0, 0.0);
            dvec3 dn = dvec3(0.0, 1.0, 0.0);
            dvec3 dr = reflect(di, dn);
            dvec3 dt = refract(di, dn, 0.5);
            dvec4 df = faceforward(dvec4(dn, 0.0), dvec4(di, 0.0), dvec4(dn, 0.0));
            half hi = half(1.0);
            half hn = half(1.0);
            half hr = reflect(hi, hn);
            half ht = refract(hi, hn, half(0.5));
            half hf = faceforward(hn, hi, hn);
            float fs = faceforward(1.0, -1.0, 1.0);
            float ft = refract(1.0, 1.0, 0.5);
            outd[0] = dr.x + dt.y + df.z;
            outf[0] = float(hr) + float(ht) + float(hf) + fs + ft;
        }
    }
}
"""

REFLECT_FLOAT3 = """__device__ inline float3 cgl_float3_reflect(float3 i, float3 n)
{
    float d = 2.0f * ((n.x * i.x) + (n.y * i.y) + (n.z * i.z));
    return make_float3(i.x - d * n.x, i.y - d * n.y, i.z - d * n.z);
}"""

REFRACT_FLOAT3 = """__device__ inline float3 cgl_float3_refract(float3 i, float3 n, float eta)
{
    float d = (n.x * i.x) + (n.y * i.y) + (n.z * i.z);
    float k = 1.0f - eta * eta * (1.0f - d * d);
    if (k < 0.0f)
    {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    float s = eta * d + sqrtf(k);
    return make_float3(eta * i.x - s * n.x, eta * i.y - s * n.y, eta * i.z - s * n.z);
}"""

FACEFORWARD_FLOAT3 = """__device__ inline float3 cgl_float3_faceforward(float3 n, float3 i, float3 nref)
{
    if (((nref.x * i.x) + (nref.y * i.y) + (nref.z * i.z)) < 0.0f)
    {
        return n;
    }
    return make_float3(-n.x, -n.y, -n.z);
}"""

REFLECT_HALF = """__device__ inline half cgl_reflect_half(half i, half n)
{
    float d = 2.0f * ((__half2float(n) * __half2float(i)));
    return __float2half(__half2float(i) - d * __half2float(n));
}"""


def translate(tmp_path, backend, source=KERNEL):
    path = tmp_path / "kernel.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def kernel_body(code):
    return code[code.index("void compute_main") :]


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_calls_use_generated_helpers(tmp_path, backend):
    lines = {
        line.strip() for line in kernel_body(translate(tmp_path, backend)).splitlines()
    }

    for statement in (
        "float3 r = cgl_float3_reflect(i, n);",
        "float3 t = cgl_float3_refract(cgl_float3_normalize(i), n, 0.5f);",
        "float3 f = cgl_float3_faceforward(n, i, n);",
        "float2 r2 = cgl_float2_reflect(make_float2(1.0f, -1.0f), make_float2(0.0f, 1.0f));",
        "float rs = cgl_reflect_float(1.0f, 1.0f);",
    ):
        assert statement in lines


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_helpers_follow_the_glsl_definitions(tmp_path, backend):
    code = translate(tmp_path, backend)

    for helper in (REFLECT_FLOAT3, REFRACT_FLOAT3, FACEFORWARD_FLOAT3):
        assert helper in code


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_no_bare_reflection_builtin_reaches_the_output(tmp_path, backend):
    for source in (KERNEL, TYPED_KERNEL):
        code = translate(tmp_path, backend, source)
        for builtin in ("reflect(", "refract(", "faceforward("):
            assert f" {builtin}" not in code and f"({builtin}" not in code


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_double_and_half_operands_get_typed_helpers(tmp_path, backend):
    code = translate(tmp_path, backend, TYPED_KERNEL)
    lines = {line.strip() for line in kernel_body(code).splitlines()}

    for signature in (
        "__device__ inline double3 cgl_double3_reflect(double3 i, double3 n)",
        "__device__ inline double3 cgl_double3_refract(double3 i, double3 n, double eta)",
        "__device__ inline double4 cgl_double4_faceforward(double4 n, double4 i, double4 nref)",
        "__device__ inline half cgl_refract_half(half i, half n, float eta)",
        "__device__ inline half cgl_faceforward_half(half n, half i, half nref)",
        "__device__ inline float cgl_faceforward_float(float n, float i, float nref)",
    ):
        assert signature in code
    assert REFLECT_HALF in code
    assert "    double s = eta * d + sqrt(k);" in code
    for statement in (
        "double3 dr = cgl_double3_reflect(di, dn);",
        "half hr = cgl_reflect_half(hi, hn);",
        "float fs = cgl_faceforward_float(1.0f, -1.0f, 1.0f);",
        "float ft = cgl_refract_float(1.0f, 1.0f, 0.5f);",
    ):
        assert statement in lines


def _compile(tmp_path, tool, args, code, suffix):
    if shutil.which(tool) is None:
        pytest.skip(f"{tool} is not installed")
    source = tmp_path / f"reflection{suffix}"
    source.write_text(code)
    result = subprocess.run(
        [tool, *args, str(source), "-o", str(tmp_path / "reflection.o")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("source", [KERNEL, TYPED_KERNEL])
def test_nvcc_compiles(tmp_path, source):
    code = translate(tmp_path, "cuda", source)
    _compile(tmp_path, "nvcc", ["-std=c++17", "-c"], code, ".cu")


@pytest.mark.parametrize("source", [KERNEL, TYPED_KERNEL])
def test_hipcc_compiles(tmp_path, source):
    code = translate(tmp_path, "hip", source)
    _compile(
        tmp_path, "hipcc", ["-c", "--offload-arch=gfx90a", "-x", "hip"], code, ".hip"
    )
