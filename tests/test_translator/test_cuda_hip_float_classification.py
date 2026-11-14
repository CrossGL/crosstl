"""CUDA and HIP output lowers ``isnan`` and ``isinf`` of a vector component-wise.

GLSL 4.60 section 8.3 defines both on vectors, returning ``bvecN``. CUDA and HIP declare them for scalars only, so the
call used to reach the output with its vector argument and nvcc and hipcc rejected it (#4586). The generators now call
a helper that builds the ``uchar``N they use for ``bvecN`` from one scalar call per component.
"""

import shutil
import subprocess

import pytest

import crosstl

KERNEL = """shader FloatClassification {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float data[];
        void main() {
            float f = data[0];
            vec2 w = vec2(f, data[1]);
            vec3 v = vec3(f, data[1], data[2]);
            vec4 q = vec4(v, data[3]);
            vec3 u = vec3(1.0, 2.0, 3.0);
            bvec2 n2 = isnan(w);
            bvec3 n3 = isnan(v);
            bvec4 n4 = isnan(q);
            bvec2 i2 = isinf(w);
            bvec3 i3 = isinf(v);
            bvec4 i4 = isinf(q * 2.0);
            bool s = isnan(f);
            bool t = isinf(f);
            vec3 m = mix(v, u, isinf(v));
            data[4] = any(isinf(v)) ? 1.0 : 0.0;
            data[5] = all(isnan(q)) ? 1.0 : 0.0;
            data[6] = isnan(v).y ? 1.0 : 0.0;
            data[7] = (n2.x || n3.y || n4.z || i2.x || i3.y || i4.w || s || t) ? m.x : 0.0;
        }
    }
}
"""

BACKENDS = ["cuda", "hip"]


def translate(tmp_path, backend, source=KERNEL):
    path = tmp_path / "kernel.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def kernel_lines(code):
    return {
        line.strip() for line in code[code.index("void compute_main") :].splitlines()
    }


@pytest.mark.parametrize("backend", BACKENDS)
def test_vector_isnan_and_isinf_call_componentwise_helpers(tmp_path, backend):
    lines = kernel_lines(translate(tmp_path, backend))

    for statement in (
        "uchar2 n2 = cgl_float2_isnan(w);",
        "uchar3 n3 = cgl_float3_isnan(v);",
        "uchar4 n4 = cgl_float4_isnan(q);",
        "uchar2 i2 = cgl_float2_isinf(w);",
        "uchar3 i3 = cgl_float3_isinf(v);",
        "uchar4 i4 = cgl_float4_isinf(cgl_float4_mul_scalar(q, 2.0f));",
    ):
        assert statement in lines


@pytest.mark.parametrize("backend", BACKENDS)
def test_helpers_call_the_scalar_function_per_component(tmp_path, backend):
    code = translate(tmp_path, backend)

    for function in ("isnan", "isinf"):
        assert (
            f"__device__ inline uchar2 cgl_float2_{function}(float2 value)\n{{\n"
            f"    return make_uchar2({function}(value.x), {function}(value.y));\n}}"
        ) in code
        assert (
            f"__device__ inline uchar3 cgl_float3_{function}(float3 value)\n{{\n"
            f"    return make_uchar3({function}(value.x), {function}(value.y), {function}(value.z));\n}}"
        ) in code
        assert (
            f"__device__ inline uchar4 cgl_float4_{function}(float4 value)\n{{\n"
            f"    return make_uchar4({function}(value.x), {function}(value.y), {function}(value.z), "
            f"{function}(value.w));\n}}"
        ) in code


@pytest.mark.parametrize("backend", BACKENDS)
def test_scalar_isnan_and_isinf_are_unchanged(tmp_path, backend):
    lines = kernel_lines(translate(tmp_path, backend))

    assert "bool s = isnan(f);" in lines
    assert "bool t = isinf(f);" in lines


@pytest.mark.parametrize("backend", BACKENDS)
def test_result_is_a_bool_vector_for_reductions_selects_and_components(
    tmp_path, backend
):
    lines = kernel_lines(translate(tmp_path, backend))

    assert "data[4] = (cgl_uchar3_any(cgl_float3_isinf(v)) ? 1.0f : 0.0f);" in lines
    assert "data[5] = (cgl_uchar4_all(cgl_float4_isnan(q)) ? 1.0f : 0.0f);" in lines
    assert "data[6] = (cgl_float3_isnan(v).y ? 1.0f : 0.0f);" in lines
    # A bvec3 selector makes mix select per component instead of blending with a scalar weight.
    mix = next(line for line in lines if line.startswith("float3 m ="))
    assert "cgl_float3_isinf(v)" in mix and "mix_scalar" not in mix


@pytest.mark.parametrize("backend", BACKENDS)
def test_no_vector_argument_reaches_the_scalar_function(tmp_path, backend):
    code = translate(tmp_path, backend)

    for argument in ("w", "v", "q"):
        assert f" isnan({argument})" not in code and f"(isnan({argument})" not in code
        assert f" isinf({argument})" not in code and f"(isinf({argument})" not in code


@pytest.mark.parametrize("backend", BACKENDS)
def test_double_and_half_vectors(tmp_path, backend):
    source = """shader WideAndNarrow {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float data[];
        void main() {
            vec3 v = vec3(data[0], data[1], data[2]);
            dvec3 d = dvec3(v);
            f16vec2 h = f16vec2(v.xy);
            bvec3 a = isnan(d);
            bvec2 b = isinf(h);
            data[3] = (a.x || b.y) ? 1.0 : 0.0;
        }
    }
}
"""
    code = translate(tmp_path, backend, source)
    lines = kernel_lines(code)

    assert "uchar3 a = cgl_double3_isnan(d);" in lines
    assert "uchar2 b = cgl_half2_isinf(h);" in lines
    assert "return make_uchar3(isnan(value.x), isnan(value.y), isnan(value.z));" in code
    assert (
        "return make_uchar2(isinf(__half2float(value.x)), isinf(__half2float(value.y)));"
        in code
    )


def _compile(command, code):
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr + "\n\n" + code


def test_cuda_output_compiles_with_nvcc(tmp_path):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is not installed")
    code = translate(tmp_path, "cuda")
    path = tmp_path / "kernel.cu"
    path.write_text(code)
    _compile([nvcc, "-std=c++17", "-c", str(path), "-o", str(tmp_path / "k.o")], code)


def test_hip_output_compiles_with_hipcc(tmp_path):
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        pytest.skip("hipcc is not installed")
    code = translate(tmp_path, "hip")
    path = tmp_path / "kernel.hip"
    path.write_text(code)
    _compile([hipcc, "-std=c++17", "-c", str(path), "-o", str(tmp_path / "k.o")], code)
