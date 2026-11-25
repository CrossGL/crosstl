"""WGSL output lowers the NaN-avoiding fmin, fmax, fmin3 and fmax3 to helpers that test the operands' bits.

WGSL declares none of these functions, so the generator refused every Metal, CUDA or HIP kernel that called them.
They return the operand that is not a NaN, which WGSL's min and max need not do (WGSL section 15.7), so each helper
tests the binary32 bits of both operands before calling min or max.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.translator.codegen.wgsl_nan_aware_min_max import WGSLNanAwareMinMaxError

PROGRAM = textwrap.dedent("""
    shader Probe {
        buffer float values[];
        buffer vec3 vectors[];
        buffer int counts[];
        %s
        compute {
            layout(local_size_x = 1) in;
            void main() {
                uint i = gl_GlobalInvocationID.x;
                %s
            }
        }
    }
""")

SCALAR_FMAX = textwrap.dedent("""\
    fn crosstl_fmax_f32(x: f32, y: f32) -> f32 {
        let x_nan = ((bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u);
        let y_nan = ((bitcast<u32>(y) & 0x7fffffffu) > 0x7f800000u);
        return select(select(max(x, y), x, y_nan), y, x_nan);
    }""")

VECTOR_FMIN = textwrap.dedent("""\
    fn crosstl_fmin_vec3_f32(x: vec3<f32>, y: vec3<f32>) -> vec3<f32> {
        let x_nan = ((bitcast<vec3<u32>>(x) & vec3<u32>(0x7fffffffu)) > vec3<u32>(0x7f800000u));
        let y_nan = ((bitcast<vec3<u32>>(y) & vec3<u32>(0x7fffffffu)) > vec3<u32>(0x7f800000u));
        return select(select(min(x, y), x, y_nan), y, x_nan);
    }""")


def validate(tmp_path, output):
    """Check ``output`` with naga when it is installed."""
    if shutil.which("naga"):
        (tmp_path / "out.wgsl").write_text(output)
        checked = subprocess.run(
            ["naga", "out.wgsl"], cwd=tmp_path, capture_output=True, text=True
        )
        assert checked.returncode == 0, checked.stdout + checked.stderr


def wgsl(tmp_path, body, helpers=""):
    """WGSL output for a compute body, checked with naga when it is installed."""
    path = tmp_path / "probe.cgl"
    path.write_text(PROGRAM % (helpers, body))
    output = crosstl.translate(str(path), backend="wgsl", format_output=False)
    validate(tmp_path, output)
    return output


def test_scalar_fmax_tests_both_operands_for_nan(tmp_path):
    output = wgsl(tmp_path, "values[i] = fmax(values[i], values[i + 1u]);")
    assert "values[i] = crosstl_fmax_f32(values[i], values[(i + 1u)]);" in output
    assert SCALAR_FMAX in output
    assert "fmax(" not in output.replace("crosstl_fmax_f32(", "")


def test_vector_fmin_tests_every_component(tmp_path):
    output = wgsl(tmp_path, "vectors[i] = fmin(vectors[i], vectors[i + 1u]);")
    # A runtime array of vec3 is a member of a storage structure (WGSL's vec3 stride).
    assert (
        "vectorsStorage.vectors[i] = crosstl_fmin_vec3_f32(vectorsStorage.vectors[i], "
        "vectorsStorage.vectors[(i + 1u)]);"
    ) in output
    assert VECTOR_FMIN in output


def test_one_helper_per_function_and_type(tmp_path):
    output = wgsl(
        tmp_path,
        "values[i] = fmax(values[i], 1.0) + fmax(2.0, values[i]); "
        "vectors[i] = fmax(vectors[i], vectors[i]);",
    )
    assert output.count("fn crosstl_fmax_f32(") == 1
    assert output.count("fn crosstl_fmax_vec3_f32(") == 1
    assert "fn crosstl_fmin_" not in output


def test_scalar_operands_of_a_vector_call_are_splatted(tmp_path):
    output = wgsl(
        tmp_path,
        "vectors[i] = fmax(vectors[i], 0.5) + fmin(values[i], vectors[i]) "
        "+ fmax(vectors[i], counts[i]);",
    )
    vectors = "vectorsStorage.vectors[i]"
    assert f"crosstl_fmax_vec3_f32({vectors}, vec3<f32>(0.5))" in output
    assert f"crosstl_fmin_vec3_f32(vec3<f32>(values[i]), {vectors})" in output
    assert f"crosstl_fmax_vec3_f32({vectors}, vec3<f32>(f32(counts[i])))" in output


def test_integer_operand_of_a_scalar_call_is_converted(tmp_path):
    output = wgsl(tmp_path, "values[i] = fmin(values[i], counts[i]);")
    assert "crosstl_fmin_f32(values[i], f32(counts[i]))" in output


@pytest.mark.parametrize(("function", "helper"), [("fmax3", "fmax"), ("fmin3", "fmin")])
def test_three_operand_forms_nest_the_helper_in_source_order(
    tmp_path, function, helper
):
    output = wgsl(
        tmp_path,
        f"values[i] = {function}(values[i], values[i + 1u], values[i + 2u]);",
    )
    assert (
        f"values[i] = crosstl_{helper}_f32(crosstl_{helper}_f32(values[i], "
        "values[(i + 1u)]), values[(i + 2u)]);"
    ) in output


def test_metal_fmax_translates_to_wgsl(tmp_path):
    source = tmp_path / "kernel.metal"
    source.write_text(textwrap.dedent("""\
        #include <metal_stdlib>
        using namespace metal;
        kernel void k(device float* x [[buffer(0)]], device float2* v [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
            x[i] = fmax(x[i], 1.0f) + fmin3(x[i], x[i + 1], 2.0f);
            v[i] = fmin(v[i], v[i + 1]);
        }
    """))
    output = crosstl.translate(str(source), backend="wgsl", format_output=False)
    validate(tmp_path, output)
    assert "crosstl_fmax_f32(" in output
    assert "crosstl_fmin_f32(crosstl_fmin_f32(" in output
    assert "crosstl_fmin_vec2_f32(" in output


def test_cuda_fmaxf_translates_to_wgsl(tmp_path):
    source = tmp_path / "kernel.cu"
    source.write_text(textwrap.dedent("""\
        __global__ void k(float* x, const float* y) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            x[i] = fmaxf(x[i], y[i]) + fminf(x[i], 0.0f);
        }
    """))
    output = crosstl.translate(str(source), backend="wgsl", format_output=False)
    validate(tmp_path, output)
    assert "crosstl_fmax_f32(" in output
    assert "crosstl_fmin_f32(" in output


def test_a_declared_fmax_is_called_as_declared(tmp_path):
    output = wgsl(
        tmp_path,
        "values[i] = fmax(values[i], 1.0);",
        helpers="float fmax(float a, float b) { return a + b; }",
    )
    assert "crosstl_fmax_f32" not in output


def test_integer_operands_are_refused(tmp_path):
    path = tmp_path / "probe.cgl"
    path.write_text(PROGRAM % ("", "counts[i] = fmax(counts[i], counts[i + 1u]);"))
    with pytest.raises(WGSLNanAwareMinMaxError, match=r"fmax\(\) on a i32 operand"):
        crosstl.translate(str(path), backend="wgsl", format_output=False)


def test_naga_spirv_of_the_helper_tests_nan_bits(tmp_path):
    """The SPIR-V naga lowers the helper to compares the bits as the coverage NaN check reads them."""
    if not (shutil.which("naga") and shutil.which("spirv-dis")):
        pytest.skip("naga and spirv-dis are not installed")
    from tools.coverage import spirv_input

    output = wgsl(tmp_path, "values[i] = fmax(values[i], values[i + 1u]);")
    (tmp_path / "out.wgsl").write_text(output)
    subprocess.run(["naga", "out.wgsl", "out.spv"], cwd=tmp_path, check=True)
    assembly = subprocess.run(
        ["spirv-dis", "--raw-id", "out.spv"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert spirv_input._tests_nan_bits(assembly)
