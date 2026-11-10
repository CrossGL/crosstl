"""Array parameters of specialized generic functions keep their concrete element and their array form.

``T firstx<T>(in vec2<T> a[2])`` specialized for ``float`` declared ``T2 a[2]``: the generators spelled the element
``T2`` before substituting it, so Metal, HLSL, CUDA, HIP and Slang named an undeclared type and GLSL and SPIR-V
refused the program (issue #4001). ``void bump<T>(inout T a[2], T d)`` became ``thread float[2] & a`` on Metal,
which MSL rejects (issue #4003). Each case below fails on that code.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl

PROGRAM = textwrap.dedent("""
    shader main {
        buffer float* results;
        T firstx<T>(in vec2<T> a[2]) { return a[0].x + a[1].y; }
        void bump<T>(inout T a[2], T d) { a[0] = a[0] + d; }
        void fill<T>(out T a[2], T d) { a[0] = d; a[1] = d; }
        compute {
        @ numthreads(1, 1, 1)
        void main() {
            vec2 v[2] = {vec2(1.0, 2.0), vec2(3.0, 4.0)};
            float a[2] = {1.0, 2.0};
            bump<float>(a, 1.0);
            float b[2];
            fill<float>(b, 3.0);
            results[0] = firstx<float>(v) + a[0] + b[1];
        }
        }
    }
""").strip()


def translate(tmp_path, backend):
    path = tmp_path / "arrays.cgl"
    path.write_text(PROGRAM)
    return crosstl.translate(str(path), backend=backend)


def _validate(tmp_path, backend, output):
    if backend == "opengl":
        tool = shutil.which("glslangValidator")
        if tool is None:
            pytest.skip("glslangValidator is not installed")
        path = tmp_path / "out.comp"
        path.write_text(output)
        command = [tool, "-S", "comp", str(path)]
    elif backend == "vulkan":
        assembler, validator = shutil.which("spirv-as"), shutil.which("spirv-val")
        if assembler is None or validator is None:
            pytest.skip("SPIRV-Tools are not installed")
        path, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
        path.write_text(output)
        subprocess.run(
            [assembler, "--target-env", "vulkan1.1", str(path), "-o", str(binary)],
            check=True,
            capture_output=True,
        )
        command = [validator, "--target-env", "vulkan1.1", str(binary)]
    elif backend == "metal":
        xcrun = shutil.which("xcrun")
        if xcrun is None:
            pytest.skip("xcrun is not installed")
        path = tmp_path / "out.metal"
        path.write_text(output)
        command = [xcrun, "-sdk", "macosx", "metal", "-c", str(path), "-o", "/dev/null"]
    else:
        tool = shutil.which("dxc")
        if tool is None:
            pytest.skip("dxc is not installed")
        path = tmp_path / "out.hlsl"
        path.write_text(output)
        command = [tool, "-HV", "2021", "-T", "cs_6_2", "-E", "CSMain", str(path)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("backend", "signatures"),
    [
        (
            "metal",
            (
                "float firstx_float(float2 a[2])",
                "void bump_float(float a[2], float d)",
                "void fill_float(float a[2], float d)",
            ),
        ),
        (
            "directx",
            (
                "float firstx_float(in float2 a[2])",
                "void bump_float(inout float a[2], float d)",
                "void fill_float(out float a[2], float d)",
            ),
        ),
        (
            "opengl",
            (
                "float firstx_float(vec2 a[2])",
                "void bump_float(inout float a[2], float d)",
                "void fill_float(out float a[2], float d)",
            ),
        ),
        ("cuda", ("__device__ float firstx_float(float2 a[2])",)),
        ("hip", ("__device__ float firstx_float(float2 a[2])",)),
    ],
)
def test_specialized_array_parameters_are_concrete_arrays(
    tmp_path, backend, signatures
):
    output = translate(tmp_path, backend)

    for signature in signatures:
        assert signature in output, signature
    assert "T2" not in output and "float[2] &" not in output


@pytest.mark.parametrize("backend", ["metal", "directx", "opengl", "vulkan"])
def test_specialized_array_parameters_translate_to_valid_targets(tmp_path, backend):
    _validate(tmp_path, backend, translate(tmp_path, backend))
