"""HIP function prototypes import as CrossGL prototypes and round trip through hipcc (#3260)."""

import re
import shutil
import subprocess

import pytest

from crosstl import translate
from crosstl.backend.import_refusals import HipImportRefused

FORWARD_DECLARED = """#include <hip/hip_runtime.h>
__device__ float twice(float x);
__device__ float quadruple(float x) { return twice(twice(x)); }
__device__ float twice(float x) { return x * 2.0f; }
"""

KERNEL_ABOVE_DEFINITION = """#include <hip/hip_runtime.h>
__device__ float twice(float x);
__global__ void probe(float* out) { out[0] = twice(out[0]); }
__device__ float twice(float x) { return x * 2.0f; }
"""

STATIC_INLINE = """#include <hip/hip_runtime.h>
static __device__ __forceinline__ float twice(float x);
static __device__ __forceinline__ float quadruple(float x) { return twice(twice(x)); }
static __device__ __forceinline__ float twice(float x) { return x * 2.0f; }
__global__ void probe(float* out) { out[0] = quadruple(out[0]); }
"""

MUTUAL_RECURSION = """#include <hip/hip_runtime.h>
__device__ bool is_odd(int n);
__device__ bool is_even(int n) { return n == 0 ? true : is_odd(n - 1); }
__device__ bool is_odd(int n) { return n == 0 ? false : is_even(n - 1); }
__global__ void probe(int* out) { out[0] = is_even(out[0]) ? 1 : 0; }
"""

PROTOTYPE_AFTER_DEFINITION = """#include <hip/hip_runtime.h>
__device__ float twice(float x) { return x * 2.0f; }
__device__ float twice(float x);
__global__ void probe(float* out) { out[0] = twice(out[0]); }
"""

ROUND_TRIPS = {
    "issue": FORWARD_DECLARED,
    "kernel-above-definition": KERNEL_ABOVE_DEFINITION,
    "static-forceinline": STATIC_INLINE,
    "mutual-recursion": MUTUAL_RECURSION,
    "prototype-after-definition": PROTOTYPE_AFTER_DEFINITION,
}


def _translate(tmp_path, source_text, backend="cgl"):
    source = tmp_path / "probe.hip"
    source.write_text(source_text, encoding="utf-8")
    return translate(str(source), backend=backend, save_shader=None)


def _compile_with_hipcc(tmp_path, code):
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        pytest.skip("hipcc is not installed")
    source = tmp_path / "round_trip.hip"
    source.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [
            hipcc,
            "--offload-arch=gfx1100",
            "-std=c++17",
            "-c",
            str(source),
            "-o",
            str(tmp_path / "round_trip.o"),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_prototype_imports_as_a_crossgl_prototype(tmp_path):
    crossgl = _translate(tmp_path, FORWARD_DECLARED)

    assert crossgl.index("f32 twice(f32 x);") < crossgl.index("f32 quadruple(f32 x) {")
    assert len(re.findall(r"\bf32 twice\(f32 x\) \{", crossgl)) == 1
    assert "f32 twice(f32 x) {\n}" not in crossgl
    assert "return twice(twice(x));" in crossgl


@pytest.mark.parametrize("source_text", ROUND_TRIPS.values(), ids=ROUND_TRIPS)
def test_prototype_round_trip_compiles_with_hipcc(tmp_path, source_text):
    hip = _translate(tmp_path, source_text, backend="hip")

    _compile_with_hipcc(tmp_path, hip)


def test_uncalled_prototype_without_a_definition_imports_as_a_prototype(tmp_path):
    crossgl = _translate(
        tmp_path,
        "__device__ float twice(float x);\n"
        "__global__ void probe(float* out) { out[0] = 1.0f; }\n",
    )

    assert "f32 twice(f32 x);" in crossgl
    assert "f32 twice(f32 x) {" not in crossgl


@pytest.mark.parametrize("call", ["twice(out[0])", "::twice(out[0])"])
def test_called_prototype_without_a_definition_is_refused(tmp_path, call):
    with pytest.raises(HipImportRefused) as raised:
        _translate(
            tmp_path,
            "__device__ float twice(float x);\n"
            f"__global__ void probe(float* out) {{ out[0] = {call}; }}\n",
        )

    assert "HIP function 'twice' is called but never defined" in str(raised.value)
