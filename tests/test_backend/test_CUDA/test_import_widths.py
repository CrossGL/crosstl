"""CUDA and HIP ``size_t``, ``ptrdiff_t`` and ``__nv_bfloat16`` keep their width through import.

CUDA C++ and HIP run only on 64-bit hosts, whose ``size_t`` and ``ptrdiff_t`` are 64-bit (CUDA 12 removed 32-bit
host support; ROCm targets x86-64 Linux and Windows). The importers mapped ``size_t`` to ``u32``, so index
arithmetic such as ``blockIdx.x * (size_t)blockDim.x`` wrapped at 2^32. ``__nv_bfloat16`` (8 exponent bits, 7
fraction bits) imported as binary16 (5 and 10), which changes every value and the range, and its conversion
intrinsics reached CrossGL as undeclared calls.
"""

import re
import shutil
import subprocess

import pytest

from crosstl import translate
from crosstl.backend.bfloat16_conversions import format_bfloat16_conversion
from crosstl.translator.semantics.sizeof_values import sizeof_type_bytes

INDEX_KERNEL = """
__global__ void k(const unsigned int* a, float* o, size_t n, ptrdiff_t d) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  ptrdiff_t back = d - 1;
  if (i < n) o[i] = (float)a[i] + (float)back;
}
"""

BFLOAT16_KERNEL = """
#include <cuda_bf16.h>
__global__ void k(__nv_bfloat162* v, float* o, const __nv_bfloat16* a) {
  unsigned i = threadIdx.x;
  float2 f = __bfloat1622float2(v[i]);
  v[i] = __floats2bfloat162_rn(f.y, f.x);
  __nv_bfloat16 h = __float2bfloat16(o[i]);
  o[i] = __bfloat162float(__low2bfloat16(v[i])) + __bfloat162float(h * a[i]);
}
"""

SIZEOF_KERNEL = """
#include <cuda_bf16.h>
__global__ void k(unsigned long long* s) {
  s[0] = sizeof(size_t);
  s[1] = sizeof(ptrdiff_t);
  s[2] = sizeof(long long);
  s[3] = sizeof(unsigned int);
  s[4] = sizeof(__nv_bfloat16);
  s[5] = sizeof(__nv_bfloat162);
  s[6] = sizeof(float);
}
"""


def _translate(tmp_path, source, backend, suffix=".cu"):
    path = tmp_path / f"kernel{suffix}"
    path.write_text(source, encoding="utf-8")
    return translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize("suffix", [".cu", ".hip"])
def test_size_t_and_ptrdiff_t_import_as_64_bit(tmp_path, suffix):
    crossgl = _translate(tmp_path, INDEX_KERNEL, "cgl", suffix)
    assert re.search(r"\bn: u64\b|\bu64 n\b", crossgl), crossgl
    assert re.search(r"\bd: i64\b|\bi64 d\b", crossgl), crossgl
    assert "var i: u64 = " in crossgl
    assert "u64(gl_WorkGroupSize.x)" in crossgl
    assert "var back: i64 = " in crossgl
    assert not re.search(r"\b(?:size_t|ptrdiff_t)\b", crossgl)


def test_size_t_index_arithmetic_is_64_bit_on_metal_and_spirv(tmp_path):
    metal = _translate(tmp_path, INDEX_KERNEL, "metal")
    assert re.search(r"uint64_t i = .*uint64_t\(", metal), metal
    spirv = _translate(tmp_path, INDEX_KERNEL, "vulkan")
    assert "OpCapability Int64" in spirv
    assert re.search(r"OpTypeInt 64 0", spirv)
    assert "OpUConvert" in spirv


def test_opengl_refuses_a_64_bit_index_instead_of_truncating_it(tmp_path):
    with pytest.raises(Exception) as raised:
        _translate(tmp_path, INDEX_KERNEL, "opengl")
    assert getattr(raised.value, "project_diagnostic_code", None), raised.value
    assert "u64" in str(raised.value)


def test_bfloat16_imports_as_bfloat16_with_its_conversions(tmp_path):
    crossgl = _translate(tmp_path, BFLOAT16_KERNEL, "cgl")
    assert "array<vec2<bfloat16> >" in crossgl or "array<vec2<bfloat16>>" in crossgl
    assert "array<bfloat16>" in crossgl
    assert "var f: vec2<f32> = vec2<f32>(v[i]);" in crossgl
    assert "vec2<bfloat16>(bfloat16(f.y), bfloat16(f.x))" in crossgl
    assert "var h: bfloat16 = bfloat16(o[i]);" in crossgl
    assert not re.search(r"\bf16\b", crossgl)
    assert "__bfloat16" not in crossgl and "__nv_bfloat16" not in crossgl
    assert "__float2bfloat16" not in crossgl and "__low2bfloat16" not in crossgl


def test_bfloat16_buffers_keep_two_byte_elements(tmp_path):
    metal = _translate(tmp_path, BFLOAT16_KERNEL, "metal")
    assert "device bfloat2* v" in metal and "device bfloat* a" in metal
    assert "bfloat h = bfloat(o[i]);" in metal
    spirv = _translate(tmp_path, BFLOAT16_KERNEL, "vulkan")
    assert "OpCapability StorageBuffer16BitAccess" in spirv
    assert "ArrayStride 2" in spirv


@pytest.mark.parametrize(
    ("name", "args", "expected"),
    [
        ("__float2bfloat16", ["x"], "bfloat16(x)"),
        ("__float2bfloat16_rn", ["x"], "bfloat16(x)"),
        ("::__bfloat162float", ["h"], "f32(h)"),
        ("__float2bfloat162_rn", ["x"], "vec2<bfloat16>(bfloat16(x))"),
        (
            "__floats2bfloat162_rn",
            ["a", "b"],
            "vec2<bfloat16>(bfloat16(a), bfloat16(b))",
        ),
        ("__float22bfloat162_rn", ["v"], "vec2<bfloat16>(v)"),
        ("__bfloat1622float2", ["v"], "vec2<f32>(v)"),
        ("__bfloat162bfloat162", ["h"], "vec2<bfloat16>(h)"),
        ("__halves2bfloat162", ["a", "b"], "vec2<bfloat16>(a, b)"),
        ("__low2bfloat16", ["v[i]"], "(v[i]).x"),
        ("__high2bfloat16", ["v"], "v.y"),
    ],
)
def test_bfloat16_conversion_spellings(name, args, expected):
    assert format_bfloat16_conversion(name, args) == expected


@pytest.mark.parametrize(
    ("name", "args"),
    [
        # Rounding a double or an integer through binary32 would round twice.
        ("__double2bfloat16", ["x"]),
        ("__int2bfloat16_rn", ["x"]),
        ("__float2bfloat16", ["x", "y"]),
        ("__half2float", ["x"]),
    ],
)
def test_other_calls_are_not_bfloat16_conversions(name, args):
    assert format_bfloat16_conversion(name, args) is None


def test_directed_bfloat16_rounding_stays_refused(tmp_path):
    source = "#include <cuda_bf16.h>\n__global__ void k(__nv_bfloat16* a, float* o) { a[0] = __float2bfloat16_rz(o[0]); }\n"
    with pytest.raises(Exception) as raised:
        _translate(tmp_path, source, "cgl")
    assert getattr(raised.value, "code", "").endswith("directed-rounding-unsupported")


def test_sizeof_operands_import_as_crossgl_types(tmp_path):
    crossgl = _translate(tmp_path, SIZEOF_KERNEL, "cgl")
    for spelling in (
        "s[0] = sizeof(u64);",
        "s[1] = sizeof(i64);",
        "s[2] = sizeof(i64);",
        "s[3] = sizeof(u32);",
        "s[4] = sizeof(bfloat16);",
        "s[5] = sizeof(vec2<bfloat16>);",
        # A spelling CrossGL already sizes is kept as written.
        "s[6] = sizeof(float);",
    ):
        assert spelling in crossgl, crossgl
    metal = _translate(tmp_path, SIZEOF_KERNEL, "metal")
    assert [int(v) for v in re.findall(r"s\[\d\] = (\d+);", metal)] == [
        8,
        8,
        8,
        4,
        2,
        4,
        4,
    ]


def test_hip_sizeof_folds_size_t_and_ptrdiff_t_to_eight(tmp_path):
    crossgl = _translate(
        tmp_path,
        "__global__ void k(unsigned long long* s) { s[0] = sizeof(size_t) + sizeof(ptrdiff_t); }\n",
        "cgl",
        ".hip",
    )
    assert "s[0] = (8 + 8);" in crossgl


def test_generic_bfloat16_vectors_have_a_size():
    assert sizeof_type_bytes("vec2<bfloat16>") == 4
    assert sizeof_type_bytes("vec4<bfloat16>") == 8
    assert sizeof_type_bytes("bfloat16vec2") == 4


def test_spirv_of_a_size_t_kernel_validates(tmp_path):
    validator = shutil.which("spirv-val")
    assembler = shutil.which("spirv-as")
    if validator is None or assembler is None:
        pytest.skip("SPIRV-Tools is not installed")
    text = tmp_path / "k.spvasm"
    text.write_text(_translate(tmp_path, INDEX_KERNEL, "vulkan"), encoding="utf-8")
    binary = tmp_path / "k.spv"
    subprocess.run(
        [assembler, "--target-env", "vulkan1.1", str(text), "-o", str(binary)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [validator, "--target-env", "vulkan1.1", str(binary)],
        check=True,
        capture_output=True,
    )
