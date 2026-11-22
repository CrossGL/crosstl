"""CUDA and HIP output lower CrossGL ``as_type<T>(value)`` to the runtimes' bit-cast intrinsics.

The generators used to emit ``as_type<u64>(x)`` verbatim, which neither nvcc nor hipcc declares. See
crosstl/translator/codegen/cuda_type_reinterpretation.py.
"""

from __future__ import annotations

import pytest

import crosstl
from crosstl.translator.codegen.cuda_refusals import CudaLoweringRefused
from crosstl.translator.codegen.hip_refusals import HipLoweringRefused

PROGRAM = """{functions}
@compute
@workgroup_size(64, 1, 1)
fn k(@group(0) @binding(0) var<storage, read_write> o: array<u32>) {{
    var i: u32 = gl_GlobalInvocationID.x;
    o[i] = i;
}}
"""

SCALARS = [
    ("u32", "f32", "__float_as_uint(v)"),
    ("i32", "f32", "__float_as_int(v)"),
    ("f32", "u32", "__uint_as_float(v)"),
    ("f32", "i32", "__int_as_float(v)"),
    ("u32", "i32", "static_cast<unsigned int>(v)"),
    ("i64", "f64", "__double_as_longlong(v)"),
    ("u64", "f64", "static_cast<unsigned long long>(__double_as_longlong(v))"),
    ("f64", "i64", "__longlong_as_double(v)"),
    ("f64", "u64", "__longlong_as_double(static_cast<long long>(v))"),
    ("u64", "i64", "static_cast<unsigned long long>(v)"),
    ("u16", "f16", "__half_as_ushort(v)"),
    ("f16", "u16", "__ushort_as_half(v)"),
    ("u32", "u32", "return v;"),
]


def translate(tmp_path, functions, backend):
    path = tmp_path / "bits.cgl"
    path.write_text(PROGRAM.format(functions=functions), encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize("backend", ["cuda", "hip"])
@pytest.mark.parametrize(("target", "source", "expected"), SCALARS)
def test_a_scalar_reinterpretation_uses_the_bit_cast_intrinsic(
    tmp_path, backend, target, source, expected
):
    output = translate(
        tmp_path,
        f"{target} cast({source} v) {{ return as_type<{target}>(v); }}",
        backend,
    )
    assert expected in output
    assert "as_type" not in output


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_the_result_type_flows_into_the_surrounding_expression(tmp_path, backend):
    functions = "u64 flip(f64 x) { return as_type<u64>(x) ^ (u64(1u) << 63u); }"
    output = translate(tmp_path, functions, backend)
    assert "(static_cast<unsigned long long>(__double_as_longlong(x)) ^" in output


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_a_vector_of_32_bit_components_uses_the_vector_helper(tmp_path, backend):
    output = translate(
        tmp_path, "uvec2 cast(vec2 v) { return as_type<uvec2>(v); }", backend
    )
    assert "as_type" not in output
    assert "cgl_float2_to_uint2_bitcast(v)" in output


@pytest.mark.parametrize(
    ("backend", "refusal"), [("cuda", CudaLoweringRefused), ("hip", HipLoweringRefused)]
)
@pytest.mark.parametrize(
    "functions",
    [
        "u64 cast(f32 v) { return as_type<u64>(v); }",
        "f32 cast(uvec2 v) { return as_type<f32>(v); }",
    ],
)
def test_a_reinterpretation_between_sizes_is_refused(
    tmp_path, backend, refusal, functions
):
    with pytest.raises(refusal, match="bit reinterpretation: as_type"):
        translate(tmp_path, functions, backend)
