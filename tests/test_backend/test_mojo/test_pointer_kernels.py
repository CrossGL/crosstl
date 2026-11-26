"""Mojo kernels with ``UnsafePointer`` buffers import as CrossGL compute entries (crosstl/backend/Mojo/pointer_kernels.py).

Each pointer parameter becomes a storage buffer bound in parameter order, read-only when its origin is immutable, the
workgroup size is the host's, and ``max.gpu``'s index values are CrossGL's built-ins, in the kernel and in the
functions it calls. Every import is translated on to HLSL and back to Mojo, so the CrossGL it writes is checked by
the CrossGL parser and two code generators.
"""

from __future__ import annotations

import re

import pytest

import crosstl
from crosstl.backend.import_refusals import MojoImportRefused
from crosstl.backend.Mojo.pointer_kernels import unsafe_pointer_arguments


def _import(tmp_path, source, name="kernel.mojo"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend="cgl")


def _onward(tmp_path, crossgl, backend):
    path = tmp_path / "imported.cgl"
    path.write_text(crossgl)
    return crosstl.translate(str(path), backend=backend)


def _flat(code):
    return " ".join(code.split())


_COPY = """from max.gpu import global_idx


def probe(fin: UnsafePointer[Float32, ImmutAnyOrigin], outp: UnsafePointer[Float32, MutAnyOrigin]):
    var i: Int = global_idx.x
    outp[i] = abs(fin[i + 0])
"""


def test_pointer_parameters_become_storage_buffers_of_a_compute_entry(tmp_path):
    crossgl = _flat(_import(tmp_path, _COPY))
    assert (
        "@compute @host_workgroup_size fn probe("
        "@group(0) @binding(0) var<storage, read> fin: array<float>, "
        "@group(0) @binding(1) var<storage, read_write> outp: array<float>) {"
    ) in crossgl
    assert "int i = gl_GlobalInvocationID.x;" in crossgl
    assert "UnsafePointer" not in crossgl
    assert "MutAnyOrigin" not in crossgl

    hlsl = _onward(tmp_path, _import(tmp_path, _COPY), "directx")
    assert "StructuredBuffer<float> fin : register(t0);" in hlsl
    assert "RWStructuredBuffer<float> outp : register(u1);" in hlsl
    assert "SV_DispatchThreadID" in hlsl

    mojo = _onward(tmp_path, _import(tmp_path, _COPY), "mojo")
    assert re.search(
        r"def probe\(fin: UnsafePointer\[Float32, \w+\], outp: UnsafePointer\[Float32, MutAnyOrigin\]\)",
        mojo,
    )
    assert "global_idx.x" in mojo


def test_scalar_and_vector_pointees_and_value_parameters(tmp_path):
    source = """from max.gpu import global_idx, grid_dim


def probe(iin: UnsafePointer[Int32, MutAnyOrigin], vout: UnsafePointer[SIMD[DType.float32, 4], MutAnyOrigin], scale: Float32):
    var i: Int = global_idx.x
    vout[i] = SIMD[DType.float32, 4](Float32(iin[i]) * scale + Float32(grid_dim.x))
"""
    crossgl = _flat(_import(tmp_path, source))
    assert "var<storage, read_write> iin: array<int>" in crossgl
    assert "var<storage, read_write> vout: array<vec4>" in crossgl
    assert "float scale)" in crossgl
    assert "gl_NumWorkGroups.x" in crossgl
    _onward(tmp_path, _import(tmp_path, source), "directx")


def test_a_helper_the_kernel_calls_reads_the_gpu_index_values(tmp_path):
    source = """from max.gpu import global_idx, thread_idx


def lane() -> Int:
    return thread_idx.x


def probe(outp: UnsafePointer[Float32, MutAnyOrigin]):
    var i: Int = global_idx.x
    outp[i] = Float32(lane())
"""
    crossgl = _flat(_import(tmp_path, source))
    assert crossgl.startswith("int lane() { return gl_LocalInvocationID.x; }")
    assert (
        "fn probe(@group(0) @binding(0) var<storage, read_write> outp: array<float>)"
        in crossgl
    )
    # The kernel reads a workgroup-relative index through the helper, so it runs at the host's workgroup size, which
    # GLSL takes from specialization constants (crosstl/translator/host_workgroup_size.py).
    glsl = _onward(tmp_path, _import(tmp_path, source), "opengl")
    assert "int lane() { return int(gl_LocalInvocationID.x); }" in _flat(glsl)
    assert "local_size_x_id = 0" in glsl
    mojo = _onward(tmp_path, _import(tmp_path, source), "mojo")
    assert "thread_idx.x" in mojo


@pytest.mark.parametrize(
    "parameter, message",
    [
        pytest.param(
            "outp: UnsafePointer[Float32, MutAnyOrigin, address_space=AddressSpace.SHARED]",
            "points into AddressSpace.SHARED",
            id="shared-address-space",
        ),
        pytest.param(
            "outp: UnsafePointer[Float32, MutAnyOrigin], n: Int = 4",
            "has a default value",
            id="default-value",
        ),
    ],
)
def test_parameters_without_a_buffer_form_are_refused(tmp_path, parameter, message):
    source = f"""from max.gpu import global_idx


def probe({parameter}):
    var i: Int = global_idx.x
    outp[i] = 1.0
"""
    with pytest.raises(MojoImportRefused, match=re.escape(message)):
        _import(tmp_path, source)


@pytest.mark.parametrize(
    "vtype, expected",
    [
        ("UnsafePointer[Float32, MutAnyOrigin]", ("Float32", "MutAnyOrigin", "")),
        ("UnsafePointer[Float32]", ("Float32", "", "")),
        (
            "UnsafePointer[SIMD[DType.float32, 4], ImmutAnyOrigin]",
            ("SIMD[DType.float32, 4]", "ImmutAnyOrigin", ""),
        ),
        (
            "UnsafePointer[mut=False, type=Int32, address_space=AddressSpace.GLOBAL]",
            ("Int32", "ImmutAnyOrigin", "AddressSpace.GLOBAL"),
        ),
        ("TileTensor[mut=True, dtype, Layout, MutAnyOrigin]", None),
        ("Float32", None),
        (None, None),
    ],
)
def test_unsafe_pointer_arguments(vtype, expected):
    assert unsafe_pointer_arguments(vtype) == expected


_UNTYPED_INDEX = """from std.gpu import block_dim, block_idx, thread_idx


def probe(fin: UnsafePointer[Float32, ImmutAnyOrigin], outp: UnsafePointer[Float32, MutAnyOrigin]):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    var pair = 2 * thread_idx.x + 1
    outp[tid] = fin[pair]
"""


def test_an_untyped_local_of_index_arithmetic_is_unsigned(tmp_path):
    # std.gpu's block_idx, block_dim and thread_idx fields are UInt, and so is arithmetic on them with literals.
    crossgl = _flat(_import(tmp_path, _UNTYPED_INDEX))
    assert (
        "uint tid = ((gl_WorkGroupID.x * gl_WorkGroupSize.x) + gl_LocalInvocationID.x);"
        in crossgl
    )
    assert "uint pair = ((2 * gl_LocalInvocationID.x) + 1);" in crossgl
    assert "uint tid" in _flat(_onward(tmp_path, crossgl, "metal"))
