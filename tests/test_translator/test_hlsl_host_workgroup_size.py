"""HLSL reads a thread-group size that the source's host chooses from macros the host defines at compile time.

Direct3D has no specialization constants, but the host compiles the HLSL source itself, so it passes the size it
dispatched the Metal kernel with as ``CROSSGL_WORKGROUP_SIZE_{X,Y,Z}``, the macros the generated GLSL reads when no
SPIR-V specialization step runs. Without them the source stops at an ``#error``.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.project.host_reflection import reflect_target_host_interface
from crosstl.project.native_loader_dispatch import (
    NativeLoaderDispatchError,
    _reflected_workgroup_size,
)
from crosstl.translator.codegen.hlsl_host_workgroup_size import (
    hlsl_entry_local_size,
    hlsl_entry_proof_local_size,
    hlsl_host_numthreads,
)
from crosstl.translator.codegen.host_workgroup_size_ids import (
    HOST_WORKGROUP_SIZE_MACROS,
)
from crosstl.translator.host_workgroup_size import HOST_WORKGROUP_SIZE_KEY

HEADER = "#include <metal_stdlib>\nusing namespace metal;\n"

ROTATE = """
kernel void rotate(device float* data [[buffer(0)]],
                   uint gid [[thread_position_in_grid]],
                   uint lid [[thread_index_in_threadgroup]],
                   uint3 count [[threads_per_threadgroup]]) {
    threadgroup float tile[256];
    tile[lid] = data[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    data[gid] = tile[(lid + 1) % count.x];
}
"""

REDUCE = """
kernel void reduce(device const float* src [[buffer(0)]],
                   device float* dst [[buffer(1)]],
                   uint gid [[thread_position_in_grid]],
                   uint group [[threadgroup_position_in_grid]],
                   uint simdgroup [[simdgroup_index_in_threadgroup]],
                   uint simdgroups [[simdgroups_per_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[32];
    float value = simd_sum(src[gid]);
    if (lane == 0) partial[simdgroup] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        float total = simd_sum(lane < simdgroups ? partial[lane] : 0.0f);
        if (lane == 0) dst[group] = total;
    }
}
"""

SCALE = """
kernel void scale(device float* data [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    data[gid] = data[gid] * 2.0f;
}
"""

MACROS = ", ".join(HOST_WORKGROUP_SIZE_MACROS)
HOST_NUMTHREADS = f"[numthreads({MACROS})]"
HOST_SIZE = f"uint3({MACROS})"
GUARD = (
    "#if !defined(CROSSGL_WORKGROUP_SIZE_X) || !defined(CROSSGL_WORKGROUP_SIZE_Y) || "
    "!defined(CROSSGL_WORKGROUP_SIZE_Z)\n"
    '#error "the host chooses this thread-group size: define CROSSGL_WORKGROUP_SIZE_{X,Y,Z}"\n'
    "#endif\n"
)


def _translate(tmp_path, source, **options):
    path = tmp_path / "kernel.metal"
    path.write_text(HEADER + source)
    return crosstl.translate(
        str(path),
        backend="directx",
        format_output=False,
        source_options=options or None,
    )


def test_size_dependent_kernel_reads_the_host_macros(tmp_path):
    code = _translate(tmp_path, ROTATE)
    assert GUARD + HOST_NUMTHREADS + "\n" in code
    assert f"uint3 count = {HOST_SIZE};" in code
    assert "[numthreads(1, 1, 1)]" not in code


def test_flattened_group_size_reads_the_host_macros(tmp_path):
    code = _translate(tmp_path, REDUCE)
    assert HOST_NUMTHREADS in code
    group_size = " * ".join(f"uint({macro})" for macro in HOST_WORKGROUP_SIZE_MACROS)
    assert f"((({group_size}) + WaveGetLaneCount() - 1u) / WaveGetLaneCount())" in code


@pytest.mark.parametrize("source", [ROTATE, REDUCE])
def test_threadgroup_size_option_keeps_literal_numthreads(tmp_path, source):
    code = _translate(tmp_path, source, threadgroup_size=(64, 1, 1))
    assert "[numthreads(64, 1, 1)]" in code
    assert "CROSSGL_WORKGROUP_SIZE" not in code


def test_size_independent_kernel_keeps_one_invocation_per_group(tmp_path):
    code = _translate(tmp_path, SCALE)
    assert "[numthreads(1, 1, 1)]" in code
    assert "CROSSGL_WORKGROUP_SIZE" not in code


def test_host_sized_config_has_no_size_for_proofs():
    config = {HOST_WORKGROUP_SIZE_KEY: {"entry": "k", "reason": "declares tile"}}
    assert hlsl_entry_local_size(config) == HOST_WORKGROUP_SIZE_MACROS
    assert hlsl_entry_proof_local_size(config) is None
    explicit = {**config, "local_size": (8, 4, 1)}
    assert hlsl_entry_local_size(explicit) == ("8", "4", "1")
    assert hlsl_entry_proof_local_size(explicit) == ("8", "4", "1")
    assert hlsl_host_numthreads() == GUARD + HOST_NUMTHREADS + "\n"


def test_reflection_reports_the_host_macros_and_no_size(tmp_path):
    artifact = tmp_path / "kernel.hlsl"
    artifact.write_text(crosstl.translate(_write(tmp_path, ROTATE), backend="directx"))
    record = reflect_target_host_interface(artifact, target="directx")
    assert record["status"] == "ready"
    (entry,) = record["entryPoints"]
    config = entry["executionConfig"]
    assert "numthreads" not in config
    assert config[HOST_WORKGROUP_SIZE_KEY]["macros"] == list(HOST_WORKGROUP_SIZE_MACROS)
    with pytest.raises(NativeLoaderDispatchError) as error:
        _reflected_workgroup_size(entry)
    assert error.value.code.endswith("workgroup-size-host-defined")


def _write(tmp_path, source):
    path = tmp_path / "kernel.metal"
    path.write_text(HEADER + source)
    return str(path)


def _dxc(dxc, artifact, output, *defines):
    command = [
        dxc,
        "-T",
        "cs_6_0",
        "-E",
        "CSMain",
        "-WX",
        str(artifact),
        "-Fc",
        str(output),
    ]
    for name, value in zip(HOST_WORKGROUP_SIZE_MACROS, defines):
        command += ["-D", f"{name}={value}"]
    return subprocess.run(command, capture_output=True, text=True, timeout=300)


@pytest.mark.parametrize("source", [ROTATE, REDUCE])
def test_dxc_compiles_the_size_the_host_defines(tmp_path, source):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")
    artifact = tmp_path / "kernel.hlsl"
    artifact.write_text(_translate(tmp_path, source), encoding="utf-8")
    listing = tmp_path / "kernel.dxil.txt"

    result = _dxc(dxc, artifact, listing, 128, 2, 1)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NumThreads=(128,2,1)" in listing.read_text()

    result = _dxc(dxc, artifact, tmp_path / "undefined.txt")
    assert result.returncode != 0
    assert "the host chooses this thread-group size" in result.stdout + result.stderr
