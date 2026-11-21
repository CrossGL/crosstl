"""WGSL output of ``@ compute fn`` entries whose buffers and plain values are parameters.

Every CUDA, HIP, OpenCL and Metal kernel import, and a CrossGL entry written in their style, is a top-level
``@ compute fn`` whose buffers carry ``@ group``/``@ binding`` and whose plain values (``int n``) the host sets per
dispatch. WGSL output kept both as entry parameters, ``fn compute_main(..., o_in: array<i32>, n: i32)`` with
``var o: array<i32> = o_in;``, which naga rejects ("Entry point arguments and return values must all have
bindings"). WGSL now takes the two shared passes the other graphics targets take:
``semantics/compute_entry_resources.py`` declares the bound parameters as module bindings and
``semantics/compute_entry_values.py`` moves the plain values into the ``<entry>_Args`` uniform block.
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess

import pytest

import crosstl

KERNEL = """
@compute
@workgroup_size(64, 1, 1)
fn k(
    @group(0) @binding(0) var<storage, read_write> o: array<i32>,
    @group(0) @binding(1) var<storage, read> src: array<f32>
) {
    var t: i32 = gl_GlobalInvocationID.x;
    o[t] = i32(src[t]) + t;
}
"""

CUDA_KERNEL = """
__global__ void saxpy(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
"""

TWO_ENTRIES = """
@compute
@workgroup_size(64)
fn first(@group(0) @binding(0) var<storage, read_write> data: array<f32>,
         gid: vec3<u32> @builtin(global_invocation_id)) {
    data[gid.x] = data[gid.x] * 2.0;
}

@compute
@workgroup_size(32)
fn second(@group(0) @binding({binding}) var<storage, read_write> data: array<f32>,
          gid: vec3<u32> @builtin(global_invocation_id)) {
    data[gid.x] = data[gid.x] + 1.0;
}
"""

ONE_VALUE = """
struct Params { float scale; };
@compute
@workgroup_size(1)
fn k(@group(0) @binding(0) var<storage, read_write> p: Params) {
    p.scale = 1.0;
}
"""


def _translate(tmp_path, source, suffix="cgl"):
    path = tmp_path / f"kernel.{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="wgsl", format_output=False)


def _naga(tmp_path, code, output=None):
    naga = shutil.which("naga")
    if naga is None:
        pytest.skip("naga is not installed")
    shader = tmp_path / "out.wgsl"
    shader.write_text(code, encoding="utf-8")
    command = [naga, str(shader)] + ([str(output)] if output else [])
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr or result.stdout
    return output


def test_bound_parameters_become_module_bindings(tmp_path):
    code = _translate(tmp_path, KERNEL)

    assert re.search(
        r"@group\(0\) @binding\(0\)\s*var<storage, read_write> o: array<i32>;", code
    )
    assert re.search(
        r"@group\(0\) @binding\(1\)\s*var<storage, read> src: array<f32>;", code
    )
    (parameters,) = re.findall(r"fn compute_main\((.*)\)\s*\{", code)
    assert "array<" not in parameters
    assert "o_in" not in code
    _naga(tmp_path, code)


def test_plain_values_of_an_imported_cuda_kernel_become_the_args_block(tmp_path):
    code = _translate(tmp_path, CUDA_KERNEL, "cu")

    assert re.search(r"struct saxpy_Args \{\s*a: f32,\s*n: i32,\s*\};", code)
    assert re.search(
        r"@group\(0\) @binding\(0\)\s*var<uniform> _saxpy_Args: saxpy_Args;", code
    )
    assert "_saxpy_Args.n" in code and "_saxpy_Args.a" in code
    (parameters,) = re.findall(r"fn compute_main\((.*)\)\s*\{", code)
    assert parameters.count(":") == 1 and "@builtin(global_invocation_id)" in parameters
    _naga(tmp_path, code)


def test_entries_share_an_identical_resource(tmp_path):
    code = _translate(tmp_path, TWO_ENTRIES.replace("{binding}", "0"))

    assert code.count("var<storage, read_write> data: array<f32>;") == 1
    assert len(re.findall(r"@compute", code)) == 2
    _naga(tmp_path, code)


def test_a_buffer_element_passed_to_an_out_parameter_writes_the_buffer(tmp_path):
    # The entry used to copy the runtime-sized buffer into a function variable, which WGSL cannot store.
    source = """
f32 store_half(f32 x, out f32 half_x) {
    half_x = x * 0.5;
    return x;
}

@compute
@workgroup_size(1, 1, 1)
fn probe(@group(0) @binding(0) var<storage, read_write> outp: array<f32>) {
    var i: u32 = gl_GlobalInvocationID.x;
    outp[(i + 1u)] = store_half(outp[i], outp[(i + 2u)]);
}
"""
    code = _translate(tmp_path, source)

    assert re.search(
        r"@group\(0\) @binding\(0\)\s*var<storage, read_write> outp: array<f32>;", code
    )
    assert "outp_in" not in code
    assert re.search(r"outp\[\w+\] = _crosstl_ref_\d+;", code), code
    _naga(tmp_path, code)


def test_bound_storage_holding_one_value_is_a_one_element_array(tmp_path):
    code = _translate(tmp_path, ONE_VALUE)

    assert re.search(r"var<storage, read_write> p: array<Params>;", code)
    assert "p[0].scale = 1.0;" in code
    _naga(tmp_path, code)


def _vulkan_runtime():
    try:
        from tests.conformance.runtimes import provision
    except ImportError:
        return None
    for runtime in ("vulkan-lavapipe", "vulkan-moltenvk"):
        try:
            provisioned = provision(runtime)
        except Exception:
            continue
        if provisioned.available and provisioned.mode == "host":
            env = dict(provisioned.env)
            if env.get("CROSSTL_CONFORMANCE_VKRUN"):
                return env
    return None


def test_imported_cuda_kernel_computes_saxpy_through_naga(tmp_path):
    env = _vulkan_runtime()
    if env is None:
        pytest.skip("no host Vulkan runtime with vkrun")
    code = _translate(tmp_path, CUDA_KERNEL, "cu")
    binary = _naga(tmp_path, code, tmp_path / "out.spv")
    count, a = 40, 3.0
    xs = [float(i) for i in range(64)]
    ys = [float(100 - i) for i in range(64)]
    x_file, y_file, args_file = (
        tmp_path / name for name in ("x.bin", "y.bin", "args.bin")
    )
    x_file.write_bytes(struct.pack("<64f", *xs))
    y_file.write_bytes(struct.pack("<64f", *ys))
    args_file.write_bytes(struct.pack("<fi8x", a, count))

    run = subprocess.run(
        [
            env["CROSSTL_CONFORMANCE_VKRUN"],
            str(binary),
            "compute_main",
            "64",
            "1",
            "1",
            f"0:uniform:{args_file}",
            f"1:inout:{x_file}",
            f"2:inout:{y_file}",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, **env},
    )

    assert run.returncode == 0 and run.stdout.startswith("ok"), run.stdout + run.stderr
    expected = [a * x + y if i < count else y for i, (x, y) in enumerate(zip(xs, ys))]
    assert list(struct.unpack("<64f", y_file.read_bytes())) == expected
