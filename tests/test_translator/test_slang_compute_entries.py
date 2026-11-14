"""Top-level ``@ compute fn`` entries reach Slang as compute entry points whose resources are module globals.

The Slang generator used to emit such an entry, the form every CUDA, HIP, OpenCL and Metal kernel import takes,
as an ordinary function: no ``[shader("compute")]``, no ``[numthreads]``, and its buffers as plain array
parameters. ``crosstl/translator/semantics/compute_entry_resources.py`` now declares the bound parameters as
module globals, and the generator emits the entry with its workgroup size.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics import ComputeEntryResourceError

KERNEL = """
@compute
@workgroup_size(8, 2)
fn k(@group(0) @binding(0) var<storage, read> a: array<f32>,
     @group(0) @binding(1) var<storage, read_write> o: array<u32>,
     Texture2D<float4> t @binding(2), SamplerState s @binding(3),
     gid: vec3<u32> @builtin(global_invocation_id),
     lid: vec3<u32> @builtin(local_invocation_id)) {
    o[gid.x] = u32(a[gid.x] + t.SampleLevel(s, vec2(0.5), 0.0).x) + lid.x;
}
"""

CUDA_KERNEL = """
__global__ void vadd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
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


def _translate(tmp_path, source, suffix="cgl"):
    path = tmp_path / f"kernel.{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend="slang", format_output=False)


def _slangc():
    return shutil.which("slangc")


def test_the_entry_is_a_compute_entry_point_with_its_size(tmp_path):
    code = _translate(tmp_path, KERNEL)
    assert re.search(
        r'\[numthreads\(8, 2, 1\)\]\s*\[shader\("compute"\)\]\s*void k\(', code
    ), code
    entry = re.search(r"void k\(([^)]*)\)", code).group(1)
    assert entry == "uint3 gid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID"


def test_bound_parameters_become_bound_globals(tmp_path):
    code = _translate(tmp_path, KERNEL)
    assert re.search(
        r"\[\[vk::binding\(0, 0\)\]\] StructuredBuffer<float> a : register\(t0", code
    ), code
    assert re.search(
        r"\[\[vk::binding\(1, 0\)\]\] RWStructuredBuffer<uint> o : register\(u1", code
    ), code
    assert re.search(r"\[\[vk::binding\(2, 0\)\]\] Texture2D<float4> t\b", code)
    assert re.search(r"\[\[vk::binding\(3, 0\)\]\] SamplerState s\b", code)


def test_an_imported_cuda_kernel_is_an_entry(tmp_path):
    code = _translate(tmp_path, CUDA_KERNEL, "cu")
    assert re.search(r'\[shader\("compute"\)\]\s*void vadd\(', code), code
    for index, name in enumerate("abc"):
        assert re.search(
            rf"\[\[vk::binding\({index}, 0\)\]\] RWStructuredBuffer<float> {name} ",
            code,
        ), code
    assert "cbuffer vadd_Args" in code
    assert not re.search(r"float \w+\[\]", code)


def test_entries_share_an_identical_resource(tmp_path):
    code = _translate(tmp_path, TWO_ENTRIES.replace("{binding}", "0"))
    assert code.count("RWStructuredBuffer<float> data") == 1, code
    assert re.search(
        r'\[numthreads\(64, 1, 1\)\]\s*\[shader\("compute"\)\]\s*void first', code
    )
    assert re.search(
        r'\[numthreads\(32, 1, 1\)\]\s*\[shader\("compute"\)\]\s*void second', code
    )


def test_entries_binding_one_name_differently_are_refused(tmp_path):
    with pytest.raises(ComputeEntryResourceError, match="differently"):
        _translate(tmp_path, TWO_ENTRIES.replace("{binding}", "1"))


def test_a_parameter_binding_a_taken_slot_is_refused(tmp_path):
    source = "RWStructuredBuffer<float> other @binding(0);\n" + KERNEL
    with pytest.raises(ComputeEntryResourceError, match="already uses"):
        _translate(tmp_path, source)


def test_bound_storage_that_is_no_array_is_refused(tmp_path):
    source = """
struct Params { float scale; };
@compute
@workgroup_size(1)
fn k(@group(0) @binding(0) var<storage, read_write> p: Params) {
    p.scale = 1.0;
}
"""
    with pytest.raises(ComputeEntryResourceError, match="runtime-sized array"):
        _translate(tmp_path, source)


@pytest.mark.skipif(_slangc() is None, reason="slangc is not available")
@pytest.mark.parametrize(
    ("source", "suffix", "entry"),
    [(KERNEL, "cgl", "k"), (CUDA_KERNEL, "cu", "vadd")],
    ids=["cgl", "cuda"],
)
def test_slangc_compiles_the_entry(tmp_path, source, suffix, entry):
    shader = tmp_path / "out.slang"
    shader.write_text(_translate(tmp_path, source, suffix))
    result = subprocess.run(
        [
            _slangc(),
            str(shader),
            "-target",
            "spirv",
            "-entry",
            entry,
            "-stage",
            "compute",
            "-o",
            str(tmp_path / "out.spv"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
