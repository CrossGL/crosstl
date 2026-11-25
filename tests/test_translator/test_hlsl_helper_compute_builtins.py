"""Helpers of a function-form stage entry receive the stage built-ins they read as parameters.

The CUDA, HIP, OpenCL C and Mojo imports write a kernel as a function-form entry (``@compute fn probe(...)``) beside
module-scope helpers, and a ``__device__`` helper that reads ``threadIdx`` reads ``gl_LocalInvocationID`` by name.
HLSL, MSL, Slang and WGSL deliver that value only as an entry-point parameter (``SV_GroupThreadID``). The built-in
threading pass handled stage blocks only, so the helper kept the bare name and DXC rejected it as an undeclared
identifier.
"""

import re
import shutil
import subprocess

import pytest

import crosstl

COMPUTE = """
i32 lane() {
    return gl_LocalInvocationID.x;
}

u32 slot() {
    return u32(lane()) + gl_WorkGroupID.x;
}

@compute
@workgroup_size(64, 1, 1)
fn probe(
    @group(0) @binding(0) var<storage, read_write> outp: array<f32>
) {
    var i: u32 = gl_GlobalInvocationID.x;
    outp[i] = f32(slot());
}
"""

# Two kernels share one helper: the second passes the built-in to the parameter the first one added.
SHARED_HELPER = """
u32 lane() {
    return gl_LocalInvocationID.x;
}

@compute
@workgroup_size(64, 1, 1)
fn first(
    @group(0) @binding(0) var<storage, read_write> outp: array<f32>
) {
    outp[gl_GlobalInvocationID.x] = f32(lane());
}

@compute
@workgroup_size(64, 1, 1)
fn second(
    @group(0) @binding(0) var<storage, read_write> outp: array<f32>
) {
    outp[gl_GlobalInvocationID.x] = f32(lane() + 1u);
}
"""

FRAGMENT = """
float side() { return gl_FrontFacing ? 1.0 : 0.0; }

@fragment
fn ps(@location(0) vec2 uv) -> @location(0) vec4 {
    return vec4(side(), float(gl_SampleID), uv.x, 1.0);
}
"""

VERTEX = """
float vid() { return float(gl_VertexID); }

@vertex
fn vs() -> @gl_Position vec4 {
    return vec4(vid(), float(gl_InstanceID), 0.0, 1.0);
}
"""

CUDA = """
__device__ int lane() { return threadIdx.x; }

__global__ void probe(float* outp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    outp[i] = (float)lane();
}
"""


def _translate(tmp_path, source, backend, suffix="cgl"):
    path = tmp_path / f"shader.{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _dxc(tmp_path, code, profile, entry, *defines):
    """Compile ``code`` with DXC when it is on PATH."""
    if not shutil.which("dxc"):
        return
    (tmp_path / "out.hlsl").write_text(code)
    command = ["dxc", "-T", profile, "-E", entry, "-WX", "out.hlsl", "-Fo", "out.dxil"]
    command += [f"-D{define}" for define in defines]
    result = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr + code


def _function(code, name):
    """The parameter list and body of the definition of ``name``."""
    match = re.search(rf"\b{name}\(([^)]*)\)[^{{;]*\{{(.*?)\n\}}", code, re.S)
    assert match, code
    return match.group(1), match.group(2)


@pytest.mark.parametrize("backend", ["directx", "metal", "slang", "wgsl"])
def test_helpers_of_function_form_compute_entries_receive_builtins(tmp_path, backend):
    code = _translate(tmp_path, COMPUTE, backend)

    assert not re.search(r"\bgl_(LocalInvocationID|WorkGroupID)\b", code), code
    lane, _ = _function(code, "lane")
    slot, slot_body = _function(code, "slot")
    assert "cgl_LocalInvocationID" in lane, code
    # slot reads the group id itself and passes lane's local id on.
    assert "cgl_LocalInvocationID" in slot and "cgl_WorkGroupID" in slot, code
    assert re.search(r"\blane\(cgl_LocalInvocationID\w*\)", slot_body), code


def test_hlsl_function_form_compute_helpers_compile(tmp_path):
    code = _translate(tmp_path, COMPUTE, "directx")

    assert re.search(r"uint3 cgl_LocalInvocationID\w* : SV_GroupThreadID", code), code
    assert re.search(r"uint3 cgl_WorkGroupID\w* : SV_GroupID", code), code
    _dxc(tmp_path, code, "cs_6_6", "CSMain")


def test_hlsl_kernels_share_a_threaded_helper(tmp_path):
    code = _translate(tmp_path, SHARED_HELPER, "directx")

    assert not re.search(r"\bgl_LocalInvocationID\b", code), code
    # One helper definition, with one parameter, called with each kernel's own SV_GroupThreadID value.
    assert len(re.findall(r"\buint lane\(uint3 \w+\)\s*\{", code)) == 1, code
    assert code.count(": SV_GroupThreadID") == 2, code


def test_hlsl_cuda_device_helper_reading_thread_index_compiles(tmp_path):
    code = _translate(tmp_path, CUDA, "directx", suffix="cu")

    assert not re.search(r"\bgl_\w+", code), code
    lane, _ = _function(code, "lane")
    assert lane.strip().startswith("uint3 "), code
    _dxc(
        tmp_path,
        code,
        "cs_6_6",
        "CSMain",
        "CROSSGL_WORKGROUP_SIZE_X=64",
        "CROSSGL_WORKGROUP_SIZE_Y=1",
        "CROSSGL_WORKGROUP_SIZE_Z=1",
    )


@pytest.mark.parametrize(
    ("source", "profile", "entry", "semantics"),
    [
        (FRAGMENT, "ps_6_0", "PSMain", ("SV_IsFrontFace", "SV_SampleIndex")),
        (VERTEX, "vs_6_0", "VSMain", ("SV_VertexID", "SV_InstanceID")),
    ],
)
def test_hlsl_function_form_graphics_entries_bind_builtins(
    tmp_path, source, profile, entry, semantics
):
    code = _translate(tmp_path, source, "directx")

    assert not re.search(r"\bgl_\w+", code), code
    for semantic in semantics:
        assert f": {semantic}" in code, code
    _dxc(tmp_path, code, profile, entry)
