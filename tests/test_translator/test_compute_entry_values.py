"""Plain-value parameters of ``@ compute fn`` entries reach HLSL, MSL, SPIR-V and Slang as one constant block."""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics import ComputeEntryValueNameError

KERNEL = """
{globals}
@compute
@workgroup_size(64, 1, 1)
fn scale(
    @group({group}) @binding(0) var<storage, read_write> A: array<f32>,
    @group({group}) @binding(1) var<storage, read_write> B: array<f32>,
    i32 n,
    s: f32,
    vec2 v
) {{
    var i: i32 = gl_GlobalInvocationID.x;
    if ((i < n)) {{
        B[i] = A[i] * s + v.x;
    }}
}}
"""

WRITTEN = """
@compute
@workgroup_size(64, 1, 1)
fn drain(@group(0) @binding(0) var<storage, read_write> A: array<i32>, i32 count) {
    var i: u32 = gl_GlobalInvocationID.x;
    while (count > 0) {
        A[i] = A[i] + count;
        count = count - 1;
    }
    count++;
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "kernel.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def _kernel(group=0, globals=""):
    return KERNEL.format(group=group, globals=globals)


def test_hlsl_reads_the_values_from_a_constant_buffer(tmp_path):
    code = _translate(tmp_path, _kernel(), "directx")
    assert re.search(
        r"cbuffer scale_Args : register\(b2\)\s*\{\s*int n;\s*float s;\s*float2 v;\s*\};",
        code,
    )
    entry = re.search(r"void CSMain\(([^)]*)\)", code).group(1)
    assert entry == "uint3 dispatchThreadID : SV_DispatchThreadID"


def test_metal_reads_the_values_from_a_constant_buffer(tmp_path):
    code = _translate(tmp_path, _kernel(), "metal")
    assert re.search(
        r"struct scale_Args \{\s*int n;\s*float s;\s*float2 v;\s*\};", code
    )
    assert "constant scale_Args &scale_Args [[buffer(2)]]" in code
    assert "scale_Args.n" in code and "scale_Args.s" in code
    assert not re.search(r"\bint n\s*[,)]", code)


def test_spirv_reads_the_values_from_a_uniform_block(tmp_path):
    code = _translate(tmp_path, _kernel(group=1), "vulkan")
    entry = re.search(r"OpEntryPoint GLCompute %\w+ \"scale\"(.*)", code).group(1)
    assert entry.split() == ["%gl_GlobalInvocationID"]
    assert "Input_int" not in code
    assert re.search(r"OpDecorate %scale_Args\w* DescriptorSet 1", code)
    assert re.search(r"OpDecorate %scale_Args\w* Binding 2", code)
    assert re.search(r"OpDecorate %scale_Args Block", code)


def test_slang_declares_the_values_as_a_constant_buffer(tmp_path):
    code = _translate(tmp_path, _kernel(), "slang")
    assert "cbuffer scale_Args : register(b2)" in code
    assert not re.search(r"\bint n\s*[,)]", code)


def test_block_takes_the_first_free_binding(tmp_path):
    source = _kernel(globals="RWStructuredBuffer<float> C @binding(2);")
    assert "cbuffer scale_Args : register(b3)" in _translate(
        tmp_path, source, "directx"
    )
    assert "[[buffer(3)]]" in _translate(tmp_path, source, "metal")
    assert re.search(
        r"OpDecorate %scale_Args\w* Binding 3", _translate(tmp_path, source, "vulkan")
    )


@pytest.mark.parametrize("backend", ["directx", "metal", "vulkan"])
def test_a_written_value_becomes_a_local_copy(tmp_path, backend):
    code = _translate(tmp_path, WRITTEN, backend)
    if backend == "directx":
        assert "int count_value = count;" in code
        assert "count_value = (count_value - 1);" in code
        assert "count_value++;" in code
    elif backend == "metal":
        assert "int count_value = drain_Args.count;" in code
        assert "drain_Args.count =" not in code
    else:
        assert re.search(r"OpName %count_value \"count_value\"", code)


def test_a_value_named_like_a_global_is_refused(tmp_path):
    source = _kernel(globals="const int n = 4;")
    with pytest.raises(ComputeEntryValueNameError, match="passes n by value"):
        _translate(tmp_path, source, "directx")


def test_glsl_keeps_its_own_value_blocks(tmp_path):
    code = _translate(tmp_path, _kernel(), "opengl")
    assert "scale_Args" not in code
    assert "uniform scale_n_Args" in code


def _metal_compiles(path):
    return subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(path), "-o", "/dev/null"],
        capture_output=True,
        text=True,
    )


@pytest.mark.skipif(
    shutil.which("spirv-as") is None or shutil.which("spirv-val") is None,
    reason="SPIRV-Tools is not installed",
)
@pytest.mark.parametrize(
    "source", [_kernel(group=1), WRITTEN], ids=["values", "written"]
)
def test_spirv_validates(tmp_path, source):
    assembly = tmp_path / "kernel.spvasm"
    binary = tmp_path / "kernel.spv"
    assembly.write_text(_translate(tmp_path, source, "vulkan"))
    for command in (
        ["spirv-as", "--target-env", "vulkan1.1", str(assembly), "-o", str(binary)],
        ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
    ):
        subprocess.run(command, check=True)


@pytest.mark.skipif(
    shutil.which("xcrun") is None
    or subprocess.run(
        ["xcrun", "-sdk", "macosx", "-f", "metal"], capture_output=True
    ).returncode,
    reason="the Metal compiler is not installed",
)
@pytest.mark.parametrize("source", [_kernel(), WRITTEN], ids=["values", "written"])
def test_metal_compiles(tmp_path, source):
    path = tmp_path / "kernel.metal"
    path.write_text(_translate(tmp_path, source, "metal"))
    result = _metal_compiles(path)
    assert result.returncode == 0, result.stderr
