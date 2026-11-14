"""``var<storage>`` and module-scope ``var<uniform>`` declarations reach every target as the buffers they declare.

The CUDA and HIP importers write ``__device__`` memory, ``__constant__`` memory and kernel pointer parameters in
these forms. Fixed-size and single-value storage variables used to become private copies or ``$Globals`` members, so
stores never reached memory; module-scope uniforms lost their binding. See
``crosstl/translator/semantics/storage_variables.py``, ``crosstl/translator/semantics/uniform_variables.py`` and
``crosstl/backend/device_globals.py``.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics import StorageVariableUnsupportedError

MODULE = """
@group(0) @binding(0) var<storage, read_write> gbuf: array<f32, 16>;
@group(0) @binding(1) var<storage, read_write> counter: i32;
@group(0) @binding(2) var<uniform> coeffs: array<f32, 4>;
@group(0) @binding(4) var<storage, read> scale: vec4<f32>;

fn weight(i: i32) -> f32 {
    return gbuf[i] * scale.x + f32(gbuf.length());
}

@compute
@workgroup_size(16, 1, 1)
fn k(@group(0) @binding(3) var<storage, read_write> out: array<f32>) {
    var i: i32 = i32(gl_LocalInvocationID.x);
    gbuf[i] = out[i] * coeffs[i % 4];
    atomicAdd(counter, 1);
    workgroupBarrier();
    out[i] = weight(15 - i) + f32(counter);
}
"""

PARAMETERS = """
struct Pair { float a; float b; };

@compute
@workgroup_size(16, 1, 1)
fn k(@group(0) @binding(0) var<storage, read_write> gbuf: array<f32, 16>,
     @group(0) @binding(1) var<storage, read_write> counter: i32,
     @group(0) @binding(2) var<storage, read> pair: Pair,
     @group(0) @binding(3) var<storage, read_write> out: array<f32>) {
    var i: i32 = i32(gl_LocalInvocationID.x);
    gbuf[i] = out[i] * pair.a + f32(gbuf.length());
    counter = counter + 1;
    out[i] = gbuf[15 - i] + pair.b;
}
"""

BLOCK = """
struct Particles { uint count; float data[]; };

@group(0) @binding(0) var<storage, read_write> parts: Particles;

@compute
@workgroup_size(16, 1, 1)
fn k() {
    var i: u32 = gl_LocalInvocationID.x;
    parts.data[i] = parts.data[i] + f32(parts.count);
}
"""

INITIALIZED_UNIFORM = """
@group(0) @binding(0) var<uniform> table: array<f32, 4> = {1.0, 2.0, 3.0, 4.0};

@compute
@workgroup_size(4, 1, 1)
fn k(@group(0) @binding(1) var<storage, read_write> out: array<f32>) {
    var i: i32 = i32(gl_LocalInvocationID.x);
    out[i] = table[i];
}
"""


def _translate(tmp_path, source, backend, suffix="cgl"):
    path = tmp_path / f"kernel.{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def test_module_storage_variables_are_hlsl_buffers(tmp_path):
    code = _translate(tmp_path, MODULE, "directx")
    assert "RWStructuredBuffer<float> gbuf : register(u0);" in code
    assert "RWStructuredBuffer<int> counter : register(u1);" in code
    assert "StructuredBuffer<float4> scale : register(t4);" in code
    assert re.search(
        r"cbuffer coeffs_block : register\(b2\)\s*\{\s*float coeffs\[4\];", code
    )
    assert "InterlockedAdd(counter[0], 1);" in code
    assert "scale[0].x" in code and "float(counter[0])" in code
    # gbuf.length() is the declared element count, whatever size of buffer the host binds.
    assert "float(int(16))" in code
    assert "float gbuf[16]" not in code


def test_module_storage_variables_are_glsl_buffer_blocks(tmp_path):
    code = _translate(tmp_path, MODULE, "opengl")
    assert "layout(std430, binding = 0) buffer gbufBuffer { float gbuf[]; };" in code
    assert (
        "layout(std430, binding = 1) buffer counterBuffer { int counter[]; };" in code
    )
    assert "readonly buffer scaleBuffer { vec4 scale[]; };" in code
    assert re.search(
        r"layout\(std140, binding = 2\) uniform coeffs_block\s*\{\s*float coeffs\[4\];",
        code,
    )
    assert "atomicAdd(counter[0], 1);" in code
    assert "uniform float" not in code and "uniform int" not in code


def test_module_storage_variables_are_metal_buffer_arguments(tmp_path):
    code = _translate(tmp_path, MODULE, "metal")
    assert "device float* gbuf [[buffer(0)]]" in code
    assert "device int* counter [[buffer(1)]]" in code
    assert "const device float4* scale [[buffer(4)]]" in code
    assert "constant coeffs_block& coeffs_block [[buffer(2)]]" in code
    assert "&counter[0]" in code


def test_module_storage_variables_are_spirv_storage_blocks(tmp_path):
    code = _translate(tmp_path, MODULE, "vulkan")
    for binding in (0, 1, 2, 3, 4):
        assert f"Binding {binding}" in code
    assert (
        "Input" not in code.split("BuiltIn LocalInvocationId")[0].split("OpDecorate")[
            -1
        ]
    )


def test_cuda_and_crossgl_targets_keep_module_storage_variables(tmp_path):
    cuda = _translate(tmp_path, MODULE, "cuda")
    assert "__device__ float gbuf[16];" in cuda
    assert "__device__ int counter;" in cuda
    assert "__constant__ float coeffs[4];" in cuda
    crossgl = _translate(tmp_path, MODULE, "cgl")
    assert "var<storage, read_write> gbuf: array<f32, 16>" in crossgl


def test_entry_storage_parameters_bind_the_buffer(tmp_path):
    metal = _translate(tmp_path, PARAMETERS, "metal")
    # The fixed-size array used to be copied into a private array, so stores never reached the buffer.
    assert "float gbuf[16] =" not in metal
    assert "device float* gbuf [[buffer(0)]]" in metal
    assert "device int* counter [[buffer(1)]]" in metal
    assert "const device Pair* pair [[buffer(2)]]" in metal
    assert "counter[0] = counter[0] + 1;" in metal
    assert "pair[0].a" in metal and "float(int(16))" in metal
    hlsl = _translate(tmp_path, PARAMETERS, "directx")
    assert "RWStructuredBuffer<int> counter : register(u1);" in hlsl
    assert "StructuredBuffer<Pair> pair : register(t2);" in hlsl
    assert re.search(r"void CSMain\(uint3 \w+ : SV_GroupThreadID\)", hlsl)
    cuda = _translate(tmp_path, PARAMETERS, "cuda")
    # A scalar passed by value lost every store; the kernel now receives the buffer.
    assert re.search(
        r"__global__ void k\(float\* gbuf, int\* counter, (?:const )?Pair\* pair, float\* out\)",
        cuda,
    )
    assert "counter[0] = (counter[0] + 1);" in cuda


def test_storage_structure_is_a_buffer_block(tmp_path):
    glsl = _translate(tmp_path, BLOCK, "opengl")
    assert re.search(
        r"layout\(std430, binding = 0\) buffer Particles\s*\{\s*uint count;\s*float data\[\];",
        glsl,
    )
    assert "parts.data[i] = (parts.data[i] + float(parts.count));" in glsl
    hlsl = _translate(tmp_path, BLOCK, "directx")
    assert "RWByteAddressBuffer parts : register(u0);" in hlsl


def test_initialized_uniform_is_a_program_constant(tmp_path):
    glsl = _translate(tmp_path, INITIALIZED_UNIFORM, "opengl")
    assert "const float table[4] = float[4](1.0, 2.0, 3.0, 4.0);" in glsl
    hlsl = _translate(tmp_path, INITIALIZED_UNIFORM, "directx")
    assert "static const float table[4] = {1.0, 2.0, 3.0, 4.0};" in hlsl
    metal = _translate(tmp_path, INITIALIZED_UNIFORM, "metal")
    assert "constant float table[4] = {1.0, 2.0, 3.0, 4.0};" in metal
    assert "[[buffer(0)]]" not in metal


@pytest.mark.parametrize(
    ("source", "match"),
    [
        (
            "@group(0) @binding(0) var<storage, read_write> hits: atomic<u32>;\n"
            "@compute @workgroup_size(1) fn k() { atomicAdd(hits, 1u); }\n",
            "holds atomic values",
        ),
        (
            "@group(0) @binding(0) var<storage, read_write> total: f32;\n"
            "@compute @workgroup_size(1) fn k() { var total: f32 = 1.0; total = total + 2.0; }\n",
            "declares another 'total'",
        ),
        (
            "struct Run { uint n; float v[]; };\n"
            "@group(0) @binding(0) var<storage, read_write> other: array<f32>;\n"
            "@compute @workgroup_size(1) fn k(@group(0) @binding(0) var<storage, read_write> run: Run) "
            "{ run.v[0] = f32(run.n); }\n",
            "already uses that name or binding",
        ),
    ],
    ids=["atomic", "shadowed", "taken-binding"],
)
def test_unplaceable_storage_variables_are_refused(tmp_path, source, match):
    with pytest.raises(StorageVariableUnsupportedError, match=match) as raised:
        _translate(tmp_path, source, "directx")
    assert raised.value.code == "project.translate.directx-storage-variable-unsupported"


def test_storage_structure_parameter_with_runtime_array_is_refused_for_cuda(tmp_path):
    source = (
        "struct Run { uint n; float v[]; };\n"
        "@compute @workgroup_size(1) fn k(@group(0) @binding(0) var<storage, read_write> run: Run) "
        "{ run.v[0] = f32(run.n); }\n"
    )
    with pytest.raises(
        StorageVariableUnsupportedError, match="kernel parameter cannot pass"
    ):
        _translate(tmp_path, source, "cuda")
    glsl = _translate(tmp_path, source, "opengl")
    assert re.search(r"layout\(std430, binding = 0\) buffer Run\s*\{", glsl)


# CUDA and HIP import: namespace-scope device memory is a storage buffer.

DEVICE_GLOBALS = """
__device__ float gbuf[16];
__device__ int counter;
__device__ __managed__ int managed;
__constant__ float coeffs[4];
__global__ void k(float* out) {
    int i = threadIdx.x;
    gbuf[i] = out[i] * coeffs[i % 4];
    atomicAdd(&counter, 1);
    __syncthreads();
    out[i] = gbuf[15 - i] + (float)counter + (float)managed;
}
"""


@pytest.mark.parametrize("suffix", ["cu", "hip"])
def test_device_memory_imports_as_storage_buffers(tmp_path, suffix):
    source = (
        DEVICE_GLOBALS
        if suffix == "cu"
        else "#include <hip/hip_runtime.h>\n" + DEVICE_GLOBALS
    )
    crossgl = _translate(tmp_path, source, "cgl", suffix)
    assert (
        "@group(0) @binding(0) var<storage, read_write> gbuf: array<f32, 16>;"
        in crossgl
    )
    assert "@group(0) @binding(1) var<storage, read_write> counter: i32;" in crossgl
    assert "@group(0) @binding(2) var<storage, read_write> managed: i32;" in crossgl
    assert "@group(0) @binding(3) var<uniform> coeffs: array<f32, 4>;" in crossgl
    assert "var gbuf:" not in crossgl and "var counter:" not in crossgl
    hlsl = _translate(tmp_path, source, "directx", suffix)
    assert "RWStructuredBuffer<int> counter : register(u1);" in hlsl
    assert "InterlockedAdd(counter[0], 1);" in hlsl
    cuda = _translate(tmp_path, source, "cuda", suffix)
    assert "__device__ float gbuf[16];" in cuda


@pytest.mark.parametrize("suffix", ["cu", "hip"])
def test_initialized_device_memory_is_refused(tmp_path, suffix):
    source = (
        "__device__ int g_uids = 0;\n__global__ void k(float* out) { out[0] = 1.0f; }\n"
    )
    if suffix == "hip":
        source = "#include <hip/hip_runtime.h>\n" + source
    with pytest.raises(Exception, match="'g_uids' has an initializer") as raised:
        _translate(tmp_path, source, "cgl", suffix)
    language = "cuda" if suffix == "cu" else "hip"
    assert raised.value.code == f"project.translate.{language}-import-unsupported"


# Native validators, when installed.


def _run(command, cwd):
    result = subprocess.run(
        command, capture_output=True, text=True, cwd=cwd, timeout=180
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "source",
    [MODULE, PARAMETERS, BLOCK, INITIALIZED_UNIFORM],
    ids=["module", "parameters", "block", "constant"],
)
def test_glslang_accepts_the_buffers(tmp_path, source):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")
    shader = tmp_path / "out.comp"
    shader.write_text(_translate(tmp_path, source, "opengl"))
    _run(
        [
            glslang,
            "--target-env",
            "opengl",
            "-S",
            "comp",
            str(shader),
            "-o",
            str(tmp_path / "out.spv"),
        ],
        tmp_path,
    )


@pytest.mark.parametrize(
    "source",
    [MODULE, PARAMETERS, BLOCK, INITIALIZED_UNIFORM],
    ids=["module", "parameters", "block", "constant"],
)
def test_spirv_val_accepts_the_buffers(tmp_path, source):
    assembler, validator = shutil.which("spirv-as"), shutil.which("spirv-val")
    if assembler is None or validator is None:
        pytest.skip("spirv-as and spirv-val are not installed")
    module = tmp_path / "out.spvasm"
    module.write_text(_translate(tmp_path, source, "vulkan"))
    binary = tmp_path / "out.spv"
    _run(
        [assembler, "--target-env", "vulkan1.1", str(module), "-o", str(binary)],
        tmp_path,
    )
    _run([validator, "--target-env", "vulkan1.1", str(binary)], tmp_path)


@pytest.mark.parametrize(
    "source",
    [MODULE, PARAMETERS, INITIALIZED_UNIFORM],
    ids=["module", "parameters", "constant"],
)
def test_metal_compiler_accepts_the_buffers(tmp_path, source):
    xcrun = shutil.which("xcrun")
    if (
        xcrun is None
        or subprocess.run(
            [xcrun, "-sdk", "macosx", "-f", "metal"], capture_output=True
        ).returncode
    ):
        pytest.skip("the Metal compiler is not installed")
    shader = tmp_path / "out.metal"
    shader.write_text(_translate(tmp_path, source, "metal"))
    _run(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(shader),
            "-o",
            str(tmp_path / "out.air"),
        ],
        tmp_path,
    )
