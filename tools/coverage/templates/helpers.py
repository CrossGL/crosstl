"""Buffer-in-helper, buffer-element-argument, invocation-id-in-helper, shared-memory-in-helper, uniform-in-helper and
barrier-in-loop probes: their form in the languages that declare resources at program scope (:data:`FAMILY`, as HLSL
writes them) and the languages that write them differently.

GLSL, HLSL, Slang, WGSL and CrossGL declare buffers, uniform blocks and workgroup arrays at program scope, so a helper
function reads and writes them by name. Metal, CUDA, HIP, OpenCL C, Mojo and Rust bind buffers to entry-point
parameters, so a helper takes each buffer as a pointer or a slice. Metal and OpenCL C declare a workgroup array in the
kernel, and Mojo allocates one there, so a helper takes a pointer to it; CUDA and HIP declare shared and constant
memory at file scope, and Rust binds both to entry-point parameters. Metal, OpenCL C, Mojo and Rust receive constants
as entry-point parameters too. The invocation id is a built-in variable in GLSL and CrossGL and a built-in call in
CUDA, HIP, OpenCL C and Mojo; HLSL, Slang, Metal, WGSL and Rust read it from an entry-point parameter only, so their
helpers take it as an argument.

A buffer element passed to an inout parameter is copied in and out in GLSL, HLSL and Slang and bound by reference in
Metal, CUDA, HIP and Rust; OpenCL C and Mojo pass a pointer to it. WGSL passes only function and private pointers to a
function (naga does not implement the ``unrestricted_pointer_parameters`` extension), so its probe copies the element
through a local.

WebGL 2 has no storage buffers, workgroup memory or compute stage, so of these it has only uniform-in-helper.

Every value the probes compute is a small multiple of a quarter, so each sum and product is exact.
"""

from __future__ import annotations

from .base import ABSENT, Template
from .statements import c_text

NAMES = (
    "buffer-in-helper",
    "buffer-element-argument",
    "invocation-id-in-helper",
    "shared-memory-in-helper",
    "uniform-in-helper",
    "barrier-in-loop",
)

_FETCH = "float fetch(uint j) {\n    return float(iin[j]) * 0.25;\n}"
_PUT = "void put(uint j, float x) {\n    outp[j] = x;\n}"
_DEPOSIT = "void deposit(uint j, float x) {\n    put(j, x * 2.0 + float(iin[j]));\n}"
_DEPOSIT_CALL = "deposit(i, fetch(i) + 1.0);"

_GROW = "void grow(inout float x, int k) {\n    x = x * 2.0 + float(k);\n}"
_GROW_CALLS = (
    "outp[i] = float(iin[i]) * 0.5;\ngrow(outp[i], iin[i]);\ngrow(outp[i], 3);"
)

_INVOCATION = "uint invocation() {\n    return gl_GlobalInvocationID.x;\n}"
_SHIFTED = "float shifted(float x) {\n    return x * 2.0 + float(invocation());\n}"
_SHIFTED_CALL = "outp[invocation()] = shifted(float(iin[i]));"
# The languages whose helpers take the entry point's invocation id ``id``.
_INVOCATION_ARGUMENT = (
    "uint invocation(uint3 tid) {\n    return tid.x;\n}\n\n"
    "float shifted(uint3 tid, float x) {\n    return x * 2.0 + float(invocation(tid));\n}"
)
_SHIFTED_ARGUMENT_CALL = "outp[invocation(id)] = shifted(id, float(iin[i]));"

_STASH = "void stash(float x) {\n    tile[0] = x;\n    tile[1] = x * 0.5;\n}"
_EXCHANGE = "float exchange(float x) {\n    stash(x);\n    $barrier;\n    return tile[0] * 2.0 + tile[1];\n}"
_EXCHANGE_CALL = "outp[i] = exchange(float(iin[i])) + 1.0;"

_SCALED = "float scaled(float x) {\n    return x * scale + 1.0;\n}"
_SCALED_CALL = "outp[i] = scaled(float(iin[i])) * 2.0;"

_LOOP = (
    "float acc = 0.0;\nfor (int k = 0; k < 3; k++) {\n    tile[0] = float(iin[i] + k);\n    $barrier;\n"
    "    acc += tile[0] * float(k + 1);\n    $barrier;\n}\noutp[i] = acc;"
)


def _barrier(text: str, barrier: str) -> str:
    return text.replace("$barrier", barrier)


def _family(shared: str, barrier: str, block: str) -> dict[str, Template]:
    """The probes in a language that declares resources at program scope: ``shared`` declares a workgroup array
    (``{}`` for its length), ``barrier`` synchronizes the workgroup and ``block`` declares the uniform ``scale``.
    """
    exchange = _barrier(_EXCHANGE, barrier)
    return {
        "buffer-in-helper": Template(
            _DEPOSIT_CALL, f"{_FETCH}\n\n{_PUT}\n\n{_DEPOSIT}"
        ),
        "buffer-element-argument": Template(_GROW_CALLS, _GROW),
        "shared-memory-in-helper": Template(
            _EXCHANGE_CALL, f"{shared.format(2)}\n\n{_STASH}\n\n{exchange}"
        ),
        "uniform-in-helper": Template(_SCALED_CALL, f"{block}\n\n{_SCALED}"),
        "barrier-in-loop": Template(_barrier(_LOOP, barrier), shared.format(1)),
    }


FAMILY = {
    **_family(
        "groupshared float tile[{}];",
        "GroupMemoryBarrierWithGroupSync()",
        "cbuffer Params : register(b3) {\n    float scale;\n};",
    ),
    "invocation-id-in-helper": Template(_SHIFTED_ARGUMENT_CALL, _INVOCATION_ARGUMENT),
}

_GLSL_BLOCK = "uniform Params {\n    float scale;\n};"

GLSL = {
    **_family(
        "shared float tile[{}];",
        "barrier()",
        f"layout(std140, binding = 3) {_GLSL_BLOCK}",
    ),
    "invocation-id-in-helper": Template(_SHIFTED_CALL, f"{_INVOCATION}\n\n{_SHIFTED}"),
}

# CrossGL declares the uniform block at shader scope and everything else in the compute block.
CROSSGL = {
    **{
        name: template for name, template in GLSL.items() if name != "uniform-in-helper"
    },
    "uniform-in-helper": Template(
        _SCALED_CALL,
        _SCALED,
        prelude="cbuffer Params @binding(3) {\n    float scale;\n};",
    ),
}

WEBGL = {
    **dict.fromkeys(NAMES, ABSENT),
    "uniform-in-helper": Template(
        _SCALED_CALL, f"layout(std140) {_GLSL_BLOCK}\n\n{_SCALED}"
    ),
}

_METAL_PARAMS = "struct Params {\n    float scale;\n};"
_METAL_TILE = "threadgroup float* tile"

METAL = {
    "buffer-in-helper": Template(
        "deposit(outp, iin, i, fetch(iin, i) + 1.0);",
        _FETCH.replace("(uint j", "(device int* iin, uint j")
        + "\n\n"
        + _PUT.replace("(uint j", "(device float* outp, uint j")
        + "\n\n"
        + _DEPOSIT.replace(
            "(uint j", "(device float* outp, device int* iin, uint j"
        ).replace("put(j", "put(outp, j"),
    ),
    "buffer-element-argument": Template(
        _GROW_CALLS, _GROW.replace("inout float", "device float&")
    ),
    "shared-memory-in-helper": Template(
        "threadgroup float tile[2];\noutp[i] = exchange(tile, float(iin[i])) + 1.0;",
        _STASH.replace("(float x", f"({_METAL_TILE}, float x")
        + "\n\n"
        + _barrier(_EXCHANGE, "threadgroup_barrier(mem_flags::mem_threadgroup)")
        .replace("(float x", f"({_METAL_TILE}, float x")
        .replace("stash(x", "stash(tile, x"),
    ),
    "uniform-in-helper": Template(
        "outp[i] = scaled(params, float(iin[i])) * 2.0;",
        f"{_METAL_PARAMS}\n\n"
        + _SCALED.replace("(float x", "(constant Params& params, float x").replace(
            "* scale", "* params.scale"
        ),
        params="constant Params& params [[buffer(3)]]",
    ),
    "barrier-in-loop": Template(
        "threadgroup float tile[1];\n"
        + _barrier(_LOOP, "threadgroup_barrier(mem_flags::mem_threadgroup)")
    ),
}


def _c(
    unsigned: str, device: str, buffer: str, local: str, barrier: str
) -> dict[str, Template]:
    """The probes CUDA, HIP and OpenCL C share, in C with casts and ``unsigned`` for uint: ``device`` precedes each
    helper, ``buffer`` qualifies a buffer pointer, ``local`` qualifies a workgroup pointer and ``barrier``
    synchronizes the workgroup."""
    fetch = _FETCH.replace("(uint j", f"({buffer}int* iin, uint j")
    put = _PUT.replace("(uint j", f"({buffer}float* outp, uint j")
    deposit = _DEPOSIT.replace(
        "(uint j", f"({buffer}float* outp, {buffer}int* iin, uint j"
    )
    helpers = "\n\n".join(
        device + c_text(h, unsigned)
        for h in (fetch, put, deposit.replace("put(j", "put(outp, j"))
    )
    return {
        "buffer-in-helper": Template(
            c_text("deposit(outp, iin, i, fetch(iin, i) + 1.0);", unsigned), helpers
        ),
        "barrier-in-loop": Template(
            f"{local}float tile[1];\n" + c_text(_barrier(_LOOP, barrier), unsigned)
        ),
    }


_CUDA_TILE = "__shared__ float tile[2];"


def _cuda() -> dict[str, Template]:
    """CUDA's probes, which HIP shares: shared and constant memory at file scope, references for inout."""
    unsigned = "unsigned int"
    device = "__device__ "
    return {
        **_c(unsigned, device, "", "__shared__ ", "__syncthreads()"),
        "buffer-element-argument": Template(
            c_text(_GROW_CALLS, unsigned),
            device + c_text(_GROW.replace("inout float", "float&"), unsigned),
        ),
        "invocation-id-in-helper": Template(
            c_text(_SHIFTED_CALL, unsigned),
            f"{device}unsigned int invocation() {{\n    return blockIdx.x * blockDim.x + threadIdx.x;\n}}\n\n"
            + device
            + c_text(_SHIFTED, unsigned),
        ),
        "shared-memory-in-helper": Template(
            c_text(_EXCHANGE_CALL, unsigned),
            f"{_CUDA_TILE}\n\n{device}{c_text(_STASH, unsigned)}\n\n"
            + device
            + c_text(_barrier(_EXCHANGE, "__syncthreads()"), unsigned),
        ),
        "uniform-in-helper": Template(
            c_text(_SCALED_CALL, unsigned),
            f"__constant__ float scale;\n\n{device}{c_text(_SCALED, unsigned)}",
        ),
    }


CUDA = _cuda()

_OPENCL_TILE = "__local float* tile"

OPENCL = {
    **_c("uint", "", "__global ", "__local ", "barrier(CLK_LOCAL_MEM_FENCE)"),
    "buffer-element-argument": Template(
        c_text(_GROW_CALLS.replace("grow(outp", "grow(&outp"), "uint"),
        "void grow(__global float* x, int k) {\n    *x = *x * 2.0f + (float)(k);\n}",
    ),
    "invocation-id-in-helper": Template(
        c_text(_SHIFTED_CALL, "uint"),
        "uint invocation() {\n    return (uint)get_global_id(0);\n}\n\n"
        + c_text(_SHIFTED, "uint"),
    ),
    "shared-memory-in-helper": Template(
        "__local float tile[2];\noutp[i] = exchange(tile, (float)(iin[i])) + 1.0f;",
        c_text(_STASH.replace("(float x", f"({_OPENCL_TILE}, float x"), "uint")
        + "\n\n"
        + c_text(
            _barrier(_EXCHANGE, "barrier(CLK_LOCAL_MEM_FENCE)")
            .replace("(float x", f"({_OPENCL_TILE}, float x")
            .replace("stash(x", "stash(tile, x"),
            "uint",
        ),
    ),
    "uniform-in-helper": Template(
        "outp[i] = scaled(scale, (float)(iin[i])) * 2.0f;",
        c_text(_SCALED.replace("(float x", "(float scale, float x"), "uint"),
        params="const float scale",
    ),
}

_WGSL_LOOP = (
    "var acc = 0.0;\nfor (var k = 0; k < 3; k++) {\n    tile[0] = f32(iin[i] + k);\n    workgroupBarrier();\n"
    "    acc += tile[0] * f32(k + 1);\n    workgroupBarrier();\n}\noutp[i] = acc;"
)

WGSL = {
    "buffer-in-helper": Template(
        _DEPOSIT_CALL,
        "fn fetch(j: u32) -> f32 {\n    return f32(iin[j]) * 0.25;\n}\n\n"
        "fn put(j: u32, x: f32) {\n    outp[j] = x;\n}\n\n"
        "fn deposit(j: u32, x: f32) {\n    put(j, x * 2.0 + f32(iin[j]));\n}",
    ),
    "buffer-element-argument": Template(
        "outp[i] = f32(iin[i]) * 0.5;\nvar x = outp[i];\ngrow(&x, iin[i]);\ngrow(&x, 3);\noutp[i] = x;",
        "fn grow(x: ptr<function, f32>, k: i32) {\n    *x = *x * 2.0 + f32(k);\n}",
    ),
    "invocation-id-in-helper": Template(
        "outp[invocation(id)] = shifted(id, f32(iin[i]));",
        "fn invocation(tid: vec3<u32>) -> u32 {\n    return tid.x;\n}\n\n"
        "fn shifted(tid: vec3<u32>, x: f32) -> f32 {\n    return x * 2.0 + f32(invocation(tid));\n}",
    ),
    "shared-memory-in-helper": Template(
        "outp[i] = exchange(f32(iin[i])) + 1.0;",
        "var<workgroup> tile: array<f32, 2>;\n\n"
        "fn stash(x: f32) {\n    tile[0] = x;\n    tile[1] = x * 0.5;\n}\n\n"
        "fn exchange(x: f32) -> f32 {\n    stash(x);\n    workgroupBarrier();\n    return tile[0] * 2.0 + tile[1];\n}",
    ),
    "uniform-in-helper": Template(
        "outp[i] = scaled(f32(iin[i])) * 2.0;",
        "struct Params {\n    scale: f32,\n}\n@group(0) @binding(3) var<uniform> params: Params;\n\n"
        "fn scaled(x: f32) -> f32 {\n    return x * params.scale + 1.0;\n}",
    ),
    "barrier-in-loop": Template(_WGSL_LOOP, "var<workgroup> tile: array<f32, 1>;"),
}

_RUST_LOOP = (
    "let mut acc = 0.0;\nfor k in 0..3 {\n    tile[0] = (iin[i] + k) as f32;\n"
    "    workgroup_memory_barrier_with_group_sync();\n    acc += tile[0] * (k + 1) as f32;\n"
    "    workgroup_memory_barrier_with_group_sync();\n}\noutp[i] = acc;"
)

RUST = {
    "buffer-in-helper": Template(
        "deposit(outp, iin, i, fetch(iin, i) + 1.0);",
        "fn fetch(iin: &[i32], j: usize) -> f32 {\n    iin[j] as f32 * 0.25\n}\n\n"
        "fn put(outp: &mut [f32], j: usize, x: f32) {\n    outp[j] = x;\n}\n\n"
        "fn deposit(outp: &mut [f32], iin: &[i32], j: usize, x: f32) {\n"
        "    put(outp, j, x * 2.0 + iin[j] as f32);\n}",
    ),
    "buffer-element-argument": Template(
        "outp[i] = iin[i] as f32 * 0.5;\ngrow(&mut outp[i], iin[i]);\ngrow(&mut outp[i], 3);",
        "fn grow(x: &mut f32, k: i32) {\n    *x = *x * 2.0 + k as f32;\n}",
    ),
    "invocation-id-in-helper": Template(
        "outp[invocation(id)] = shifted(id, iin[i] as f32);",
        "fn invocation(tid: UVec3) -> usize {\n    tid.x as usize\n}\n\n"
        "fn shifted(tid: UVec3, x: f32) -> f32 {\n    x * 2.0 + invocation(tid) as f32\n}",
    ),
    "shared-memory-in-helper": Template(
        "outp[i] = exchange(tile, iin[i] as f32) + 1.0;",
        "fn stash(tile: &mut [f32; 2], x: f32) {\n    tile[0] = x;\n    tile[1] = x * 0.5;\n}\n\n"
        "fn exchange(tile: &mut [f32; 2], x: f32) -> f32 {\n    stash(tile, x);\n"
        "    workgroup_memory_barrier_with_group_sync();\n    tile[0] * 2.0 + tile[1]\n}",
        params="#[spirv(workgroup)] tile: &mut [f32; 2]",
    ),
    "uniform-in-helper": Template(
        "outp[i] = scaled(scale, iin[i] as f32) * 2.0;",
        "fn scaled(scale: &f32, x: f32) -> f32 {\n    x * *scale + 1.0\n}",
        params="#[spirv(uniform, descriptor_set = 0, binding = 3)] scale: &f32",
    ),
    "barrier-in-loop": Template(
        _RUST_LOOP, params="#[spirv(workgroup)] tile: &mut [f32; 1]"
    ),
}

_MOJO_POINTER = "UnsafePointer[{}, MutAnyOrigin]"
_MOJO_SHARED = (
    "UnsafePointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED]"
)
_MOJO_IMPORTS = "from std.memory import stack_allocation\nfrom max.gpu import barrier\nfrom max.gpu.memory import AddressSpace"
_MOJO_LOOP = (
    "var tile = stack_allocation[1, Float32, address_space=AddressSpace.SHARED]()\nvar acc: Float32 = 0.0\n"
    "for k in range(3):\n    tile[0] = Float32(iin[i] + Int32(k))\n    barrier()\n"
    "    acc += tile[0] * Float32(k + 1)\n    barrier()\noutp[i] = acc"
)

MOJO = {
    "buffer-in-helper": Template(
        "deposit(outp, iin, i, fetch(iin, i) + 1.0)",
        f"def fetch(iin: {_MOJO_POINTER.format('Int32')}, j: Int) -> Float32:\n"
        "    return Float32(iin[j]) * 0.25\n\n\n"
        f"def put(outp: {_MOJO_POINTER.format('Float32')}, j: Int, x: Float32):\n    outp[j] = x\n\n\n"
        f"def deposit(outp: {_MOJO_POINTER.format('Float32')}, iin: {_MOJO_POINTER.format('Int32')}, j: Int, "
        "x: Float32):\n    put(outp, j, x * 2.0 + Float32(iin[j]))",
    ),
    "buffer-element-argument": Template(
        "outp[i] = Float32(iin[i]) * 0.5\ngrow(outp + i, iin[i])\ngrow(outp + i, 3)",
        f"def grow(x: {_MOJO_POINTER.format('Float32')}, k: Int32):\n    x[] = x[] * 2.0 + Float32(k)",
    ),
    "invocation-id-in-helper": Template(
        "outp[invocation()] = shifted(Float32(iin[i]))",
        "def invocation() -> Int:\n    return global_idx.x\n\n\n"
        "def shifted(x: Float32) -> Float32:\n    return x * 2.0 + Float32(invocation())",
    ),
    "shared-memory-in-helper": Template(
        "var tile = stack_allocation[2, Float32, address_space=AddressSpace.SHARED]()\n"
        "outp[i] = exchange(tile, Float32(iin[i])) + 1.0",
        f"def stash(tile: {_MOJO_SHARED}, x: Float32):\n    tile[0] = x\n    tile[1] = x * 0.5\n\n\n"
        f"def exchange(tile: {_MOJO_SHARED}, x: Float32) -> Float32:\n    stash(tile, x)\n    barrier()\n"
        "    return tile[0] * 2.0 + tile[1]",
        prelude=_MOJO_IMPORTS,
    ),
    "uniform-in-helper": Template(
        "outp[i] = scaled(scale, Float32(iin[i])) * 2.0",
        "def scaled(scale: Float32, x: Float32) -> Float32:\n    return x * scale + 1.0",
        params="scale: Float32",
    ),
    "barrier-in-loop": Template(_MOJO_LOOP, prelude=_MOJO_IMPORTS),
}
