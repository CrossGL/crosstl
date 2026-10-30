"""Restrict, coherent, volatile and write-only buffer probes: the memory qualifiers a compute kernel gives its buffers.

Each probe doubles its input into its output buffer with one qualifier on its buffers: ``restrict`` on both, or
``coherent``, ``volatile`` or ``writeonly`` on the output. GLSL and CrossGL spell all four on storage buffers. HLSL
and Slang have ``globallycoherent``, Metal ``__restrict``, ``coherent(device)`` and ``volatile`` on device pointers,
CUDA and HIP ``__restrict__`` and ``volatile``, and OpenCL C ``restrict`` and ``volatile``.

HLSL and Slang have no restrict, volatile or write-only buffers, Metal no write-only device memory, and the kernel
languages no coherent or write-only buffers. WGSL's storage buffers are ``read`` or ``read_write`` with no
qualifiers, rust-gpu has none either, and Mojo's pointers and GLSL ES 3.00 have no such buffers.
"""

from __future__ import annotations

from .base import ABSENT, Template

NAMES = ("restrict-buffer", "coherent-buffer", "volatile-buffer", "write-only-buffer")

# Each probe's qualifiers on its input and its output buffer, in GLSL and CrossGL.
_QUALIFIERS = {
    "restrict-buffer": ("restrict ", "restrict "),
    "coherent-buffer": ("", "coherent "),
    "volatile-buffer": ("", "volatile "),
    "write-only-buffer": ("", "writeonly "),
}

GLSL = {
    name: Template(
        program="#version 460\nlayout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;\n"
        f"layout(std430, binding = 0) {source}buffer B0 {{ float fin[]; }};\n"
        f"layout(std430, binding = 1) {target}buffer B1 {{ float outp[]; }};\n"
        "void main() {\n    uint i = gl_GlobalInvocationID.x;\n    outp[i] = fin[i] * 2.0;\n}\n"
    )
    for name, (source, target) in _QUALIFIERS.items()
}
CROSSGL = {
    name: Template(
        program="shader Probe {\n    compute {\n"
        "        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;\n"
        f"        {source}buffer float* fin;\n        {target}buffer float* outp;\n"
        "        void main() {\n            uint i = gl_GlobalInvocationID.x;\n"
        "            outp[i] = fin[i] * 2.0;\n        }\n    }\n}\n"
    )
    for name, (source, target) in _QUALIFIERS.items()
}
WEBGL = dict.fromkeys(NAMES, ABSENT)
WGSL = dict.fromkeys(NAMES, ABSENT)
RUST = dict.fromkeys(NAMES, ABSENT)
MOJO = dict.fromkeys(NAMES, ABSENT)

_HLSL_COHERENT = (
    "RWStructuredBuffer<float> fin : register(u0);\n"
    "globallycoherent RWStructuredBuffer<float> outp : register(u1);\n"
    "[numthreads(1, 1, 1)]\nvoid main(uint3 id : SV_DispatchThreadID) {\n"
    "    uint i = id.x;\n    outp[i] = fin[i] * 2.0;\n}\n"
)
HLSL = {
    **dict.fromkeys(NAMES, ABSENT),
    "coherent-buffer": Template(program=_HLSL_COHERENT, entry="main"),
}
SLANG = {
    **dict.fromkeys(NAMES, ABSENT),
    "coherent-buffer": Template(
        program=_HLSL_COHERENT.replace(" : register(u0)", "")
        .replace(" : register(u1)", "")
        .replace("[numthreads", '[shader("compute")]\n[numthreads'),
        entry="main",
    ),
}

# Each Metal probe's input and output parameters.
_METAL_PARAMETERS = {
    "restrict-buffer": (
        "device float* __restrict fin",
        "device float* __restrict outp",
    ),
    "coherent-buffer": ("device float* fin", "coherent(device) device float* outp"),
    "volatile-buffer": ("device float* fin", "volatile device float* outp"),
}
METAL = {
    **{
        name: Template(
            program="#include <metal_stdlib>\nusing namespace metal;\n\n"
            f"kernel void probe({source} [[buffer(0)]], {target} [[buffer(1)]], "
            "uint3 id [[thread_position_in_grid]]) {\n"
            "    uint i = id.x;\n    outp[i] = fin[i] * 2.0;\n}\n"
        )
        for name, (source, target) in _METAL_PARAMETERS.items()
    },
    "write-only-buffer": ABSENT,
}

_CUDA_BODY = "    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n    outp[i] = fin[i] * 2.0;\n}\n"
# Each CUDA and HIP probe's input and output parameters.
_CUDA_PARAMETERS = {
    "restrict-buffer": "float* __restrict__ fin, float* __restrict__ outp",
    "volatile-buffer": "float* fin, volatile float* outp",
}
_ABSENT_IN_KERNELS = dict.fromkeys(("coherent-buffer", "write-only-buffer"), ABSENT)
CUDA = {
    **{
        name: Template(
            program=f'extern "C" __global__ void probe({parameters}) {{\n{_CUDA_BODY}'
        )
        for name, parameters in _CUDA_PARAMETERS.items()
    },
    **_ABSENT_IN_KERNELS,
}
HIP = {
    **{
        name: Template(program="#include <hip/hip_runtime.h>\n" + template.program)
        for name, template in CUDA.items()
        if template is not ABSENT
    },
    **_ABSENT_IN_KERNELS,
}

_OPENCL_PARAMETERS = {
    "restrict-buffer": "__global float* restrict fin, __global float* restrict outp",
    "volatile-buffer": "__global float* fin, __global volatile float* outp",
}
OPENCL = {
    **{
        name: Template(
            program="__kernel __attribute__((reqd_work_group_size(1, 1, 1))) "
            f"void probe({parameters}) {{\n"
            "    uint i = get_global_id(0);\n    outp[i] = fin[i] * 2.0;\n}\n"
        )
        for name, parameters in _OPENCL_PARAMETERS.items()
    },
    **_ABSENT_IN_KERNELS,
}
