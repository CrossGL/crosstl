"""Probe templates for the compute languages: CUDA, HIP, OpenCL C, Mojo and Rust."""

from __future__ import annotations

from . import composites, operators
from .base import ABSENT, C_FAMILY, Template, scalar

_CSTDINT = "#include <cstdint>"
# What only the graphics APIs have: fragment discard and every stage but compute.
_GRAPHICS = (
    "discard",
    "vertex",
    "fragment",
    "geometry",
    "tessellation-control",
    "tessellation-evaluation",
    "task",
    "mesh",
    "ray-generation",
    "intersection",
    "any-hit",
    "closest-hit",
    "miss",
    "callable",
)

CUDA = {
    **C_FAMILY,
    "function-call": Template(
        C_FAMILY["function-call"].body,
        "__device__ float twice(float x) {\n    return x * 2.0f;\n}",
    ),
    "out-parameter": Template(
        C_FAMILY["out-parameter"].body,
        "__device__ void split(float x, float& a, float& b) {\n    a = x;\n    b = x * 2.0f;\n}",
    ),
    "inout-parameter": Template(
        C_FAMILY["inout-parameter"].body,
        "__device__ void bump(float& a) {\n    a = a + 1.0f;\n}",
    ),
    **operators.c_casts("unsigned int"),
    **composites.CUDA,
    # CUDA vector types are plain structs: no swizzles, no arithmetic operators, and no matrix types at all.
    "swizzle": ABSENT,
    "vector-scalar-product": ABSENT,
    "matrix-vector-product": ABSENT,
    "matrix-product": ABSENT,
    "vector-matrix-product": ABSENT,
    "global-constant": Template(
        "outp[i] = fin[i] * SCALE;", "constexpr float SCALE = 2.0f;"
    ),
    # A CUDA buffer is a pointer; its length is not part of the language.
    "runtime-array-length": ABSENT,
    "half": scalar("__half", "fin", "#include <cuda_fp16.h>"),
    "double": scalar("double", "fin"),
    "int8": scalar("int8_t", prelude=_CSTDINT),
    "uint8": scalar("uint8_t", prelude=_CSTDINT),
    "int16": scalar("int16_t", prelude=_CSTDINT),
    "uint16": scalar("uint16_t", prelude=_CSTDINT),
    "int64": scalar("int64_t", prelude=_CSTDINT),
    "uint64": scalar("uint64_t", prelude=_CSTDINT),
    "shared-memory": Template(
        "__shared__ float tile[1];\ntile[0] = fin[i];\n__syncthreads();\noutp[i] = tile[0];"
    ),
    "local-invocation-id": Template("outp[i] = float(threadIdx.x);"),
    "workgroup-id": Template("outp[i] = float(blockIdx.x);"),
    # CUDA has no built-in for the flattened thread index; kernels compute it the way GLSL defines it.
    "local-invocation-index": Template(
        "outp[i] = float(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z));"
    ),
    "num-workgroups": Template("outp[i] = float(gridDim.x);"),
    "uniform-buffer": Template(
        "outp[i] = fin[i] * scale;", "__constant__ float scale;"
    ),
    "read-only-buffer": Template(
        "outp[i] = rin[i];", params="const float* __restrict__ rin"
    ),
    "sampled-texture": Template(
        "outp[i] = tex2DLod<float4>(tex, fin[i], 0.5f, 0.0f).x;",
        params="cudaTextureObject_t tex",
    ),
    "storage-image": Template(
        "surf2Dwrite(make_float4(fin[i], fin[i], fin[i], fin[i]), img, 0, 0);\noutp[i] = 0.0f;",
        params="cudaSurfaceObject_t img",
    ),
    # CUDA has kernels only; graphics and ray-tracing stages belong to the graphics APIs and OptiX.
    **dict.fromkeys(_GRAPHICS, ABSENT),
}

# HIP keeps CUDA's kernel language and renames the runtime types.
HIP = {
    **CUDA,
    "half": scalar("__half", "fin", "#include <hip/hip_fp16.h>"),
    "sampled-texture": Template(
        CUDA["sampled-texture"].body, params="hipTextureObject_t tex"
    ),
    "storage-image": Template(
        CUDA["storage-image"].body, params="hipSurfaceObject_t img"
    ),
}


def _opencl_scalar(spelling: str, source: str = "iin", prelude: str = "") -> Template:
    return Template(
        f"{spelling} v = ({spelling}){source}[i];\noutp[i] = (float)(v * v);",
        prelude=prelude,
    )


_OPENCL_PAIR = "typedef struct {\n    float a;\n    float b;\n} Pair;"

# OpenCL C is C99 with vector types: C casts instead of constructors, pointers instead of references, typedef'd
# structs, no matrices, and buffers whose length the kernel is not told.
OPENCL = {
    **C_FAMILY,
    "short-circuit-logic": Template(
        "if (iin[i] > 0 && fin[i] / (float)iin[i] > 1.0f) {\n    outp[i] = 1.0f;\n}"
        " else if (iin[i] < 0 || fin[i] < 0.0f) {\n    outp[i] = 2.0f;\n}"
    ),
    "out-parameter": Template(
        "float a;\nfloat b;\nsplit(fin[i], &a, &b);\noutp[i] = a + b;",
        "void split(float x, float* a, float* b) {\n    *a = x;\n    *b = x * 2.0f;\n}",
    ),
    "inout-parameter": Template(
        "float a = fin[i];\nbump(&a);\noutp[i] = a;",
        "void bump(float* a) {\n    *a = *a + 1.0f;\n}",
    ),
    "array": Template(
        "float values[4];\nfor (int k = 0; k < 4; k++) {\n    values[k] = fin[i] * (float)k;\n}\n"
        "outp[i] = values[iin[i] & 3];"
    ),
    "struct": Template(C_FAMILY["struct"].body, _OPENCL_PAIR),
    "array-of-structs": Template(C_FAMILY["array-of-structs"].body, _OPENCL_PAIR),
    "swizzle": Template(
        "float4 v = (float4)(fin[i], 1.0f, 2.0f, 3.0f);\noutp[i] = v.wzyx.x + v.xy.y;"
    ),
    **operators.c_casts("uint"),
    **composites.OPENCL,
    "vector-scalar-product": Template(
        "float2 v = (float2)(fin[i], 1.0f) * 4.0f;\nfloat2 w = 0.5f * v;\noutp[i] = w.x + w.y;"
    ),
    "matrix-vector-product": ABSENT,
    "matrix-product": ABSENT,
    "vector-matrix-product": ABSENT,
    "global-constant": Template(
        "outp[i] = fin[i] * SCALE;", "__constant float SCALE = 2.0f;"
    ),
    "runtime-array-length": ABSENT,
    "half": _opencl_scalar(
        "half", "fin", "#pragma OPENCL EXTENSION cl_khr_fp16 : enable"
    ),
    "double": _opencl_scalar("double", "fin"),
    "int8": _opencl_scalar("char"),
    "uint8": _opencl_scalar("uchar"),
    "int16": _opencl_scalar("short"),
    "uint16": _opencl_scalar("ushort"),
    "int64": _opencl_scalar("long"),
    "uint64": _opencl_scalar("ulong"),
    "shared-memory": Template(
        "__local float tile[1];\ntile[0] = fin[i];\nbarrier(CLK_LOCAL_MEM_FENCE);\noutp[i] = tile[0];"
    ),
    "local-invocation-id": Template("outp[i] = (float)get_local_id(0);"),
    "workgroup-id": Template("outp[i] = (float)get_group_id(0);"),
    "local-invocation-index": Template("outp[i] = (float)get_local_linear_id();"),
    "num-workgroups": Template("outp[i] = (float)get_num_groups(0);"),
    # A value every invocation shares is a by-value kernel argument.
    "uniform-buffer": Template("outp[i] = fin[i] * scale;", params="const float scale"),
    "read-only-buffer": Template(
        "outp[i] = rin[i];", params="__global const float* restrict rin"
    ),
    "sampled-texture": Template(
        "outp[i] = read_imagef(tex, smp, (float2)(fin[i], 0.5f)).x;",
        "__constant sampler_t smp = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;",
        params="__read_only image2d_t tex",
    ),
    "storage-image": Template(
        "write_imagef(img, (int2)(0, 0), (float4)(fin[i]));\noutp[i] = 0.0f;",
        params="__write_only image2d_t img",
    ),
    # OpenCL C has kernels only.
    **dict.fromkeys(_GRAPHICS, ABSENT),
}


def _mojo_scalar(spelling: str, source: str = "iin") -> Template:
    return Template(
        f"var v: {spelling} = {spelling}({source}[i])\noutp[i] = Float32(v * v)"
    )


_MOJO_PAIR = "@fieldwise_init\nstruct Pair(Copyable, Movable):\n    var a: Float32\n    var b: Float32"
_MOJO_LOOP = (
    "for k in range(8):\n    if k == Int(iin[i]):\n        {}\n    acc += fin[i]"
)
_MOJO_GPU = "from max.gpu import {}"

# Mojo writes Python's statements with typed declarations. It has no switch or do-while statement, no matrix types,
# no output arguments (``out`` names a function's single result), and kernels address memory through pointers only.
MOJO = {
    "if-else": Template(
        "if fin[i] > 0.0:\n    outp[i] = 1.0\nelse:\n    outp[i] = 2.0"
    ),
    "for-loop": Template(
        "var acc: Float32 = 0.0\nfor _ in range(Int(iin[i])):\n    acc += fin[i]\noutp[i] = acc"
    ),
    "while-loop": Template(
        "var acc: Float32 = 0.0\nvar k: Int32 = 0\nwhile k < iin[i]:\n    acc += fin[i]\n    k += 1\n"
        "outp[i] = acc"
    ),
    "break": Template(
        "var acc: Float32 = 0.0\n" + _MOJO_LOOP.format("break") + "\noutp[i] = acc"
    ),
    "continue": Template(
        "var acc: Float32 = 0.0\n" + _MOJO_LOOP.format("continue") + "\noutp[i] = acc"
    ),
    "early-return": Template(
        "if fin[i] < 0.0:\n    outp[i] = 0.0\n    return\noutp[i] = fin[i]"
    ),
    "conditional-expression": Template("outp[i] = fin[i] if fin[i] > 0.0 else 0.0"),
    "short-circuit-logic": Template(
        "if iin[i] > 0 and fin[i] / Float32(iin[i]) > 1.0:\n    outp[i] = 1.0\n"
        "elif iin[i] < 0 or fin[i] < 0.0:\n    outp[i] = 2.0"
    ),
    "function-call": Template(
        "outp[i] = twice(fin[i])",
        "def twice(x: Float32) -> Float32:\n    return x * 2.0",
    ),
    "inout-parameter": Template(
        "var a: Float32 = fin[i]\nbump(a)\noutp[i] = a",
        "def bump(mut a: Float32):\n    a = a + 1.0",
    ),
    "array": Template(
        "var values = Array[Float32, 4](fill=0.0)\nfor k in range(4):\n    values[k] = fin[i] * Float32(k)\n"
        "outp[i] = values[Int(iin[i] & 3)]"
    ),
    "struct": Template(
        "var p = Pair(0.0, 0.0)\np.a = fin[i]\np.b = 1.0\noutp[i] = p.a + p.b",
        _MOJO_PAIR,
    ),
    "array-of-structs": Template(
        "var ps = Array[Pair, 2](fill=Pair(0.0, 0.0))\nps[0].a = fin[i]\nps[1].b = 2.0\n"
        "outp[i] = ps[0].a + ps[1].b",
        _MOJO_PAIR,
    ),
    # SIMD values have no swizzle syntax; shuffle and slice select their elements.
    "swizzle": Template(
        "var v = $vec4(fin[i], 1.0, 2.0, 3.0)\noutp[i] = v.shuffle[3, 2, 1, 0]()[0] + v.slice[2]()[1]"
    ),
    "global-constant": Template(
        "outp[i] = fin[i] * SCALE", "comptime SCALE: Float32 = 2.0"
    ),
    "compute": Template("outp[i] = fin[i]"),
    **operators.MOJO,
    **composites.MOJO,
    "half": _mojo_scalar("Float16", "fin"),
    "double": _mojo_scalar("Float64", "fin"),
    "int8": _mojo_scalar("Int8"),
    "uint8": _mojo_scalar("UInt8"),
    "int16": _mojo_scalar("Int16"),
    "uint16": _mojo_scalar("UInt16"),
    "int64": _mojo_scalar("Int64"),
    "uint64": _mojo_scalar("UInt64"),
    "shared-memory": Template(
        "var tile = stack_allocation[1, Float32, address_space=AddressSpace.SHARED]()\ntile[0] = fin[i]\n"
        "barrier()\noutp[i] = tile[0]",
        prelude="from std.memory import stack_allocation\n"
        + _MOJO_GPU.format("barrier")
        + "\nfrom max.gpu.memory import AddressSpace",
    ),
    "local-invocation-id": Template(
        "outp[i] = Float32(thread_idx.x)", prelude=_MOJO_GPU.format("thread_idx")
    ),
    "workgroup-id": Template(
        "outp[i] = Float32(block_idx.x)", prelude=_MOJO_GPU.format("block_idx")
    ),
    # MAX has no built-in for the flattened thread index; kernels compute it the way GLSL defines it.
    "local-invocation-index": Template(
        "outp[i] = Float32(thread_idx.x + block_dim.x * (thread_idx.y + block_dim.y * thread_idx.z))",
        prelude=_MOJO_GPU.format("block_dim, thread_idx"),
    ),
    "num-workgroups": Template(
        "outp[i] = Float32(grid_dim.x)", prelude=_MOJO_GPU.format("grid_dim")
    ),
    # A kernel's scalar arguments are its uniform values.
    "uniform-buffer": Template("outp[i] = fin[i] * scale", params="scale: Float32"),
    "read-only-buffer": Template(
        "outp[i] = rin[i]", params="rin: UnsafePointer[Float32, ImmutAnyOrigin]"
    ),
    **dict.fromkeys(
        (
            "switch",
            "do-while-loop",
            "out-parameter",
            "matrix-vector-product",
            "runtime-array-length",
            "sampled-texture",
            "storage-image",
            "discard",
            "vertex",
            "fragment",
            "geometry",
            "tessellation-control",
            "tessellation-evaluation",
            "task",
            "mesh",
            "ray-generation",
            "intersection",
            "any-hit",
            "closest-hit",
            "miss",
            "callable",
        ),
        ABSENT,
    ),
}


def _rust_scalar(spelling: str, source: str = "iin") -> Template:
    return Template(
        f"let v: {spelling} = {source}[i] as {spelling};\noutp[i] = (v * v) as f32;"
    )


_RUST_PAIR = (
    "#[derive(Clone, Copy, Default)]\nstruct Pair {\n    a: f32,\n    b: f32,\n}"
)
_RUST_LOOP = (
    "let mut acc = 0.0;\nfor k in 0..8 {{\n    if k == iin[i] {{\n        {}\n    }}\n    acc += fin[i];\n}}\n"
    "outp[i] = acc;"
)
_RUST_STAGE = (
    "#![no_std]\nuse spirv_std::glam::*;\nuse spirv_std::spirv;\n\n#[spirv({stage})]\npub fn main({params}) {{\n"
    "{body}}}\n"
)
_RUST_COLOR = "    *color = Vec4::new(uv.x, uv.y, 0.0, 1.0);\n"
# Ray-tracing stage programs, as GLSL's: each writes a buffer or the ray data it is handed, and the hit stages also
# read the triangle's barycentrics.
_RUST_WRITE = dict(
    params="#[spirv(storage_buffer, descriptor_set = 0, binding = 0)] outp: &mut [f32]",
    body="    outp[0] = 1.0;\n",
)
_RUST_PAYLOAD = "pub struct Payload {\n    pub color: Vec4,\n}\n\n"
_RUST_HIT = dict(
    params="#[spirv(incoming_ray_payload)] payload: &mut Payload, "
    "#[spirv(hit_attribute)] attributes: &mut Vec2",
    body="    payload.color = Vec4::new(attributes.x, attributes.y, 0.0, 1.0);\n",
)


def _rust_stage(stage: str, declarations: str = "", **parts: str) -> Template:
    program = _RUST_STAGE.format(stage=stage, **parts)
    if declarations:
        program = program.replace("#[spirv(", declarations + "#[spirv(", 1)
    return Template(program=program, entry="main")


_RUST_RAYTRACING_STAGES = {
    "ray-generation": _rust_stage("ray_generation", **_RUST_WRITE),
    "intersection": _rust_stage("intersection", **_RUST_WRITE),
    "any-hit": _rust_stage("any_hit", _RUST_PAYLOAD, **_RUST_HIT),
    "closest-hit": _rust_stage("closest_hit", _RUST_PAYLOAD, **_RUST_HIT),
    "miss": _rust_stage(
        "miss",
        _RUST_PAYLOAD,
        params="#[spirv(incoming_ray_payload)] payload: &mut Payload",
        body="    payload.color = Vec4::new(0.0, 0.0, 0.0, 1.0);\n",
    ),
    "callable": _rust_stage(
        "callable",
        "pub struct CallData {\n    pub color: Vec4,\n}\n\n",
        params="#[spirv(incoming_callable_data)] data: &mut CallData",
        body="    data.color = Vec4::new(1.0, 0.0, 0.0, 1.0);\n",
    ),
}

# rust-gpu kernels take their resources and built-in inputs as entry-point parameters. Rust has no do-while
# statement; a loop that tests at its end is written with ``loop`` and ``break``.
RUST = {
    "if-else": Template(
        "if fin[i] > 0.0 {\n    outp[i] = 1.0;\n} else {\n    outp[i] = 2.0;\n}"
    ),
    "switch": Template(
        "match iin[i] {\n    0 => outp[i] = 1.0,\n    1 => outp[i] = 3.0,\n    _ => outp[i] = 2.0,\n}"
    ),
    "for-loop": Template(
        "let mut acc = 0.0;\nfor _ in 0..iin[i] {\n    acc += fin[i];\n}\noutp[i] = acc;"
    ),
    "while-loop": Template(
        "let mut acc = 0.0;\nlet mut k = 0;\nwhile k < iin[i] {\n    acc += fin[i];\n    k += 1;\n}\n"
        "outp[i] = acc;"
    ),
    "do-while-loop": ABSENT,
    "break": Template(_RUST_LOOP.format("break;")),
    "continue": Template(_RUST_LOOP.format("continue;")),
    "early-return": Template(
        "if fin[i] < 0.0 {\n    outp[i] = 0.0;\n    return;\n}\noutp[i] = fin[i];"
    ),
    "conditional-expression": Template(
        "outp[i] = if fin[i] > 0.0 { fin[i] } else { 0.0 };"
    ),
    "short-circuit-logic": Template(
        "if iin[i] > 0 && fin[i] / iin[i] as f32 > 1.0 {\n    outp[i] = 1.0;\n}"
        " else if iin[i] < 0 || fin[i] < 0.0 {\n    outp[i] = 2.0;\n}"
    ),
    "function-call": Template(
        "outp[i] = twice(fin[i]);", "fn twice(x: f32) -> f32 {\n    x * 2.0\n}"
    ),
    "out-parameter": Template(
        "let mut a = 0.0;\nlet mut b = 0.0;\nsplit(fin[i], &mut a, &mut b);\noutp[i] = a + b;",
        "fn split(x: f32, a: &mut f32, b: &mut f32) {\n    *a = x;\n    *b = x * 2.0;\n}",
    ),
    "inout-parameter": Template(
        "let mut a = fin[i];\nbump(&mut a);\noutp[i] = a;",
        "fn bump(a: &mut f32) {\n    *a += 1.0;\n}",
    ),
    "array": Template(
        "let mut values = [0.0; 4];\nfor k in 0..4 {\n    values[k] = fin[i] * k as f32;\n}\n"
        "outp[i] = values[(iin[i] & 3) as usize];"
    ),
    "struct": Template(
        "let mut p = Pair::default();\np.a = fin[i];\np.b = 1.0;\noutp[i] = p.a + p.b;",
        _RUST_PAIR,
    ),
    "array-of-structs": Template(
        "let mut ps = [Pair::default(); 2];\nps[0].a = fin[i];\nps[1].b = 2.0;\n"
        "outp[i] = ps[0].a + ps[1].b;",
        _RUST_PAIR,
    ),
    "swizzle": Template(
        "let v = $vec4::new(fin[i], 1.0, 2.0, 3.0);\noutp[i] = v.wzyx().x + v.xy().y;"
    ),
    "matrix-vector-product": Template(
        "let m = $mat2::from_cols($vec2::new(fin[i], 1.0), $vec2::new(2.0, 3.0));\n"
        "let v = m * $vec2::new(1.0, fin[i]);\noutp[i] = v.y;"
    ),
    **operators.RUST,
    **composites.RUST,
    "global-constant": Template("outp[i] = fin[i] * SCALE;", "const SCALE: f32 = 2.0;"),
    "runtime-array-length": Template("outp[i] = outp.len() as f32;"),
    "compute": Template("outp[i] = fin[i];"),
    "half": Template(
        "let v: f16 = fin[i] as f16;\noutp[i] = (v * v) as f32;",
        prelude="#![feature(f16)]",
    ),
    "double": _rust_scalar("f64", "fin"),
    "int8": _rust_scalar("i8"),
    "uint8": _rust_scalar("u8"),
    "int16": _rust_scalar("i16"),
    "uint16": _rust_scalar("u16"),
    "int64": _rust_scalar("i64"),
    "uint64": _rust_scalar("u64"),
    "shared-memory": Template(
        "tile[0] = fin[i];\nworkgroup_memory_barrier_with_group_sync();\noutp[i] = tile[0];",
        params="#[spirv(workgroup)] tile: &mut [f32; 1]",
    ),
    "local-invocation-id": Template(
        "outp[i] = local.x as f32;",
        params="#[spirv(local_invocation_id)] local: UVec3",
    ),
    "workgroup-id": Template(
        "outp[i] = group.x as f32;", params="#[spirv(workgroup_id)] group: UVec3"
    ),
    "local-invocation-index": Template(
        "outp[i] = index as f32;",
        params="#[spirv(local_invocation_index)] index: u32",
    ),
    "num-workgroups": Template(
        "outp[i] = groups.x as f32;", params="#[spirv(num_workgroups)] groups: UVec3"
    ),
    "uniform-buffer": Template(
        "outp[i] = fin[i] * *scale;",
        params="#[spirv(uniform, descriptor_set = 0, binding = 3)] scale: &f32",
    ),
    "read-only-buffer": Template(
        "outp[i] = rin[i];",
        params="#[spirv(storage_buffer, descriptor_set = 0, binding = 3)] rin: &[f32]",
    ),
    "sampled-texture": Template(
        "let color: Vec4 = tex.sample_by_lod(*sampler, Vec2::new(fin[i], 0.5), 0.0);\noutp[i] = color.x;",
        prelude="use spirv_std::{Image, Sampler};",
        params="#[spirv(descriptor_set = 0, binding = 3)] tex: &Image!(2D, type=f32, sampled)\n"
        "#[spirv(descriptor_set = 0, binding = 4)] sampler: &Sampler",
    ),
    "storage-image": Template(
        "unsafe { img.write(UVec2::new(0, 0), Vec4::splat(fin[i])) };\noutp[i] = 0.0;",
        prelude="use spirv_std::Image;",
        params="#[spirv(descriptor_set = 0, binding = 3)] img: &Image!(2D, format = rgba32f, sampled = false)",
    ),
    "vertex": Template(
        program=_RUST_STAGE.format(
            stage="vertex",
            params="position: Vec4, #[spirv(position)] out_position: &mut Vec4",
            body="    *out_position = position;\n",
        ),
        entry="main",
    ),
    "fragment": Template(
        program=_RUST_STAGE.format(
            stage="fragment", params="uv: Vec2, color: &mut Vec4", body=_RUST_COLOR
        ),
        entry="main",
    ),
    "discard": Template(
        program=_RUST_STAGE.format(
            stage="fragment",
            params="uv: Vec2, color: &mut Vec4",
            body="    if uv.x < 0.5 {\n        spirv_std::arch::kill();\n    }\n"
            + _RUST_COLOR,
        ),
        entry="main",
    ),
    "geometry": _rust_stage(
        "geometry(input_points = 1, output_points = 1, output_vertices = 1)",
        params="position: [Vec4; 1], #[spirv(position)] out_position: &mut Vec4",
        body="    *out_position = position[0] * 2.0;\n    unsafe {\n        spirv_std::arch::emit_vertex();\n"
        "        spirv_std::arch::end_primitive();\n    }\n",
    ),
    # A triangle patch of three control points, tessellated once.
    "tessellation-control": _rust_stage(
        "tessellation_control(output_vertices = 3)",
        params="#[spirv(invocation_id)] id: u32, position: [Vec4; 3], out_position: &mut [Vec4; 3], "
        "#[spirv(tess_level_outer)] outer: &mut [f32; 4], #[spirv(tess_level_inner)] inner: &mut [f32; 2]",
        body="    out_position[id as usize] = position[id as usize] * 2.0;\n"
        "    outer[0] = 1.0;\n    outer[1] = 1.0;\n    outer[2] = 1.0;\n    inner[0] = 1.0;\n",
    ),
    "tessellation-evaluation": _rust_stage(
        "tessellation_evaluation(triangles, spacing_equal, vertex_order_ccw)",
        params="#[spirv(tess_coord)] coord: Vec3, position: [Vec4; 3], #[spirv(position)] out_position: &mut Vec4",
        body="    *out_position = position[0] * coord.x + position[1] * coord.y + position[2] * coord.z;\n",
    ),
    # A mesh shader that writes one triangle, and a task shader that launches one mesh workgroup with a payload.
    "mesh": _rust_stage(
        "mesh_ext(threads(1), output_vertices = 3, output_primitives_ext = 1, output_triangles_ext)",
        params="#[spirv(position)] positions: &mut [Vec4; 3], "
        "#[spirv(primitive_triangle_indices_ext)] indices: &mut [UVec3; 1]",
        body="    unsafe {\n        spirv_std::arch::set_mesh_outputs_ext(3, 1);\n    }\n"
        "    positions[0] = Vec4::new(0.0, 0.0, 0.0, 1.0);\n"
        "    positions[1] = Vec4::new(1.0, 0.0, 0.0, 1.0);\n"
        "    positions[2] = Vec4::new(0.0, 1.0, 0.0, 1.0);\n"
        "    indices[0] = UVec3::new(0, 1, 2);\n",
    ),
    "task": _rust_stage(
        "task_ext(threads(1))",
        "pub struct Payload {\n    pub value: u32,\n}\n\n",
        params="#[spirv(task_payload_workgroup_ext)] payload: &mut Payload",
        body="    payload.value = 1;\n    unsafe {\n"
        "        spirv_std::arch::emit_mesh_tasks_ext_payload(1, 1, 1, payload);\n    }\n",
    ),
    **_RUST_RAYTRACING_STAGES,
}
