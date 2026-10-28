"""Shared-vector-array, shared-struct-array and shared-2d-array probes: workgroup memory that holds four-component
vectors, structs or a two-dimensional array, written, synchronised and read back with a dynamic index, in every
language that has workgroup memory.

Each probe fills two elements, copies or changes one of them, waits at a workgroup barrier and reads an element the
invocation's input selects, so a translation has to keep the element type, the copy and the barrier. The floats come
from the integer input and combine exactly. Metal, CUDA, HIP, OpenCL C and Mojo declare workgroup memory in the
kernel's body, GLSL, HLSL, WGSL and CrossGL at program scope, and Rust as an entry-point parameter. WebGL 2 has no
compute stage.
"""

from __future__ import annotations

import re

from .base import ABSENT, PAIR, Template
from .statements import c_text

NAMES = ("shared-vector-array", "shared-struct-array", "shared-2d-array")

_VECTORS = (
    "float f = float(iin[i]);\ntile[0] = $vec4(f, 1.0, 2.0, 3.0);\ntile[1] = tile[0];\ntile[1].x *= 0.5;\n"
    "tile[1].y += f;\n{sync};\n$vec4 t = tile[iin[i] & 1];\n"
    "outp[i] = t.x * 8.0 + t.y * 4.0 + t.z * 2.0 + t.w + tile[1].y;"
)
_STRUCTS = (
    "float f = float(iin[i]);\ntile[0].a = f;\ntile[0].b = f * 2.0 + 1.0;\ntile[1] = tile[0];\ntile[1].a += 3.0;\n"
    "{sync};\nPair p = tile[iin[i] & 1];\noutp[i] = p.a * 10.0 + p.b + tile[1].a;"
)
_GRID = (
    "int n = iin[i];\ntile[0][0] = float(n);\ntile[0][1] = float(n + 1);\ntile[0][2] = float(n * 2);\n"
    "tile[1][0] = float(n + 10);\ntile[1][1] = float(n - 3);\ntile[1][2] = float(n * 3);\n{sync};\n"
    "int k = n & 1;\noutp[i] = tile[k][2] * 2.0 + tile[1 - k][0] + tile[1][1];"
)
_BODIES = {
    "shared-vector-array": _VECTORS,
    "shared-struct-array": _STRUCTS,
    "shared-2d-array": _GRID,
}
# Each probe's workgroup array as a C declarator.
_TILES = {
    "shared-vector-array": "$vec4 tile[2]",
    "shared-struct-array": "Pair tile[2]",
    "shared-2d-array": "float tile[2][3]",
}


def _bodies(sync: str) -> dict[str, str]:
    return {name: body.replace("{sync}", sync) for name, body in _BODIES.items()}


def _scoped(qualifier: str, sync: str, pair: str = PAIR) -> dict[str, Template]:
    """The probes of a language that declares workgroup memory at program scope with ``qualifier``."""
    bodies = _bodies(sync)
    return {
        name: Template(
            bodies[name],
            f"{qualifier} {tile};",
            prelude=pair if "Pair" in tile else "",
        )
        for name, tile in _TILES.items()
    }


def _local(
    qualifier: str, sync: str, text=lambda body: body, pair: str = PAIR
) -> dict[str, Template]:
    """The probes of a language that declares workgroup memory in the kernel's body with ``qualifier``; ``text``
    rewrites the body into the language."""
    bodies = _bodies(sync)
    return {
        name: Template(
            text(f"{qualifier} {tile};\n{bodies[name]}"),
            prelude=pair if "Pair" in tile else "",
        )
        for name, tile in _TILES.items()
    }


GLSL = _scoped("shared", "barrier()")
CROSSGL = _scoped("shared", "barrier()")
FAMILY = _scoped("groupshared", "GroupMemoryBarrierWithGroupSync()")
METAL = _local("threadgroup", "threadgroup_barrier(mem_flags::mem_threadgroup)")
WEBGL = dict.fromkeys(NAMES, ABSENT)


def _cuda(body: str) -> str:
    return c_text(body.replace("$vec4(", "make_float4("), "unsigned int")


def _opencl(body: str) -> str:
    return c_text(body.replace("$vec4(", "(float4)("), "uint")


CUDA = _local("__shared__", "__syncthreads()", _cuda)
OPENCL = _local(
    "__local",
    "barrier(CLK_LOCAL_MEM_FENCE)",
    _opencl,
    "typedef struct {\n    float a;\n    float b;\n} Pair;",
)

_WGSL_PAIR = "struct Pair {\n    a: f32,\n    b: f32,\n}"
_WGSL_TILES = {
    "shared-vector-array": "array<vec4<f32>, 2>",
    "shared-struct-array": "array<Pair, 2>",
    "shared-2d-array": "array<array<f32, 3>, 2>",
}
# WGSL infers the type of each local and converts with f32().
_WGSL_BODIES = {
    name: (
        re.sub(r"(?:int|float|Pair|\$vec4) (\w) =", r"let \1 =", body).replace(
            "float(", "f32("
        )
    )
    for name, body in _bodies("workgroupBarrier()").items()
}

WGSL = {
    name: Template(
        _WGSL_BODIES[name],
        "\n".join(
            (_WGSL_PAIR,) * ("Pair" in tile) + (f"var<workgroup> tile: {tile};",)
        ),
    )
    for name, tile in _WGSL_TILES.items()
}

_RUST_PAIR = "#[derive(Clone, Copy, Default)]\n#[repr(C)]\npub struct Pair {\n    a: f32,\n    b: f32,\n}"
_RUST_SYNC = "workgroup_memory_barrier_with_group_sync();"
_RUST_TILES = {
    "shared-vector-array": "[Vec4; 2]",
    "shared-struct-array": "[Pair; 2]",
    "shared-2d-array": "[[f32; 3]; 2]",
}
_RUST_BODIES = {
    "shared-vector-array": (
        "let f = iin[i] as f32;\ntile[0] = Vec4::new(f, 1.0, 2.0, 3.0);\ntile[1] = tile[0];\ntile[1].x *= 0.5;\n"
        f"tile[1].y += f;\n{_RUST_SYNC}\nlet t = tile[(iin[i] & 1) as usize];\n"
        "outp[i] = t.x * 8.0 + t.y * 4.0 + t.z * 2.0 + t.w + tile[1].y;"
    ),
    "shared-struct-array": (
        "let f = iin[i] as f32;\ntile[0].a = f;\ntile[0].b = f * 2.0 + 1.0;\ntile[1] = tile[0];\n"
        f"tile[1].a += 3.0;\n{_RUST_SYNC}\nlet p = tile[(iin[i] & 1) as usize];\n"
        "outp[i] = p.a * 10.0 + p.b + tile[1].a;"
    ),
    "shared-2d-array": (
        "let n = iin[i];\ntile[0][0] = n as f32;\ntile[0][1] = (n + 1) as f32;\ntile[0][2] = (n * 2) as f32;\n"
        "tile[1][0] = (n + 10) as f32;\ntile[1][1] = (n - 3) as f32;\ntile[1][2] = (n * 3) as f32;\n"
        f"{_RUST_SYNC}\nlet k = (n & 1) as usize;\noutp[i] = tile[k][2] * 2.0 + tile[1 - k][0] + tile[1][1];"
    ),
}

RUST = {
    name: Template(
        _RUST_BODIES[name],
        _RUST_PAIR if "Pair" in tile else "",
        params=f"#[spirv(workgroup)] tile: &mut {tile}",
    )
    for name, tile in _RUST_TILES.items()
}

_MOJO_IMPORTS = "from std.memory import stack_allocation\nfrom max.gpu import barrier\nfrom max.gpu.memory import AddressSpace"
_MOJO_PAIR = "@fieldwise_init\nstruct Pair(TrivialRegisterPassable):\n    var a: Float32\n    var b: Float32"
_MOJO_TILE = (
    "var tile = stack_allocation[{}, {}, address_space=AddressSpace.SHARED]()\n"
)
_MOJO_VEC4 = "SIMD[DType.float32, 4]"

MOJO = {
    "shared-vector-array": Template(
        _MOJO_TILE.format(2, _MOJO_VEC4)
        + f"var f = Float32(iin[i])\ntile[0] = {_MOJO_VEC4}(f, 1.0, 2.0, 3.0)\ntile[1] = tile[0]\n"
        "tile[1][0] *= 0.5\ntile[1][1] += f\nbarrier()\nvar t = tile[Int(iin[i] & 1)]\n"
        "outp[i] = t[0] * 8.0 + t[1] * 4.0 + t[2] * 2.0 + t[3] + tile[1][1]",
        prelude=_MOJO_IMPORTS,
    ),
    "shared-struct-array": Template(
        _MOJO_TILE.format(2, "Pair")
        + "var f = Float32(iin[i])\ntile[0] = Pair(f, f * 2.0 + 1.0)\ntile[1] = tile[0]\ntile[1].a += 3.0\n"
        "barrier()\nvar p = tile[Int(iin[i] & 1)]\noutp[i] = p.a * 10.0 + p.b + tile[1].a",
        _MOJO_PAIR,
        prelude=_MOJO_IMPORTS,
    ),
    # Mojo's Array cannot live in workgroup memory, so a two-dimensional tile is a LayoutTensor there; reading an
    # element gives a one-lane SIMD value.
    "shared-2d-array": Template(
        "var tile = LayoutTensor[DType.float32, Layout.row_major(2, 3), MutAnyOrigin, "
        "address_space=AddressSpace.SHARED].stack_allocation()\n"
        "var n = iin[i]\ntile[0, 0] = Float32(n)\ntile[0, 1] = Float32(n + 1)\ntile[0, 2] = Float32(n * 2)\n"
        "tile[1, 0] = Float32(n + 10)\ntile[1, 1] = Float32(n - 3)\ntile[1, 2] = Float32(n * 3)\nbarrier()\n"
        "var k = Int(n & 1)\noutp[i] = tile[k, 2][0] * 2.0 + tile[1 - k, 0][0] + tile[1, 1][0]",
        prelude="from max.gpu import barrier\nfrom max.gpu.memory import AddressSpace\n"
        "from layout import Layout, LayoutTensor",
    ),
}
