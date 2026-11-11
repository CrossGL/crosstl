"""Probe templates for Mojo."""

from __future__ import annotations

from . import (
    abstractions,
    aggregates,
    assignments,
    bindings,
    blocks,
    booleans,
    branches,
    calls,
    columns,
    composites,
    constructors,
    declarations,
    draws,
    edges,
    elements,
    evaluation,
    extensions,
    formats,
    forms,
    fragments,
    functions,
    helpers,
    identifiers,
    idioms,
    images,
    initializers,
    integers,
    interfaces,
    interlocks,
    layouts,
    lexical,
    lvalues,
    meshes,
    operators,
    orientation,
    packing,
    pipeline,
    portability,
    primitives,
    qualifiers,
    rasterization,
    rays,
    records,
    relations,
    rows,
    samplers,
    scopes,
    shapes,
    specializations,
    statements,
    storage,
    subgroups,
    tasks,
    tessellation,
    textures,
    values,
    vectors,
    vertices,
    workgroups,
)
from .base import ABSENT, Template


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
    **functions.MOJO,
    **constructors.MOJO,
    **scopes.MOJO,
    **declarations.MOJO,
    **calls.MOJO,
    **branches.MOJO,
    **shapes.MOJO,
    **evaluation.MOJO,
    **assignments.MOJO,
    **statements.MOJO,
    **lvalues.MOJO,
    **aggregates.MOJO,
    **values.MOJO,
    **edges.MOJO,
    **lexical.MOJO,
    **forms.MOJO,
    **idioms.MOJO,
    **abstractions.MOJO,
    **portability.MOJO,
    **initializers.MOJO,
    **helpers.MOJO,
    **layouts.MOJO,
    **elements.MOJO,
    **blocks.MOJO,
    **workgroups.MOJO,
    **records.MOJO,
    **identifiers.MOJO,
    **integers.MOJO,
    **columns.MOJO,
    **booleans.MOJO,
    **packing.MOJO,
    **orientation.MOJO,
    **rows.MOJO,
    **vectors.MOJO,
    **relations.MOJO,
    **fragments.KERNELS,
    **interfaces.KERNELS,
    **vertices.KERNELS,
    **draws.KERNELS,
    **rasterization.KERNELS,
    **pipeline.KERNELS,
    **extensions.KERNELS,
    **primitives.KERNELS,
    **tessellation.KERNELS,
    **meshes.KERNELS,
    **tasks.KERNELS,
    **rays.KERNELS,
    **bindings.KERNELS,
    **interlocks.KERNELS,
    **subgroups.MOJO,
    **specializations.KERNELS,
    **qualifiers.MOJO,
    **textures.MOJO,
    **images.MOJO,
    **formats.MOJO,
    **storage.MOJO,
    **samplers.KERNELS,
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
