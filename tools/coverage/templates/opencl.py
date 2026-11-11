"""Probe templates for OpenCL C."""

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
from .base import ABSENT, GRAPHICS_ONLY, Template
from .family import C_FAMILY


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
    **aggregates.FAMILY,
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
    **functions.OPENCL,
    **constructors.OPENCL,
    **scopes.OPENCL,
    **declarations.OPENCL,
    **calls.OPENCL,
    **branches.OPENCL,
    **shapes.OPENCL,
    **evaluation.OPENCL,
    **assignments.OPENCL,
    **statements.OPENCL,
    **lvalues.OPENCL,
    **aggregates.OPENCL,
    **values.OPENCL,
    **edges.OPENCL,
    **lexical.OPENCL,
    **forms.OPENCL,
    **idioms.OPENCL,
    **abstractions.OPENCL,
    **portability.OPENCL,
    **initializers.OPENCL,
    **helpers.OPENCL,
    **layouts.OPENCL,
    **elements.OPENCL,
    **blocks.OPENCL,
    **workgroups.OPENCL,
    **records.OPENCL,
    **identifiers.OPENCL,
    **integers.OPENCL,
    **columns.OPENCL,
    **booleans.OPENCL,
    **packing.OPENCL,
    **orientation.OPENCL,
    **rows.OPENCL,
    **vectors.OPENCL,
    **relations.OPENCL,
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
    **subgroups.OPENCL,
    **specializations.KERNELS,
    **qualifiers.OPENCL,
    **textures.OPENCL,
    **images.OPENCL,
    **formats.OPENCL,
    **storage.OPENCL,
    **samplers.OPENCL,
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
    **dict.fromkeys(GRAPHICS_ONLY, ABSENT),
}
