"""Probe templates for WGSL."""

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
    components,
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
from .family import C_FAMILY

WGSL = {
    "if-else": C_FAMILY["if-else"],
    "switch": Template(
        "switch (iin[i]) {\n    case 0: {\n        outp[i] = 1.0;\n    }\n    case 1: {\n        outp[i] = 3.0;\n"
        "    }\n    default: {\n        outp[i] = 2.0;\n    }\n}"
    ),
    "for-loop": Template(
        "var acc = 0.0;\nfor (var k = 0; k < iin[i]; k++) {\n    acc += fin[i];\n}\noutp[i] = acc;"
    ),
    "while-loop": Template(
        "var acc = 0.0;\nvar k = 0;\nwhile (k < iin[i]) {\n    acc += fin[i];\n    k++;\n}\noutp[i] = acc;"
    ),
    # WGSL's post-tested loop is ``loop`` with a ``break if`` in its continuing block.
    "do-while-loop": Template(
        "var acc = 0.0;\nvar k = 0;\nloop {\n    acc += fin[i];\n    k++;\n    continuing {\n"
        "        break if k >= iin[i];\n    }\n}\noutp[i] = acc;"
    ),
    "break": Template(
        "var acc = 0.0;\nfor (var k = 0; k < 8; k++) {\n    if (k == iin[i]) {\n        break;\n    }\n"
        "    acc += fin[i];\n}\noutp[i] = acc;"
    ),
    "continue": Template(
        "var acc = 0.0;\nfor (var k = 0; k < 8; k++) {\n    if (k == iin[i]) {\n        continue;\n    }\n"
        "    acc += fin[i];\n}\noutp[i] = acc;"
    ),
    "early-return": C_FAMILY["early-return"],
    "conditional-expression": Template("outp[i] = select(0.0, fin[i], fin[i] > 0.0);"),
    "short-circuit-logic": Template(
        "if (iin[i] > 0 && fin[i] / f32(iin[i]) > 1.0) {\n    outp[i] = 1.0;\n}"
        " else if (iin[i] < 0 || fin[i] < 0.0) {\n    outp[i] = 2.0;\n}"
    ),
    "function-call": Template(
        "outp[i] = twice(fin[i]);", "fn twice(x: f32) -> f32 {\n    return x * 2.0;\n}"
    ),
    "out-parameter": Template(
        "var a: f32;\nvar b: f32;\nsplit(fin[i], &a, &b);\noutp[i] = a + b;",
        "fn split(x: f32, a: ptr<function, f32>, b: ptr<function, f32>) {\n    *a = x;\n    *b = x * 2.0;\n}",
    ),
    "inout-parameter": Template(
        "var a = fin[i];\nbump(&a);\noutp[i] = a;",
        "fn bump(a: ptr<function, f32>) {\n    *a = *a + 1.0;\n}",
    ),
    "array": Template(
        "var values: array<f32, 4>;\nfor (var k = 0; k < 4; k++) {\n    values[k] = fin[i] * f32(k);\n}\n"
        "outp[i] = values[iin[i] & 3];"
    ),
    "struct": Template(
        "var p: Pair;\np.a = fin[i];\np.b = 1.0;\noutp[i] = p.a + p.b;",
        "struct Pair {\n    a: f32,\n    b: f32,\n}",
    ),
    "array-of-structs": Template(
        "var ps: array<Pair, 2>;\nps[0].a = fin[i];\nps[1].b = 2.0;\noutp[i] = ps[0].a + ps[1].b;",
        "struct Pair {\n    a: f32,\n    b: f32,\n}",
    ),
    "swizzle": Template(
        "let v = vec4<f32>(fin[i], 1.0, 2.0, 3.0);\noutp[i] = v.wzyx.x + v.xy.y;"
    ),
    "matrix-vector-product": Template(
        "let m = mat2x2<f32>(fin[i], 1.0, 2.0, 3.0);\nlet v = m * vec2<f32>(1.0, fin[i]);\noutp[i] = v.y;"
    ),
    **operators.WGSL,
    **composites.WGSL,
    **functions.WGSL,
    **constructors.WGSL,
    **scopes.WGSL,
    **declarations.WGSL,
    **calls.WGSL,
    **branches.WGSL,
    **shapes.WGSL,
    **evaluation.WGSL,
    **assignments.WGSL,
    **statements.WGSL,
    **lvalues.WGSL,
    **aggregates.WGSL,
    **values.WGSL,
    **edges.WGSL,
    **lexical.WGSL,
    **forms.WGSL,
    **idioms.WGSL,
    **abstractions.WGSL,
    **portability.WGSL,
    **initializers.WGSL,
    **helpers.WGSL,
    **layouts.WGSL,
    **elements.WGSL,
    **blocks.WGSL,
    **workgroups.WGSL,
    **records.WGSL,
    **identifiers.WGSL,
    **integers.WGSL,
    **columns.WGSL,
    **booleans.WGSL,
    **packing.WGSL,
    **orientation.WGSL,
    **rows.WGSL,
    **vectors.WGSL,
    **relations.WGSL,
    **fragments.WGSL,
    **interfaces.WGSL,
    **vertices.WGSL,
    **draws.WGSL,
    **rasterization.WGSL,
    **pipeline.WGSL,
    **extensions.WGSL,
    **primitives.WGSL,
    **tessellation.WGSL,
    **meshes.WGSL,
    **tasks.WGSL,
    **rays.WGSL,
    **bindings.WGSL,
    **interlocks.WGSL,
    **components.ELSEWHERE,
    **subgroups.WGSL,
    **specializations.WGSL,
    **qualifiers.WGSL,
    **textures.WGSL,
    **images.WGSL,
    **formats.WGSL,
    **storage.WGSL,
    **samplers.WGSL,
    "global-constant": Template("outp[i] = fin[i] * SCALE;", "const SCALE: f32 = 2.0;"),
    "runtime-array-length": Template("outp[i] = f32(arrayLength(&outp));"),
    "half": Template(
        "let v = f16(fin[i]);\noutp[i] = f32(v * v);", prelude="enable f16;"
    ),
    **dict.fromkeys(
        ("double", "int8", "uint8", "int16", "uint16", "int64", "uint64"), ABSENT
    ),
    "shared-memory": Template(
        "tile[0] = fin[i];\nworkgroupBarrier();\noutp[i] = tile[0];",
        "var<workgroup> tile: array<f32, 1>;",
    ),
    "local-invocation-id": Template(
        "outp[i] = f32(lid.x);", params="@builtin(local_invocation_id) lid: vec3<u32>"
    ),
    "workgroup-id": Template(
        "outp[i] = f32(gid.x);", params="@builtin(workgroup_id) gid: vec3<u32>"
    ),
    "local-invocation-index": Template(
        "outp[i] = f32(lidx);", params="@builtin(local_invocation_index) lidx: u32"
    ),
    "num-workgroups": Template(
        "outp[i] = f32(groups.x);", params="@builtin(num_workgroups) groups: vec3<u32>"
    ),
    "uniform-buffer": Template(
        "outp[i] = fin[i] * params.scale;",
        "struct Params {\n    scale: f32,\n}\n@group(0) @binding(3) var<uniform> params: Params;",
    ),
    "read-only-buffer": Template(
        "outp[i] = rin[i];", "@group(0) @binding(3) var<storage, read> rin: array<f32>;"
    ),
    "sampled-texture": Template(
        "outp[i] = textureSampleLevel(tex, smp, vec2<f32>(fin[i], 0.5), 0.0).x;",
        "@group(0) @binding(4) var tex: texture_2d<f32>;\n@group(0) @binding(5) var smp: sampler;",
    ),
    "storage-image": Template(
        "textureStore(img, vec2<i32>(0, 0), vec4<f32>(fin[i]));\noutp[i] = 0.0;",
        "@group(0) @binding(6) var img: texture_storage_2d<rgba32float, write>;",
    ),
    "compute": C_FAMILY["compute"],
    "vertex": Template(
        program="@vertex\nfn main(@location(0) position: vec4<f32>) -> @builtin(position) vec4<f32> {\n"
        "    return position * 2.0;\n}\n"
    ),
    "fragment": Template(
        program="@fragment\nfn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {\n"
        "    return vec4<f32>(uv, 0.0, 1.0);\n}\n"
    ),
    "discard": Template(
        program="@fragment\nfn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {\n"
        "    if (uv.x < 0.5) {\n        discard;\n    }\n    return vec4<f32>(uv, 0.0, 1.0);\n}\n"
    ),
    # WGSL has only vertex, fragment and compute shaders.
    **dict.fromkeys(
        (
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
