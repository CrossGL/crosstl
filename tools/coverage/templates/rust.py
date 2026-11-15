"""Probe templates for Rust (rust-gpu)."""

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
    memory,
    meshes,
    operators,
    orientation,
    packing,
    pipeline,
    pointers,
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
    **functions.RUST,
    **constructors.RUST,
    **scopes.RUST,
    **declarations.RUST,
    **calls.RUST,
    **branches.RUST,
    **shapes.RUST,
    **evaluation.RUST,
    **assignments.RUST,
    **statements.RUST,
    **lvalues.RUST,
    **aggregates.RUST,
    **values.RUST,
    **edges.RUST,
    **lexical.RUST,
    **forms.RUST,
    **idioms.RUST,
    **abstractions.RUST,
    **portability.RUST,
    **initializers.RUST,
    **helpers.RUST,
    **layouts.RUST,
    **elements.RUST,
    **blocks.RUST,
    **workgroups.RUST,
    **records.RUST,
    **identifiers.RUST,
    **integers.RUST,
    **columns.RUST,
    **booleans.RUST,
    **packing.RUST,
    **orientation.RUST,
    **rows.RUST,
    **vectors.RUST,
    **relations.RUST,
    **fragments.RUST,
    **interfaces.RUST,
    **vertices.RUST,
    **draws.RUST,
    **rasterization.RUST,
    **pipeline.RUST,
    **extensions.RUST,
    **primitives.RUST,
    **tessellation.RUST,
    **meshes.RUST,
    **tasks.RUST,
    **rays.RUST,
    **bindings.RUST,
    **interlocks.RUST,
    **components.ELSEWHERE,
    **pointers.RUST,
    **memory.ELSEWHERE,
    **subgroups.RUST,
    **specializations.RUST,
    **qualifiers.RUST,
    **textures.RUST,
    **images.RUST,
    **formats.RUST,
    **storage.RUST,
    **samplers.RUST,
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
