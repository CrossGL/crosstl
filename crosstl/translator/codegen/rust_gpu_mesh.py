"""Mesh and task entry points for rust-gpu (SPV_EXT_mesh_shader), written from CrossGL's HLSL form of those stages.

CrossGL writes a mesh stage the way HLSL shader model 6.5 does: the entry point takes its output vertices, primitive
indices and per-primitive values as ``out`` array parameters marked with their role, the task payload as an ``in``
parameter, and built-in inputs as parameters with their semantic; ``SetMeshOutputCounts(v, p)`` sets the counts::

    mesh {
        void main(@ vertices out Vertex verts[3], @ indices out uvec3 tris[1], @ primitives out Primitive prims[1],
                  @ mesh_payload in Payload payload, uvec3 group @ gl_WorkGroupID)
            @numthreads(1, 1, 1) @outputtopology(triangle) {
            SetMeshOutputCounts(3, 1); verts[0].position = ...; tris[0] = uvec3(0, 1, 2); prims[0].color = ...;
        }
    }

A task stage declares its payload at module scope (``groupshared Payload payload @ taskPayloadSharedEXT;``) and hands
it on with ``DispatchMesh(x, y, z, payload)`` or GLSL's ``EmitMeshTasksEXT(x, y, z)``.

rust-gpu (0.11) declares a mesh shader as ``#[spirv(mesh_ext(threads(1, 1, 1), output_vertices = 3,
output_primitives_ext = 1, output_triangles_ext))]`` whose parameters are arrays in the Output storage class, one
per vertex or primitive member (rustc_codegen_spirv/src/symbols.rs, ``EXECUTION_MODES`` and ``BUILTINS``; the
spirv-std mesh example). :func:`lower_rust_mesh_stages` therefore rewrites the entry point of each mesh and task stage
before it is written: every member of the vertex and primitive structures becomes an entry-parameter record
``<param>_<member>: &mut [T; N]``, a built-in member with its rust-gpu built-in (``position``, ``point_size``,
``primitive_id``, ``layer``, ``viewport_index``, ``cull_primitive_ext``, each primitive one ``per_primitive_ext``),
a user member with the location the other targets give it (its explicit one, else its semantic's, else the first
free one; per-primitive locations follow the per-vertex ones); ``verts[i].member`` reads ``verts_member[i]``. The
index array is ``primitive_triangle_indices_ext``, ``primitive_line_indices_ext`` or ``primitive_point_indices_ext``,
the payload ``task_payload_workgroup_ext``, and a built-in input parameter takes its compute built-in. Records are
threaded to helpers and written through their reference like geometry outputs (:mod:`.rust_gpu_geometry`).
``SetMeshOutputCounts``/``SetMeshOutputsEXT`` call ``spirv_std::arch::set_mesh_outputs_ext``, and
``DispatchMesh``/``EmitMeshTasksEXT`` call ``emit_mesh_tasks_ext_payload`` or ``emit_mesh_tasks_ext``.

Vulkan requires a 32-bit signed integer for ``PrimitiveId``, ``Layer`` and ``ViewportIndex``
(VUID-PrimitiveId-PrimitiveId-04337 and the Layer and ViewportIndex rules), so an HLSL ``uint`` member of those
built-ins is written as an ``i32`` record and converted on each read and write.

Refused, with the value named: a mesh entry point in a program with other entry points (spirv-val checks the
primitive index built-in against every entry point of the module), a mesh entry point without vertices and indices,
an output that is not a sized array, a structure member of an output structure, an array member (clip and cull
distances), an index type that does not match the output topology, a task payload that is not a module-scope
variable, and parameters of any other kind.
"""

from __future__ import annotations

from ..ast import (
    ArrayAccessNode,
    ArrayType,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    MeshOpNode,
    PrimitiveType,
    VariableNode,
)
from ..semantics.rewrite import child_slots, get_child, set_child
from ..semantics.stage_varyings import explicit_location, location_span
from .rust_gpu_geometry import STAGE_INPUT_KIND, STAGE_OUTPUT_KIND
from .rust_gpu_interface import COMPUTE_SEMANTIC_BUILTINS, workgroup_type_name
from .rust_gpu_subset import generator_stage_name, program_stage_names
from .rust_resource_bindings import referenced_identifier_names
from .stage_io_semantic_locations import allocate_locations, semantic_location

__all__ = [
    "MESH_STAGES",
    "generate_rust_mesh_call",
    "lower_rust_mesh_stages",
    "rust_mesh_stage_attribute",
]

MESH_STAGES = frozenset({"mesh", "task", "amplification", "object"})
_TASK_STAGES = frozenset({"task", "amplification", "object"})
_PAYLOAD_ATTRIBUTE = "task_payload_workgroup_ext"
_PAYLOAD_NAMES = frozenset({"taskpayloadsharedext", "payload"})

# Output topology -> (execution mode, CrossGL index type, index built-in).
_TOPOLOGIES = {
    "triangle": ("output_triangles_ext", "uvec3", "primitive_triangle_indices_ext"),
    "line": ("output_lines_ext", "uvec2", "primitive_line_indices_ext"),
    "point": ("output_points", "uint", "primitive_point_indices_ext"),
}
_VERTEX_BUILTINS = {
    "gl_position": "position",
    "sv_position": "position",
    "gl_pointsize": "point_size",
    "psize": "point_size",
}
_PRIMITIVE_BUILTINS = {
    "gl_primitiveid": "primitive_id",
    "sv_primitiveid": "primitive_id",
    "gl_layer": "layer",
    "sv_rendertargetarrayindex": "layer",
    "gl_viewportindex": "viewport_index",
    "sv_viewportarrayindex": "viewport_index",
    "gl_cullprimitiveext": "cull_primitive_ext",
    "sv_cullprimitive": "cull_primitive_ext",
}
_SIGNED_BUILTINS = frozenset({"primitive_id", "layer", "viewport_index"})
_ROLES = {
    "vertices": "vertices",
    "vertex": "vertices",
    "indices": "indices",
    "index": "indices",
    "primitives": "primitives",
    "primitive": "primitives",
    "mesh_payload": "payload",
    "meshpayload": "payload",
    "payload": "payload",
}
_COUNT_CALLS = frozenset({"SetMeshOutputCounts", "SetMeshOutputsEXT"})
_TASK_CALLS = frozenset({"DispatchMesh", "EmitMeshTasksEXT"})


def _lower_names(items):
    return [
        str(getattr(item, "name", item)).strip().lower().replace("-", "_")
        for item in items or []
    ]


def _literal(node):
    value = node.value if isinstance(node, LiteralNode) else node
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip().rstrip("uU") if isinstance(value, str) else ""
    return int(text) if text.isdigit() else None


def _stage_name(stage_type):
    return str(getattr(stage_type, "value", stage_type)).split(".")[-1].lower()


def _role(parameter):
    for name in _lower_names(getattr(parameter, "attributes", None)):
        if name in _ROLES:
            return _ROLES[name]
    return None


def _member_key(generator, member):
    return str(generator.rust_stage_io_semantic(member) or "").strip().lower()


class _MeshPlan:
    """The records and member renames of one mesh or task entry point."""

    def __init__(self):
        self.records = {}
        self.members = {}  # (parameter name, member name) -> record name
        self.renames = (
            {}
        )  # parameter name -> record name, for a name the program uses elsewhere
        self.modes = []
        self.outside = (
            frozenset()
        )  # names referenced outside the entry point being planned
        self.used = set()

    def name(self, stage, parameter_name):
        """The record name of a parameter: its own, unless the program uses that name outside the entry point."""
        if parameter_name not in self.outside:
            return parameter_name
        name = _unique(f"{stage}_{parameter_name}", self.used)
        self.renames[parameter_name] = name
        return name


def _array_size(parameter):
    param_type = getattr(parameter, "param_type", None)
    if not isinstance(param_type, ArrayType):
        return None, None
    return param_type.element_type, _literal(param_type.size)


def _unique(name, used):
    candidate, suffix = name, 2
    while candidate in used:
        candidate, suffix = f"{name}_{suffix}", suffix + 1
    used.add(candidate)
    return candidate


def _record(node, type_name, kind, attribute):
    return {"node": node, "type_name": type_name, "kind": kind, "attribute": attribute}


def _plan_struct_outputs(generator, plan, parameter, used, builtins, per_primitive):
    """Records for each member of an output structure array; ``(record name, location key, span)`` of user ones."""
    element, size = _array_size(parameter)
    element_name = generator.convert_type_node_to_string(element)
    members = generator.rust_stage_io_members(element_name)
    if members is None:
        generator.refuse_lowering(
            f"stage: mesh output '{parameter.name}' is an array of {element_name}; rust-gpu mesh outputs are "
            "arrays of interface structures"
        )
        return []
    users = []
    for member_name, member_type, member_node in members:
        if generator.rust_stage_io_members(member_type) is not None or "[" in str(
            member_type
        ):
            generator.refuse_lowering(
                f"stage: mesh output '{parameter.name}.{member_name}' is a {member_type}; rust-gpu mesh outputs "
                "are scalars and vectors"
            )
            continue
        name = _unique(f"{parameter.name}_{member_name}", used)
        builtin = builtins.get(_member_key(generator, member_node))
        if builtin in _SIGNED_BUILTINS and generator.map_type(member_type) == "u32":
            # Vulkan requires a 32-bit signed integer (VUID-PrimitiveId-PrimitiveId-04337 and the Layer and
            # ViewportIndex rules); HLSL's uint values convert on every read and write.
            member_type = "int"
        node = VariableNode(
            name,
            ArrayType(member_node.member_type, LiteralNode(size, PrimitiveType("int"))),
        )
        type_name = workgroup_type_name(f"{member_type}[{size}]")
        suffix = ", per_primitive_ext" if per_primitive else ""
        plan.members[(parameter.name, member_name)] = name
        if builtin is not None:
            plan.records[name] = _record(
                node, type_name, STAGE_OUTPUT_KIND, builtin + suffix
            )
            continue
        plan.records[name] = _record(node, type_name, STAGE_OUTPUT_KIND, suffix)
        fixed = explicit_location(member_node)
        if fixed is None:
            fixed = semantic_location(generator.rust_stage_io_semantic(member_node))
        users.append((name, fixed, location_span(member_node.member_type)))
    return users


def _assign_locations(plan, users, occupied=()):
    locations = allocate_locations(
        [(fixed, span) for _n, fixed, span in users], occupied
    )
    taken = set()
    for (name, _fixed, span), location in zip(users, locations):
        record = plan.records[name]
        record["attribute"] = f"location = {location}{record['attribute']}"
        taken.update(range(location, location + max(span, 1)))
    return taken


def _plan_mesh_entry(generator, func, used, outside):
    plan = _MeshPlan()
    plan.outside, plan.used = outside, used
    topology = generator.rust_mesh_output_topology(func, "mesh", required=True)
    topology = {"triangle_cw": "triangle", "triangle_ccw": "triangle"}.get(
        topology, topology
    )
    mode, index_type, index_builtin = _TOPOLOGIES[topology]
    roles = {}
    vertex_users, primitive_users = [], []
    for parameter in getattr(func, "parameters", None) or []:
        role = _role(parameter)
        if role in roles:
            generator.refuse_lowering(
                f"stage: mesh entry point '{func.name}' takes two {role} parameters"
            )
            continue
        if role is None:
            _plan_builtin_input(generator, plan, func, parameter)
            continue
        roles[role] = parameter
        if role == "payload":
            type_name = generator.function_parameter_type(parameter)
            shared = generator.rust_stage_values.get(parameter.name)
            if shared is not None:
                # The task stage's module-scope payload: the mesh entry reads the same task payload storage.
                if shared.get("attribute") != _PAYLOAD_ATTRIBUTE or shared[
                    "type_name"
                ] != workgroup_type_name(type_name):
                    generator.refuse_lowering(
                        f"stage: mesh payload '{parameter.name}' is a {type_name}, and the program's other "
                        f"'{parameter.name}' is not a task payload of that type"
                    )
                continue
            record_name = plan.name("mesh", parameter.name)
            plan.records[record_name] = _record(
                VariableNode(record_name, parameter.param_type),
                workgroup_type_name(type_name),
                STAGE_OUTPUT_KIND,
                _PAYLOAD_ATTRIBUTE,
            )
            continue
        element, size = _array_size(parameter)
        if not size:
            generator.refuse_lowering(
                f"stage: mesh output '{parameter.name}' is not a sized array"
            )
            continue
        if role == "vertices":
            plan.modes.append(f"output_vertices = {size}")
            vertex_users = _plan_struct_outputs(
                generator, plan, parameter, used, _VERTEX_BUILTINS, False
            )
        elif role == "primitives":
            primitive_users = _plan_struct_outputs(
                generator, plan, parameter, used, _PRIMITIVE_BUILTINS, True
            )
        else:
            plan.modes.append(f"output_primitives_ext = {size}")
            element_name = generator.convert_type_node_to_string(element)
            if generator.map_type(element_name) != generator.map_type(index_type):
                generator.refuse_lowering(
                    f"stage: mesh indices '{parameter.name}' are {element_name}; {topology} outputs take "
                    f"{index_type} indices"
                )
                continue
            record_name = plan.name("mesh", parameter.name)
            plan.records[record_name] = _record(
                VariableNode(record_name, parameter.param_type),
                workgroup_type_name(f"{index_type}[{size}]"),
                STAGE_OUTPUT_KIND,
                index_builtin,
            )
    if "vertices" not in roles or "indices" not in roles:
        generator.refuse_lowering(
            f"stage: mesh entry point '{func.name}' needs a vertices and an indices output parameter"
        )
    plan.modes.append(mode)
    taken = _assign_locations(plan, vertex_users)
    _assign_locations(plan, primitive_users, taken)
    return plan


def _plan_builtin_input(generator, plan, func, parameter):
    semantic = generator.rust_stage_io_semantic(parameter)
    key = generator.rust_stage_parameter_semantic_key(semantic) if semantic else None
    builtin = COMPUTE_SEMANTIC_BUILTINS.get(key)
    type_name = generator.function_parameter_type(parameter)
    if builtin is None or generator.map_type(type_name) != generator.map_type(
        builtin[1]
    ):
        generator.refuse_lowering(
            f"stage: entry parameter '{parameter.name}' of '{func.name}' is neither a mesh output, a task payload "
            "nor a compute built-in input of its built-in's type"
        )
        return
    record_name = plan.name(generator.current_mesh_stage, parameter.name)
    plan.records[record_name] = _record(
        VariableNode(record_name, parameter.param_type),
        type_name,
        STAGE_INPUT_KIND,
        builtin[0],
    )


def _plan_task_payloads(generator, ast, plan):
    for node in getattr(ast, "global_variables", None) or []:
        names = set(_lower_names(getattr(node, "attributes", None)))
        if not isinstance(node, VariableNode) or not names & _PAYLOAD_NAMES:
            continue
        type_name = generator.rust_variable_source_type(node)
        plan.records[node.name] = _record(
            node, workgroup_type_name(type_name), STAGE_OUTPUT_KIND, _PAYLOAD_ATTRIBUTE
        )


def _rewrite_members(plan):
    def rewrite(node):
        if isinstance(node, IdentifierNode) and node.name in plan.renames:
            return IdentifierNode(plan.renames[node.name])
        if isinstance(node, MemberAccessNode) and isinstance(
            node.object, ArrayAccessNode
        ):
            array = node.object.array
            if isinstance(array, IdentifierNode):
                name = plan.members.get((array.name, node.member))
                if name is not None:
                    return ArrayAccessNode(IdentifierNode(name), node.object.index)
        return None

    return rewrite


def _rewrite_tree(node, rewrite):
    """Replace, bottom-up, every descendant of ``node`` that ``rewrite`` returns a replacement for."""
    for field, key in list(child_slots(node)):
        child = get_child(node, field, key)
        _rewrite_tree(child, rewrite)
        replacement = rewrite(child)
        if replacement is not None:
            set_child(node, field, key, replacement)


def _names_outside(ast, stage):
    """The names identifiers spell anywhere in ``ast`` except in ``stage``."""
    parts = list(getattr(ast, "functions", None) or [])
    parts += list(getattr(ast, "global_variables", None) or [])
    for other in (getattr(ast, "stages", None) or {}).values():
        if other is not stage:
            parts.append(other)
    names = set()
    for part in parts:
        names |= referenced_identifier_names(part)
    return frozenset(names)


def lower_rust_mesh_stages(generator, ast) -> None:
    """Rewrite the mesh and task entry points of ``ast`` for rust-gpu and record their interface on ``generator``.

    Adds the records to ``generator.rust_stage_values`` and the execution modes of each entry point, keyed by the
    entry's ``id``, to ``generator.rust_mesh_modes``.
    """
    generator.rust_mesh_modes = {}
    used = set(getattr(generator, "rust_referenced_names", ()) or ())
    stage_names = program_stage_names(ast)
    if "mesh" in stage_names and len(stage_names) > 1:
        generator.refuse_lowering(
            "stage: a mesh entry point must be the only entry point of its rust-gpu crate; spirv-val checks the "
            "primitive index built-in against every entry point's execution modes "
            "(VUID-PrimitiveTriangleIndicesEXT-PrimitiveTriangleIndicesEXT-07054)"
        )
    if stage_names & MESH_STAGES:
        payloads = _MeshPlan()
        _plan_task_payloads(generator, ast, payloads)
        generator.rust_stage_values.update(payloads.records)
    for func in getattr(ast, "functions", None) or []:
        stage = generator_stage_name(func)
        if stage in MESH_STAGES:
            generator.refuse_lowering(
                f"stage: {stage} entry point '{func.name}' is a stage-qualified function; rust-gpu output writes "
                "mesh and task entry points from stage blocks"
            )
    for stage_type, stage in (getattr(ast, "stages", None) or {}).items():
        name = _stage_name(stage_type)
        func = getattr(stage, "entry_point", None)
        if name not in MESH_STAGES or func is None:
            continue
        threads = generator.rust_mesh_stage_numthreads(func, name) or (
            generator.rust_mesh_stage_layout_local_size(stage, name)
        )
        if threads is None:
            generator.refuse_lowering(
                f"stage: {name} entry point '{func.name}' names no thread-group size"
            )
            threads = ("1", "1", "1")
        generator.current_mesh_stage = name
        outside = _names_outside(ast, stage)
        if name in _TASK_STAGES:
            plan = _MeshPlan()
            plan.outside, plan.used = outside, used
            for parameter in getattr(func, "parameters", None) or []:
                _plan_builtin_input(generator, plan, func, parameter)
            attribute = "task_ext"
        else:
            plan = _plan_mesh_entry(generator, func, used, outside)
            attribute = "mesh_ext"
        func.parameters = []
        if hasattr(func, "params"):
            func.params = []
        _rewrite_tree(func, _rewrite_members(plan))
        generator.rust_stage_values.update(plan.records)
        modes = [f"threads({', '.join(threads)})", *plan.modes]
        generator.rust_mesh_modes[id(func)] = f"{attribute}({', '.join(modes)})"


def rust_mesh_stage_attribute(generator, func) -> str:
    return f"#[spirv({generator.rust_mesh_modes[id(func)]})]\n"


def _call_name(expr):
    if isinstance(expr, MeshOpNode):
        return expr.operation
    if isinstance(expr, FunctionCallNode):
        function = getattr(expr, "function", None)
        return getattr(function, "name", function)
    return None


def generate_rust_mesh_call(generator, expr):
    """The ``spirv_std::arch`` call of a mesh or task built-in, or ``None`` when ``expr`` is not one."""
    name = _call_name(expr)
    if name not in _COUNT_CALLS and name not in _TASK_CALLS:
        return None
    if isinstance(expr, FunctionCallNode) and generator.is_user_defined_function(name):
        return None
    stage = generator.current_shader_type
    arguments = list(getattr(expr, "arguments", None) or [])
    if isinstance(expr, MeshOpNode):
        # The stage and operand contracts of the HLSL intrinsics (rust_stage_contracts).
        generator.validate_rust_mesh_op_stage(name, stage)
        generator.validate_rust_mesh_op_arguments(name, arguments, stage)
    expected_stages = {"mesh"} if name in _COUNT_CALLS else _TASK_STAGES
    if stage not in expected_stages:
        return generator.refuse_lowering(
            f"function: '{name}' is called outside a {'mesh' if name in _COUNT_CALLS else 'task'} entry point",
            "()",
        )
    values = [
        generator.generate_expression_with_type(argument, "uint")
        for argument in arguments[:3]
    ]
    if name in _COUNT_CALLS:
        if len(arguments) != 2:
            return None
        return f"unsafe {{ spirv_std::arch::set_mesh_outputs_ext({values[0]}, {values[1]}) }}"
    if len(arguments) == 3:
        return f"unsafe {{ spirv_std::arch::emit_mesh_tasks_ext({', '.join(values)}) }}"
    payload = arguments[3] if len(arguments) == 4 else None
    payload_name = getattr(payload, "name", None)
    record = (getattr(generator, "rust_stage_values", None) or {}).get(payload_name)
    if (
        record is None
        or record.get("attribute") != _PAYLOAD_ATTRIBUTE
        or not generator.is_rust_workgroup_reference(payload_name)
    ):
        return generator.refuse_lowering(
            f"function: '{name}' hands on a payload that is not the task payload variable",
            "()",
        )
    reference = generator.rust_identifier(payload.name)
    return f"unsafe {{ spirv_std::arch::emit_mesh_tasks_ext_payload({', '.join(values)}, {reference}) }}"
