"""DXR inline ray queries as rust-gpu's ``spirv_std::ray_tracing::RayQuery``.

CrossGL spells inline ray queries the DXR way (:mod:`crosstl.translator.ray_query_vocabulary`): a ``RayQuery<FLAGS>``
local, ``TraceRayInline`` with an acceleration structure, ray flags, an instance mask and a ``RayDesc``, then
``Proceed`` and the candidate and committed getters. spirv-std 0.11 declares a query with its ``ray_query!`` macro, an
``OpVariable`` of ``OpTypeRayQueryKHR`` in the Function storage class, and has an ``unsafe`` method for each
SPV_KHR_ray_query instruction, so each CrossGL method is one spirv-std call:

- the flags of ``RayQuery<FLAGS>`` are combined with the flags operand of ``initialize``, which is what DXR does with
  them (D3D12 raytracing specification, "RayQuery");
- ``CandidateType`` and ``CommittedStatus`` return spirv-std's ``CandidateIntersection`` and ``CommittedIntersection``
  enums, whose discriminants are the DXR values, and ``RayFlags`` a ``RayFlags`` bit set, so each is read as a ``u32``;
- DXR's ``CandidateProceduralPrimitiveNonOpaque`` is the negation of ``get_intersection_candidate_aabb_opaque``;
- ``CommitProceduralPrimitiveHit`` and the committed triangle's vertex positions have no spirv-std method, so they are
  the SPIR-V instruction in ``asm!``, written the way spirv-std writes its own methods;
- the DXR ray flag and intersection kind constants are their ``u32`` values, which are the same in DXR and SPIR-V;
- a ``RayDesc`` the program does not declare is a struct with DXR's four fields;
- an acceleration structure is a ``&AccelerationStructure`` resource parameter, which :mod:`.rust_gpu_images` binds
  with the images and samplers.

The object-to-world and world-to-object matrices are 4x3 and 3x4 matrices, which glam has no type for, so they are
refused by name, as is a query that is not a local of a function: spirv-std makes a query only with ``ray_query!``.
"""

from __future__ import annotations

import re

from ..ast import (
    ArrayAccessNode,
    BinaryOpNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    NamedType,
    PrimitiveType,
    RayQueryOpNode,
    StructMemberNode,
    StructNode,
    VectorType,
)
from ..ray_query_vocabulary import (
    ray_flag_value,
    ray_query_constant,
    ray_query_type_flags,
)
from .ast_walk import walk_ast_values

# The CrossGL spellings of an acceleration structure type.
ACCELERATION_STRUCTURE_TYPE_NAMES = frozenset(
    {
        "accelerationStructureEXT",
        "accelerationStructure",
        "AccelerationStructure",
        "RaytracingAccelerationStructure",
        "RayTracingAccelerationStructure",
    }
)

# DXR getter -> (spirv-std method, CrossGL result type, Rust form of the call's value).
_GETTERS = {
    "Proceed": ("proceed", "bool", "{}"),
    "RayTMin": ("get_ray_t_min", "float", "{}"),
    "RayFlags": ("get_ray_flags", "uint", "({}).bits()"),
    "WorldRayOrigin": ("get_world_ray_origin", "vec3", "{}"),
    "WorldRayDirection": ("get_world_ray_direction", "vec3", "{}"),
    "CandidateType": ("get_candidate_intersection_type", "uint", "({}) as u32"),
    "CommittedStatus": ("get_committed_intersection_type", "uint", "({}) as u32"),
    "CandidateTriangleRayT": ("get_candidate_intersection_t", "float", "{}"),
    "CommittedRayT": ("get_committed_intersection_t", "float", "{}"),
    # DXR's InstanceID is the application's custom index, its InstanceIndex the instance's position.
    "CandidateInstanceID": (
        "get_candidate_intersection_instance_custom_index",
        "uint",
        "{}",
    ),
    "CommittedInstanceID": (
        "get_committed_intersection_instance_custom_index",
        "uint",
        "{}",
    ),
    "CandidateInstanceIndex": ("get_candidate_intersection_instance_id", "uint", "{}"),
    "CommittedInstanceIndex": ("get_committed_intersection_instance_id", "uint", "{}"),
    "CandidateInstanceContributionToHitGroupIndex": (
        "get_candidate_intersection_shader_binding_table_record_offset",
        "uint",
        "{}",
    ),
    "CommittedInstanceContributionToHitGroupIndex": (
        "get_committed_intersection_shader_binding_table_record_offset",
        "uint",
        "{}",
    ),
    "CandidateGeometryIndex": (
        "get_candidate_intersection_geometry_index",
        "uint",
        "{}",
    ),
    "CommittedGeometryIndex": (
        "get_committed_intersection_geometry_index",
        "uint",
        "{}",
    ),
    "CandidatePrimitiveIndex": (
        "get_candidate_intersection_primitive_index",
        "uint",
        "{}",
    ),
    "CommittedPrimitiveIndex": (
        "get_committed_intersection_primitive_index",
        "uint",
        "{}",
    ),
    "CandidateTriangleBarycentrics": (
        "get_candidate_intersection_barycentrics",
        "vec2",
        "{}",
    ),
    "CommittedTriangleBarycentrics": (
        "get_committed_intersection_barycentrics",
        "vec2",
        "{}",
    ),
    "CandidateTriangleFrontFace": (
        "get_candidate_intersection_front_face",
        "bool",
        "{}",
    ),
    "CommittedTriangleFrontFace": (
        "get_committed_intersection_front_face",
        "bool",
        "{}",
    ),
    # DXR asks whether the candidate is non-opaque, SPIR-V whether it is opaque.
    "CandidateProceduralPrimitiveNonOpaque": (
        "get_intersection_candidate_aabb_opaque",
        "bool",
        "!({})",
    ),
    "CandidateObjectRayOrigin": (
        "get_candidate_intersection_object_ray_origin",
        "vec3",
        "{}",
    ),
    "CommittedObjectRayOrigin": (
        "get_committed_intersection_object_ray_origin",
        "vec3",
        "{}",
    ),
    "CandidateObjectRayDirection": (
        "get_candidate_intersection_object_ray_direction",
        "vec3",
        "{}",
    ),
    "CommittedObjectRayDirection": (
        "get_committed_intersection_object_ray_direction",
        "vec3",
        "{}",
    ),
}
# Methods with no result.
_COMMANDS = {
    "Abort": "terminate",
    "CommitNonOpaqueTriangleHit": "confirm_intersection",
}
_INITIALIZE = frozenset({"TraceRayInline", "Initialize"})
_VERTEX_POSITIONS = {
    "CandidateTriangleVertexPositions": 0,
    "CommittedTriangleVertexPositions": 1,
}
_MATRICES = frozenset(
    {
        "CandidateObjectToWorld",
        "CommittedObjectToWorld",
        "CandidateObjectToWorld3x4",
        "CommittedObjectToWorld3x4",
        "CandidateObjectToWorld4x3",
        "CommittedObjectToWorld4x3",
        "CandidateWorldToObject",
        "CommittedWorldToObject",
        "CandidateWorldToObject3x4",
        "CommittedWorldToObject3x4",
        "CandidateWorldToObject4x3",
        "CommittedWorldToObject4x3",
    }
)
_ARITIES = {
    **{name: (0,) for name in _GETTERS},
    **{name: (0,) for name in _COMMANDS},
    **{name: (1,) for name in _VERTEX_POSITIONS},
    "CommitProceduralPrimitiveHit": (1,),
    "TraceRayInline": (4, 7),
    "Initialize": (7,),
}
RAY_QUERY_OPERATIONS = frozenset(_ARITIES) | _MATRICES
_RAY_DESC_FIELDS = (
    ("Origin", "vec3"),
    ("TMin", "float"),
    ("Direction", "vec3"),
    ("TMax", "float"),
)
_CONSTANT_PREFIXES = (
    "RAY_FLAG_",
    "COMMITTED_",
    "CANDIDATE_",
    "gl_RayFlags",
    "gl_RayQuery",
)
_RAY_QUERY_MACRO = re.compile(r"\bray_query!\(")
_ACCELERATION_STRUCTURE = re.compile(r"\bAccelerationStructure\b")


def is_ray_query_type(type_name) -> bool:
    """Whether the CrossGL type ``type_name`` is ``RayQuery<FLAGS>``, or the GLSL importer's ``rayQueryEXT``."""
    return (
        re.match(r"^\s*(?:RayQuery\b|rayQuery(?:EXT)?$)", str(type_name or ""))
        is not None
    )


def is_acceleration_structure_type(type_name) -> bool:
    """Whether the CrossGL type ``type_name`` names an acceleration structure."""
    return str(type_name or "").strip() in ACCELERATION_STRUCTURE_TYPE_NAMES


def _flag_operands(node):
    """The ray flag operands of one ``RayQuery`` template argument, or ``None`` for one that is not a flag."""
    if (
        isinstance(node, BinaryOpNode)
        and str(getattr(node, "op", getattr(node, "operator", ""))) == "|"
    ):
        left, right = _flag_operands(node.left), _flag_operands(node.right)
        return None if left is None or right is None else left + right
    if isinstance(node, (IdentifierNode, NamedType)) and not getattr(
        node, "generic_args", None
    ):
        return (node.name,)
    if isinstance(node, LiteralNode) and isinstance(node.value, int):
        return (str(node.value),)
    return None


def ray_query_flags_value(generator, type_node):
    """The combined value of the template flags of the ``RayQuery`` type ``type_node``, or ``None``."""
    if isinstance(type_node, NamedType) and type_node.name in {
        "RayQuery",
        "rayQuery",
        "rayQueryEXT",
    }:
        operands = ()
        for argument in type_node.generic_args or []:
            argument_operands = _flag_operands(argument)
            if argument_operands is None:
                return None
            operands += argument_operands
        return ray_flag_value(operands)
    flags = ray_query_type_flags(generator.type_name_string(type_node))
    return None if flags is None else ray_flag_value(flags)


def _receiver_type(generator, query):
    return generator.type_name_string(generator.expression_result_type(query)) or ""


def _is_query(generator, query):
    return is_ray_query_type(_receiver_type(generator, query))


def _unsafe(text):
    return f"unsafe {{ {text} }}"


def rust_gpu_ray_query_declaration(generator, stmt, vtype):
    """The ``ray_query!`` declaration of the ``RayQuery`` local ``stmt``, or ``None`` when it is not a query."""
    type_name = generator.type_name_string(vtype)
    if not is_ray_query_type(type_name):
        return None
    name = generator.rust_identifier(stmt.name)
    if getattr(stmt, "initial_value", None) is not None:
        generator.refuse_lowering(
            f"ray query: '{stmt.name}' has an initializer; spirv-std makes a query only with ray_query!"
        )
    declared = getattr(stmt, "vtype", None) or getattr(stmt, "var_type", None) or vtype
    if ray_query_flags_value(generator, declared) is None:
        generator.refuse_lowering(
            f"ray query: the type '{type_name}' of '{stmt.name}' has template flags that are not ray flags"
        )
    # The declared type node keeps the template flags as expressions; the registered type is its spelling.
    generator.rust_ray_query_types = getattr(generator, "rust_ray_query_types", {})
    generator.rust_ray_query_types[stmt.name] = declared
    # ray_query! declares `name` as a `&mut RayQuery` to an OpVariable of OpTypeRayQueryKHR.
    return f"spirv_std::ray_query!(let mut {name});"


def _template_flags(generator, query):
    name = query.name if isinstance(query, IdentifierNode) else None
    declared = (getattr(generator, "rust_ray_query_types", None) or {}).get(name)
    return ray_query_flags_value(
        generator, declared or generator.expression_result_type(query)
    )


def _initialize(generator, query, receiver, arguments):
    if len(arguments) == 4:
        acceleration, flags, mask, ray = arguments
        ray_type = generator.type_name_string(generator.expression_result_type(ray))
        if ray_type != "RayDesc":
            return generator.refuse_lowering(
                f"ray query: TraceRayInline takes a RayDesc, got '{ray_type or '?'}'",
                "()",
            )
        ray_values = [
            generator.generate_expression(MemberAccessNode(ray, field))
            for field, _type in _RAY_DESC_FIELDS
        ]
    else:
        acceleration, flags, mask, *rest = arguments
        ray_values = [
            generator.generate_expression_with_type(value, field_type)
            for value, (_field, field_type) in zip(rest, _RAY_DESC_FIELDS)
        ]
    template = _template_flags(generator, query)
    flags_text = generator.generate_expression_with_type(flags, "uint")
    if template:
        flags_text = f"({flags_text}) | {template:#x}u32"
    operands = [
        generator.generate_expression(acceleration),
        f"spirv_std::ray_tracing::RayFlags::from_bits_retain({flags_text})",
        generator.generate_expression_with_type(mask, "uint"),
        *ray_values,
    ]
    return _unsafe(f"{receiver}.initialize({', '.join(operands)})")


def _vertex_positions(generator, receiver, intersection, target):
    out = generator.generate_expression(target)
    if intersection == 0:
        return f"{out} = {_unsafe(f'{receiver}.get_intersection_triangle_vertex_positions()')}"
    # spirv-std 0.11 reads only the candidate's positions; the committed ones are the same instruction on
    # intersection 1, written as spirv-std writes the candidate method.
    return (
        f"{out} = {{ let mut positions: [Vec3; 3] = Default::default(); "
        'unsafe { asm!("%u32 = OpTypeInt 32 0", "%intersection = OpConstant %u32 1", '
        '"%result = OpRayQueryGetIntersectionTriangleVertexPositionsKHR typeof*{result} {ray_query} '
        '%intersection", "OpStore {result} %result", '
        f"ray_query = in(reg) &*{receiver}, result = in(reg) &mut positions) }}; positions }}"
    )


def generate_rust_gpu_ray_query_op(generator, operation, query, arguments):
    """The spirv-std call of the ray query method ``operation`` on ``query``, or a recorded refusal."""
    arguments = list(arguments or [])
    if not _is_query(generator, query):
        return generator.refuse_lowering(
            f"ray query: '{operation}' needs a RayQuery receiver, got "
            f"'{_receiver_type(generator, query) or '?'}'",
            "()",
        )
    if operation in _MATRICES:
        return generator.refuse_lowering(
            f"ray query: '{operation}' returns a 4x3 or 3x4 matrix, and glam has no non-square matrix type",
            "()",
        )
    arities = _ARITIES.get(operation)
    if arities is None:
        return generator.refuse_lowering(
            f"ray query: '{operation}' has no spirv_std::ray_tracing lowering", "()"
        )
    if len(arguments) not in arities:
        expected = " or ".join(str(arity) for arity in arities)
        return generator.refuse_lowering(
            f"ray query: '{operation}' takes {expected} arguments, got {len(arguments)}",
            "()",
        )
    receiver = generator.generate_expression(query)
    if operation in _INITIALIZE:
        return _initialize(generator, query, receiver, arguments)
    if operation in _COMMANDS:
        return _unsafe(f"{receiver}.{_COMMANDS[operation]}()")
    if operation in _VERTEX_POSITIONS:
        return _vertex_positions(
            generator, receiver, _VERTEX_POSITIONS[operation], arguments[0]
        )
    if operation == "CommitProceduralPrimitiveHit":
        # spirv-std 0.11 has no method for OpRayQueryGenerateIntersectionKHR.
        # The hit distance is bound to an f32 first: an asm! operand gives an untyped float literal no type.
        hit_t = generator.generate_expression_with_type(arguments[0], "float")
        generate = _unsafe(
            'asm!("OpRayQueryGenerateIntersectionKHR {} {}", '
            f"in(reg) &*{receiver}, in(reg) hit_t)"
        )
        return f"{{ let hit_t: f32 = {hit_t}; {generate} }}"
    method, _result_type, form = _GETTERS[operation]
    return form.format(_unsafe(f"{receiver}.{method}()"))


def _member_call_parts(generator, call):
    function = getattr(call, "function", getattr(call, "name", None))
    if not isinstance(function, MemberAccessNode):
        return None
    query = getattr(function, "object_expr", getattr(function, "object", None))
    if not _is_query(generator, query):
        return None
    return function.member, query, getattr(call, "arguments", getattr(call, "args", []))


def generate_rust_gpu_ray_query_member_call(generator, function, arguments):
    """The spirv-std call of ``query.Method(...)`` when ``query`` is a ``RayQuery``, else ``None``."""
    query = getattr(function, "object_expr", getattr(function, "object", None))
    if not _is_query(generator, query):
        return None
    return generate_rust_gpu_ray_query_op(generator, function.member, query, arguments)


def rust_gpu_ray_query_result_type(generator, expr):
    """The CrossGL result type of a ray query method call ``expr``, ``"void"`` for a command, else ``None``."""
    if isinstance(expr, RayQueryOpNode):
        operation, query = expr.operation, expr.query_expr
    elif isinstance(expr, FunctionCallNode):
        parts = _member_call_parts(generator, expr)
        if parts is None:
            return None
        operation, query, _arguments = parts
    else:
        return None
    if not _is_query(generator, query):
        return None
    getter = _GETTERS.get(operation)
    return getter[1] if getter is not None else "void"


def rust_gpu_ray_query_mutated_roots(generator, expr):
    """The local names a ray query call writes: the array a vertex-position query fills."""
    if isinstance(expr, RayQueryOpNode):
        operation, arguments = expr.operation, list(expr.arguments or [])
    elif isinstance(expr, FunctionCallNode):
        # Local types are registered as statements are written, after the mutated names are collected, so the
        # method name alone decides here; a `let mut` on an array no query writes is harmless.
        function = getattr(expr, "function", getattr(expr, "name", None))
        if not isinstance(function, MemberAccessNode):
            return set()
        operation = function.member
        arguments = list(getattr(expr, "arguments", getattr(expr, "args", [])) or [])
    else:
        return set()
    if operation not in _VERTEX_POSITIONS or len(arguments) != 1:
        return set()
    root = arguments[0]
    while isinstance(root, (ArrayAccessNode, MemberAccessNode)):
        root = getattr(root, "array_expr", None) or getattr(root, "object_expr", None)
    return {root.name} if isinstance(root, IdentifierNode) else set()


def rust_gpu_ray_query_constant(generator, name):
    """The ``u32`` value of a DXR or GLSL ray flag or intersection kind ``name`` the program does not declare."""
    if not isinstance(name, str) or not name.startswith(_CONSTANT_PREFIXES):
        return None
    if name in (getattr(generator, "variable_types", None) or {}):
        return None
    constant = ray_query_constant(name)
    return None if constant is None else f"{constant.value:#x}u32"


def _uses_type(ast, type_name):
    for value in walk_ast_values(ast):
        for attribute in (
            "vtype",
            "var_type",
            "param_type",
            "member_type",
            "return_type",
        ):
            declared = getattr(value, attribute, None)
            if getattr(declared, "name", None) == type_name or declared == type_name:
                return True
    return False


def _type_spelling(generator, type_node):
    if isinstance(type_node, NamedType):
        return type_node.name
    return generator.type_name_string(type_node) if type_node is not None else ""


def _refuse_queries_outside_locals(generator, ast):
    """Refuse each ``RayQuery`` that is a parameter, a return value, a struct member or a module-scope variable."""
    module_scope = list(getattr(ast, "global_variables", None) or [])
    for stage in (getattr(ast, "stages", None) or {}).values():
        module_scope.extend(getattr(stage, "local_variables", None) or [])
    for node in module_scope:
        if is_ray_query_type(_type_spelling(generator, getattr(node, "vtype", None))):
            generator.refuse_lowering(
                f"ray query: '{node.name}' is declared at module scope; spirv-std makes a query only as a "
                "function local with ray_query!"
            )
    for value in walk_ast_values(ast):
        if isinstance(value, StructNode):
            for member in value.members or []:
                if is_ray_query_type(
                    _type_spelling(generator, getattr(member, "member_type", None))
                ):
                    generator.refuse_lowering(
                        f"ray query: member '{member.name}' of struct '{value.name}' is a RayQuery; spirv-std "
                        "makes a query only as a function local with ray_query!"
                    )
            continue
        parameters = getattr(value, "parameters", None)
        if not isinstance(parameters, list) or not hasattr(value, "return_type"):
            continue
        if is_ray_query_type(_type_spelling(generator, value.return_type)):
            generator.refuse_lowering(
                f"ray query: function '{value.name}' returns a RayQuery, which spirv-std cannot copy"
            )
        for parameter in parameters:
            if is_ray_query_type(
                _type_spelling(generator, getattr(parameter, "param_type", None))
            ):
                generator.refuse_lowering(
                    f"ray query: parameter '{parameter.name}' of '{value.name}' is a RayQuery; the rust-gpu "
                    "output keeps each query in the function that declares it"
                )


def prepare_rust_gpu_ray_queries(generator, ast):
    """Refuse queries that are not function locals, and declare DXR's ``RayDesc`` in ``ast`` when the program uses
    the type without declaring it."""
    _refuse_queries_outside_locals(generator, ast)
    structs = getattr(ast, "structs", None)
    if structs is None or any(getattr(s, "name", None) == "RayDesc" for s in structs):
        return
    if not _uses_type(ast, "RayDesc"):
        return
    float_type = PrimitiveType("float")
    members = [
        StructMemberNode(
            field, VectorType(float_type, 3) if field_type == "vec3" else float_type
        )
        for field, field_type in _RAY_DESC_FIELDS
    ]
    structs.insert(0, StructNode("RayDesc", members))


def add_rust_gpu_ray_query_imports(code):
    """``code`` with the spirv-std ray tracing imports its queries and acceleration structures need."""
    names = []
    if _ACCELERATION_STRUCTURE.search(code):
        names.append("AccelerationStructure")
    if _RAY_QUERY_MACRO.search(code):
        names.append("RayQuery")
    anchor = "use spirv_std::spirv;\n"
    if not names or anchor not in code:
        return code
    if len(names) == 1:
        line = f"use spirv_std::ray_tracing::{names[0]};\n"
    else:
        line = f"use spirv_std::ray_tracing::{{{', '.join(names)}}};\n"
    if "ray_query!" in code and "#![feature(asm_experimental_arch)]" not in code:
        # ray_query! declares the query with SPIR-V inline assembly, an experimental asm! architecture.
        code = code.replace(
            "#![no_std]\n", "#![no_std]\n#![feature(asm_experimental_arch)]\n", 1
        )
    return code.replace(anchor, anchor + line, 1)
