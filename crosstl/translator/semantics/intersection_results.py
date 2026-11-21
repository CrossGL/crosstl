"""Intersection results returned Metal's way, lowered to the hit reports of the other ray tracing targets.

CrossGL carries both ways an intersection stage can hand back a hit on a custom (bounding-box) primitive:

- Metal's: the entry returns a structure whose ``@accept_intersection`` ``bool`` member says whether the ray hits
  the primitive and whose ``@distance`` ``float`` member says where. Its inputs are entry parameters marked
  ``@origin``, ``@min_distance`` and the other intersection function input attributes (Metal Shading Language
  Specification, section 5.2.3.7, and the ``[[intersection(bounding_box)]]`` function attribute of section 5.1.6);
- DXR's and GL_EXT_ray_tracing's: a ``void`` entry calls ``ReportHit(t, kind, attributes)`` or
  ``reportIntersectionEXT(t, kind)``, CrossGL's ``ReportHit`` with three or two arguments, and reads its inputs
  through ``RayTMin()``, ``ObjectRayOrigin()`` and the other ray state intrinsics.

Only Metal output writes the first form. For every other target :func:`lower_intersection_results` rewrites it into
the second. HLSL output had returned the structure from the ``[shader("intersection")]`` entry, which DXC rejects;
GLSL output had written the returned value as an expression statement, so the shader reported no hit at all; and
SPIR-V output had stored it in an ``Output`` variable, which spirv-val rejects. The rewrite:

- replaces each ``return value;`` of the entry with a local copy of ``value``, then ``if (accept)
  ReportHit(distance, 0u);`` and ``return;``. The hit kind is 0, the first user-defined kind, since Metal has none.
  For HLSL and Slang, whose ``ReportHit`` takes an attribute structure (DXR, "ReportHit"), the call passes an
  empty structure: Metal hands no attributes to the hit, and nothing in the hit group reads them. A hit outside
  ``[RayTMin(), RayTCurrent()]`` is ignored by ``ReportHit`` as Metal ignores an accepted distance outside the
  ray's interval;
- makes the entry ``void``, drops the function attributes that only select Metal's intersection function kind
  (``@bounding_box``, ``@intersection(...)``, ``@instancing``, ...) and the role attributes of the result members,
  which the other targets would print as semantics;
- replaces each input parameter with a local initialized from the intrinsic that gives the same value:
  ``@origin`` and ``@direction`` (object space) from ``ObjectRayOrigin()`` and ``ObjectRayDirection()``,
  ``@world_space_origin`` and ``@world_space_direction`` from ``WorldRayOrigin()`` and ``WorldRayDirection()``,
  ``@min_distance`` from ``RayTMin()``, ``@max_distance`` (the current closest distance) from ``RayTCurrent()``,
  ``@primitive_id``, ``@geometry_id``, ``@instance_id`` and ``@user_instance_id`` from ``PrimitiveIndex()``,
  ``GeometryIndex()``, ``InstanceIndex()`` and ``InstanceID()``, and ``@object_to_world_transform`` and
  ``@world_to_object_transform`` (CrossGL ``mat4x3``, MSL ``float4x3``) from ``ObjectToWorld3x4()`` and
  ``WorldToObject3x4()``. A later pass spells those intrinsics for GLSL and SPIR-V (:mod:`.dxr_ray_interfaces`).

Refused with :class:`IntersectionResultUnsupportedError`, because the other targets have no counterpart:

- a triangle intersection function (``@triangle``, or a ``bool`` result): Direct3D and Vulkan run no intersection
  shader for triangle geometry, and an any-hit shader decides there whether a hit counts;
- a ``@continue_search`` result member: an intersection shader cannot end the search, only an any-hit shader can
  (``AcceptHitAndEndSearch``, ``terminateRayEXT``);
- a result structure without a ``bool`` ``@accept_intersection`` or a ``float`` ``@distance`` member, or with a
  member that has neither role;
- any other entry parameter, such as a ``@payload``: an intersection shader receives no payload and no buffers;
- an input parameter whose type is not the type of its intrinsic, and a program that defines a function with the
  name of one of the intrinsics, which would be called instead.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .compute_entry_values import _unique
from .dxr_ray_interfaces import defined_function_names
from .hlsl_ray_interfaces import _functions, _stages, _transform, _walk
from .hub_types import type_from_name, type_from_node

__all__ = [
    "IntersectionResultUnsupportedError",
    "KEEPING_TARGETS",
    "lower_intersection_results",
]

# Targets whose intersection functions return the result structure themselves.
KEEPING_TARGETS = frozenset({"metal", "cgl", "crossgl"})
# Targets whose ReportHit takes an attribute structure as its third argument.
_ATTRIBUTE_TARGETS = frozenset(
    {"directx", "hlsl", "dx11", "dx12", "d3d11", "d3d12", "slang"}
)
# Result member attribute, letters and digits only -> role.
_ROLES = {
    "acceptintersection": "accept",
    "acceptintersectionext": "accept",
    "distance": "distance",
    "hitdistance": "distance",
    "intersectiondistance": "distance",
    "continuesearch": "continue",
}
# Input parameter attribute -> (intrinsic that gives its value, the CrossGL type of that value).
_INPUTS = {
    "origin": ("ObjectRayOrigin", "vec3"),
    "direction": ("ObjectRayDirection", "vec3"),
    "world_space_origin": ("WorldRayOrigin", "vec3"),
    "world_space_direction": ("WorldRayDirection", "vec3"),
    "min_distance": ("RayTMin", "float"),
    "max_distance": ("RayTCurrent", "float"),
    "primitive_id": ("PrimitiveIndex", "uint"),
    "geometry_id": ("GeometryIndex", "uint"),
    "instance_id": ("InstanceIndex", "uint"),
    "user_instance_id": ("InstanceID", "uint"),
    "object_to_world_transform": ("ObjectToWorld3x4", "mat4x3"),
    "world_to_object_transform": ("WorldToObject3x4", "mat4x3"),
}
_BOUNDING_BOX = frozenset({"bounding_box", "boundingbox"})
_TRIANGLE = frozenset({"triangle", "triangles"})
_KIND_ATTRIBUTES = frozenset({"intersection", "primitive_type"})
# Function attributes that only tell Metal which inputs its intersection function takes.
_METAL_ONLY_ATTRIBUTES = frozenset(
    {
        "instancing",
        "triangle_data",
        "world_space_data",
        "extended_limits",
        "primitive_motion",
        "instance_motion",
        "max_levels",
    }
)


class IntersectionResultUnsupportedError(TranslationDiagnostic):
    """A Metal-style intersection function the target's intersection shaders cannot express."""

    code = CODE_PREFIX + "intersection-result-unsupported"

    def __init__(self, entry, detail, target):
        super().__init__(
            f"ray_intersection entry '{entry}': {detail}; the target's intersection shaders report hits "
            "with ReportHit and have no counterpart",
            target=target,
        )


def _attribute_name(attribute) -> str:
    return str(getattr(attribute, "name", attribute) or "").strip().lower()


def _attribute_text(argument) -> str:
    value = getattr(argument, "name", None) or getattr(argument, "value", argument)
    return str(value or "").strip().strip("\"'").lower().replace("-", "_")


def _primitive_kind(entry, returns_bool) -> str:
    for attribute in getattr(entry, "attributes", None) or []:
        name = _attribute_name(attribute)
        if name in _BOUNDING_BOX:
            return "bounding_box"
        if name in _TRIANGLE:
            return "triangle"
        arguments = getattr(attribute, "arguments", None) or []
        if name in _KIND_ATTRIBUTES and arguments:
            requested = _attribute_text(arguments[0])
            if requested in _BOUNDING_BOX:
                return "bounding_box"
            if requested in _TRIANGLE:
                return "triangle"
            return requested
    return "triangle" if returns_bool else "bounding_box"


def _is_void(type_node) -> bool:
    if type_node is None:
        return True
    name = getattr(type_node, "name", type_node)
    return isinstance(name, str) and name.strip() == "void"


def _same_type(type_node, spelling) -> bool:
    if spelling == "mat4x3":
        if isinstance(type_node, A.MatrixType):
            element = getattr(type_node.element_type, "name", type_node.element_type)
            return element == "float" and (type_node.rows, type_node.cols) == (3, 4)
        return getattr(type_node, "name", None) == "mat4x3"
    expected = type_from_name(spelling)
    return expected is not None and type_from_node(type_node) == expected


def _struct(program, stage, type_node):
    if not isinstance(type_node, A.NamedType) or type_node.generic_args:
        return None
    for struct in [
        *(getattr(program, "structs", None) or []),
        *(getattr(stage, "local_structs", None) or []),
    ]:
        if getattr(struct, "name", None) == type_node.name and isinstance(
            struct, A.StructNode
        ):
            return struct
    return None


class _Entry:
    def __init__(self, program, stage, target):
        self.program = program
        self.stage = stage
        self.entry = stage.entry_point
        self.target = target
        self.name = getattr(self.entry, "name", "main")

    def refuse(self, detail):
        raise IntersectionResultUnsupportedError(self.name, detail, self.target)

    def roles(self, struct):
        roles = {}
        for member in struct.members or []:
            role = None
            for attribute in getattr(member, "attributes", None) or []:
                compact = "".join(
                    ch for ch in _attribute_name(attribute) if ch.isalnum()
                )
                role = _ROLES.get(compact, role)
            if role is None:
                self.refuse(
                    f"result member '{member.name}' of '{struct.name}' is neither "
                    "@accept_intersection nor @distance"
                )
            if role == "continue":
                self.refuse(
                    f"result member '{member.name}' is @continue_search; only an any-hit shader can end "
                    "the search (AcceptHitAndEndSearch, terminateRayEXT)"
                )
            if role in roles:
                self.refuse(f"result structure '{struct.name}' has two {role} members")
            roles[role] = member
        for role, spelling, attribute in (
            ("accept", "bool", "@accept_intersection"),
            ("distance", "float", "@distance"),
        ):
            member = roles.get(role)
            if member is None or not _same_type(member.member_type, spelling):
                self.refuse(
                    f"result structure '{struct.name}' needs a {spelling} {attribute} member"
                )
        return roles

    def inputs(self):
        """Locals that take the place of the entry's input parameters, in parameter order."""
        defined = defined_function_names(self.program)
        locals_ = []
        for parameter in getattr(self.entry, "parameters", None) or []:
            names = [
                _attribute_name(attribute)
                for attribute in getattr(parameter, "attributes", None) or []
            ]
            known = [name for name in names if name in _INPUTS]
            if len(known) != 1 or len(names) != 1:
                self.refuse(
                    f"parameter '{parameter.name}' is not one intersection input "
                    f"({', '.join('@' + name for name in sorted(_INPUTS))}); an intersection shader "
                    "receives no payload, buffers or other arguments"
                )
            intrinsic, spelling = _INPUTS[known[0]]
            if not _same_type(parameter.param_type, spelling):
                self.refuse(
                    f"parameter '{parameter.name}' @{known[0]} must be a {spelling}, the type of {intrinsic}()"
                )
            if intrinsic in defined:
                self.refuse(
                    f"the program defines a function named {intrinsic}, which a call reading "
                    f"@{known[0]} would call instead"
                )
            locals_.append(
                A.VariableNode(
                    parameter.name,
                    parameter.param_type,
                    A.FunctionCallNode(A.IdentifierNode(intrinsic), []),
                )
            )
        return locals_

    def taken_names(self):
        taken = {
            getattr(struct, "name", None)
            for struct in getattr(self.program, "structs", None) or []
        }
        taken.update(function.name for function in _functions(self.program))
        for variable in [
            *(getattr(self.program, "global_variables", None) or []),
            *(getattr(self.stage, "local_variables", None) or []),
        ]:
            taken.add(getattr(variable, "name", None))
        for node in _walk(self.entry):
            for field in ("name", "identifier"):
                value = getattr(node, field, None)
                if isinstance(value, str):
                    taken.add(value)
        return taken

    def report(self, roles, result_name, attributes_type, taken):
        def member(role):
            return A.MemberAccessNode(A.IdentifierNode(result_name), roles[role].name)

        arguments = [member("distance"), A.LiteralNode(0, A.PrimitiveType("uint"))]
        statements = []
        if attributes_type is not None:
            attributes_name = _unique("hit_attributes", taken)
            statements.append(A.VariableNode(attributes_name, attributes_type))
            arguments.append(A.IdentifierNode(attributes_name))
        statements.append(
            A.ExpressionStatementNode(A.RayTracingOpNode("ReportHit", arguments))
        )
        return A.IfNode(member("accept"), A.BlockNode(statements))

    def lower(self):
        entry = self.entry
        returns_bool = getattr(entry.return_type, "name", None) == "bool"
        kind = _primitive_kind(entry, returns_bool)
        if kind == "triangle":
            self.refuse(
                "it is a triangle intersection function; Direct3D and Vulkan run no intersection shader "
                "for triangle geometry, where an any-hit shader decides whether a hit counts"
            )
        if kind != "bounding_box":
            self.refuse(f"intersection function kind '{kind}' is not a bounding box")
        struct = _struct(self.program, self.stage, entry.return_type)
        if struct is None:
            self.refuse(
                "a bounding-box intersection function must return a result structure"
            )
        roles = self.roles(struct)
        inputs = self.inputs()
        taken = self.taken_names()
        attributes_type = None
        if self.target in _ATTRIBUTE_TARGETS:
            attributes_name = _unique("IntersectionAttributes", taken)
            if getattr(self.program, "structs", None) is None:
                self.program.structs = []
            self.program.structs.append(A.StructNode(attributes_name, []))
            attributes_type = A.NamedType(attributes_name)
        return_type = entry.return_type

        def visit(node):
            if not isinstance(node, A.ReturnNode) or node.value is None:
                return None
            result_name = _unique("intersection_result", taken)
            return A.BlockNode(
                [
                    A.VariableNode(result_name, return_type, node.value),
                    self.report(roles, result_name, attributes_type, taken),
                    A.ReturnNode(None),
                ]
            )

        _transform(entry.body, visit)
        entry.body.statements[:0] = inputs
        entry.parameters = []
        entry.return_type = A.PrimitiveType("void")
        entry.attributes = [
            attribute
            for attribute in entry.attributes or []
            if _attribute_name(attribute)
            not in _BOUNDING_BOX | _KIND_ATTRIBUTES | _METAL_ONLY_ATTRIBUTES
        ]
        return list(roles.values())


def lower_intersection_results(program, target) -> int:
    """Rewrite the result-returning intersection entries of ``program`` into hit reports for ``target``.

    Returns how many entries changed. Metal and CrossGL output keep the result form."""
    target = str(target or "").lower()
    if target in KEEPING_TARGETS or not isinstance(program, A.ShaderNode):
        return 0
    changed, members = 0, []
    for stage in _stages(program):
        entry = getattr(stage, "entry_point", None)
        if (
            getattr(stage, "stage", None) != A.ShaderStage.RAY_INTERSECTION
            or entry is None
            or _is_void(getattr(entry, "return_type", None))
            or not isinstance(getattr(entry, "body", None), A.BlockNode)
        ):
            continue
        members.extend(_Entry(program, stage, target).lower())
        changed += 1
    # Only after every entry is lowered, since several entries may return the same structure.
    for member in members:
        member.attributes = []
    return changed
