"""Metal post-tessellation vertex functions for CrossGL tessellation evaluation (domain) stages.

Metal has no domain stage. Its fixed-function tessellator feeds a post-tessellation vertex function, marked
``[[patch(triangle, N)]]`` or ``[[patch(quad, N)]]``, that reads the patch's control points as a
``patch_control_point<T>`` member of its ``[[stage_in]]`` structure, the per-patch values as the other
``[[attribute(n)]]`` members of that structure, the position in the patch as ``[[position_in_patch]]`` and the patch's
index as ``[[patch_id]]`` (Metal Shading Language Specification 4.1, section 5.1.5, "Post-Tessellation Vertex
Functions", and section 5.2.3.2, "Post-Tessellation Vertex Function Input Attributes"). A CrossGL tessellation
evaluation entry, as the HLSL importer writes a domain shader::

    tessellation_evaluation {
        DSOut main(PatchOut pc, vec3 uvw @ SV_DomainLocation, OutputPatch<HSOut, 3> patch) @domain(tri) { ... }
    }

becomes such a function; an entry in the GLSL form, which reads per-vertex input arrays, ``patch in`` variables and
``gl_TessCoord`` and has its domain in a stage ``layout(...) in``, is first rewritten into this form by
:mod:`.metal_glsl_tessellation`. :func:`lower_metal_post_tessellation` takes the patch, the patch-constant structure, the
domain location (``SV_DomainLocation`` or ``gl_TessCoord``) and the patch index (``SV_PrimitiveID`` or
``gl_PrimitiveID``) out of the entry's parameters and records them in a :class:`PostTessellationPlan` annotation.
The generator's vertex path writes an entry that has a plan
(:meth:`MetalPostTessellationMixin.metal_post_tessellation_interface`):

* ``[[patch(triangle, N)]]`` or ``[[patch(quad, N)]]`` before ``vertex``, N being the ``OutputPatch`` size, or no N
  when the draw sets it (a GLSL-form entry without a control stage). Metal has no isoline patches, so an
  ``@domain(isoline)`` entry is refused;
* a ``<entry>_PatchPoint`` structure that holds the control-point members as ``[[attribute(n)]]`` and a
  ``<entry>_PatchIn`` stage-input structure that holds ``patch_control_point<<entry>_PatchPoint>`` and the
  patch-constant members other than the tessellation factors. The control-point members take attributes 0, 1, ...
  in declaration order and the patch-constant members continue after them, the order in which
  :mod:`..semantics.tessellation_patches` links a hull stage to a domain stage;
* the body starts by copying the control points into a local array of the control-point structure under the patch's
  name, and the patch-constant members into a local of the patch-constant structure under its name, so the body and
  the helpers it passes them to are unchanged. A helper that takes an ``InputPatch`` or ``OutputPatch`` takes an
  array of its control points.

The tessellation factors reach the Metal tessellator through a buffer the host binds
(``setTessellationFactorBuffer``), not through the post-tessellation vertex function, so an entry that reads a
tessellation-factor member, or uses the whole patch-constant structure while it has such members, is refused.
``@partitioning`` and ``@outputtopology`` are dropped: Metal sets the partition mode and the winding order in the
render pipeline descriptor (``tessellationPartitionMode``, ``tessellationOutputWindingOrder``), not in the shader.
A control-point or patch-constant structure that no other entry takes or returns loses its member semantics, which
have no Metal attribute outside a stage interface. The input AST is not modified.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass

from ..ast import (
    ArrayType,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    NamedType,
    PrimitiveType,
    StructNode,
)
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from ..stage_utils import STAGE_QUALIFIER_NAMES, normalize_stage_name
from .stage_utils import collect_stage_entry_records, stage_layout_entries

# An entry whose control-point count the draw sets (:mod:`.metal_glsl_tessellation`).
OPEN_PATCH_ANNOTATION = "metal_post_tessellation_open_patch"

__all__ = [
    "MetalPostTessellationError",
    "MetalPostTessellationMixin",
    "PostTessellationPlan",
    "lower_metal_post_tessellation",
    "post_tessellation_plan",
]

PLAN_ANNOTATION = "metal_post_tessellation"
_STAGE = "tessellation_evaluation"
_PATCH_TYPES = {"tri": "triangle", "triangle": "triangle", "quad": "quad"}
_DOMAIN_LOCATIONS = frozenset({"sv_domainlocation", "gl_tesscoord"})
_PATCH_IDS = frozenset({"sv_primitiveid", "gl_primitiveid"})
_FACTORS = frozenset(
    {"sv_tessfactor", "sv_insidetessfactor", "gl_tesslevelouter", "gl_tesslevelinner"}
)
_PATCH_OBJECTS = frozenset({"InputPatch", "OutputPatch"})
_PIPELINE_ATTRIBUTES = frozenset({"partitioning", "outputtopology"})
_GLSL_DOMAINS = frozenset({"triangles", "quads", "isolines"})
# A vertex attribute is a scalar or vector of these (MSL 4.1 section 5.2.3.1, "Vertex Function Input Attributes").
_ATTRIBUTE_TYPE = re.compile(r"(?:float|half|int|uint|short|ushort|char|uchar)[234]?")


class MetalPostTessellationError(TranslationDiagnostic):
    """A tessellation evaluation entry that has no Metal post-tessellation vertex function."""

    code = CODE_PREFIX + "metal-post-tessellation-unsupported"

    def __init__(self, entry: str, reason: str):
        self.entry = entry
        super().__init__(
            f"tessellation evaluation entry '{entry}': {reason} (Metal Shading Language Specification 5.1.5)",
            target="metal",
        )


@dataclass(frozen=True)
class PatchValue:
    """A member of the control-point or patch-constant structure the entry reads through ``[[stage_in]]``."""

    name: str
    type_node: object
    location: int | None = None


@dataclass(frozen=True)
class PostTessellationPlan:
    """What :func:`lower_metal_post_tessellation` took out of one tessellation evaluation entry."""

    patch_type: str
    count: int | None
    # The [[patch]] attribute's control-point count; None when the draw sets it, and ``count`` bounds the copy.
    declared_count: int | None
    points: str | None
    point_type: object
    point_members: tuple[PatchValue, ...]
    constants: str | None
    constants_type: object
    constant_members: tuple[PatchValue, ...]
    location: tuple[str, object] | None
    patch_id: tuple[str, object] | None


def post_tessellation_plan(func):
    """The :class:`PostTessellationPlan` of a lowered entry, else None."""
    get_annotation = getattr(func, "get_annotation", None)
    return get_annotation(PLAN_ANNOTATION) if callable(get_annotation) else None


def _semantic(node, codegen):
    semantic = codegen.semantic_from_node(node)
    return str(semantic).lower() if semantic else None


def _attribute_name(attribute):
    return str(getattr(attribute, "name", attribute) or "").lower()


def _attribute_text(attribute):
    arguments = getattr(attribute, "arguments", None) or []
    if len(arguments) != 1:
        return None
    argument = arguments[0]
    text = getattr(argument, "name", None)
    if text is None:
        text = getattr(argument, "value", None)
    return None if text is None else str(text).strip("\"'").lower()


def _literal_int(node):
    if isinstance(node, LiteralNode):
        node = node.value
    if isinstance(node, int) and not isinstance(node, bool):
        return node
    if isinstance(node, str) and node.isdigit():
        return int(node)
    return None


def _explicit_location(member):
    """The ``n`` of a member's ``@location(n)``, else None."""
    for attribute in getattr(member, "attributes", None) or []:
        if _attribute_name(attribute) == "location":
            arguments = getattr(attribute, "arguments", None) or []
            return _literal_int(arguments[0]) if len(arguments) == 1 else None
    return None


def _attribute_indices(entry, values):
    """``[[attribute(n)]]`` of each value: its own ``@location``, else the next index no value claims."""
    claimed = [value.location for value in values if value.location is not None]
    if len(set(claimed)) != len(claimed):
        raise MetalPostTessellationError(
            entry, "two control-point or patch-constant members claim the same location"
        )
    taken, indices, candidate = set(claimed), [], 0
    for value in values:
        if value.location is not None:
            indices.append(value.location)
            continue
        while candidate in taken:
            candidate += 1
        taken.add(candidate)
        indices.append(candidate)
    return indices


def _patch_object(type_node):
    """``(control-point type, count node)`` of an ``InputPatch``/``OutputPatch`` type, else None."""
    if not isinstance(type_node, NamedType) or type_node.name not in _PATCH_OBJECTS:
        return None
    arguments = list(getattr(type_node, "generic_args", None) or [])
    return tuple(arguments) if len(arguments) == 2 else None


def _structs(ast):
    containers = [getattr(ast, "structs", None)]
    containers.extend(
        getattr(stage, "local_structs", None)
        for stage in (getattr(ast, "stages", None) or {}).values()
    )
    return {
        struct.name: struct
        for container in containers
        for struct in container or []
        if isinstance(struct, StructNode) and getattr(struct, "name", None)
    }


def _entries(ast):
    """``(stage, entry point)`` of each tessellation evaluation stage of the program."""
    return [
        (stage, stage.entry_point)
        for stage_type, stage in (getattr(ast, "stages", None) or {}).items()
        if normalize_stage_name(stage_type) == _STAGE
        and getattr(stage, "entry_point", None) is not None
    ]


def _glsl_form(stage, func):
    """Whether the stage gives its domain in a GLSL ``layout(...) in`` and the entry no ``@domain``."""
    if any(
        _attribute_name(attribute) == "domain"
        for attribute in getattr(func, "attributes", None) or []
    ):
        return False
    return any(
        str(getattr(entry, "name", "")).lower() in _GLSL_DOMAINS
        for entry in stage_layout_entries(stage, "in")
    )


class _EntryLowering:
    """Builds the plan of one entry and removes what it records from the entry."""

    def __init__(self, func, structs, codegen):
        self.func = func
        self.structs = structs
        self.codegen = codegen
        self.name = getattr(func, "name", None) or "main"

    def used_names(self):
        body = getattr(self.func, "body", None)
        return {
            node.name
            for node in (body.walk() if body is not None else ())
            if isinstance(node, IdentifierNode)
        }

    def refuse(self, reason):
        raise MetalPostTessellationError(self.name, reason)

    def patch_type(self):
        kept, patch_type = [], None
        for attribute in getattr(self.func, "attributes", None) or []:
            name = _attribute_name(attribute)
            if name == "domain":
                domain = _attribute_text(attribute)
                patch_type = _PATCH_TYPES.get(domain)
                if patch_type is None:
                    self.refuse(
                        f"Metal tessellates only triangle and quad patches, not the '{domain}' domain"
                    )
            elif name not in _PIPELINE_ATTRIBUTES:
                kept.append(attribute)
        if patch_type is None:
            self.refuse(
                "Metal's [[patch]] attribute needs the patch type, and the entry has no @domain"
            )
        self.func.attributes = kept
        return patch_type

    def members(self, struct_name, *, drop_factors):
        struct = self.structs.get(struct_name)
        if struct is None:
            self.refuse(f"structure '{struct_name}' is not declared")
        values, factors = [], set()
        for member in getattr(struct, "members", None) or []:
            if drop_factors and _semantic(member, self.codegen) in _FACTORS:
                factors.add(member.name)
                continue
            values.append(
                PatchValue(
                    member.name,
                    getattr(member, "member_type", None),
                    _explicit_location(member),
                )
            )
        return tuple(values), factors

    def classify(self):
        """``(points, constants, location, patch_id)`` parameters; the others stay entry parameters."""
        points = constants = location = patch_id = None
        kept = []
        for parameter in getattr(self.func, "parameters", None) or []:
            semantic = _semantic(parameter, self.codegen)
            parameter_type = getattr(parameter, "param_type", None)
            if _patch_object(parameter_type) is not None:
                if points is not None:
                    self.refuse("it takes more than one control-point patch")
                points = parameter
            elif semantic in _DOMAIN_LOCATIONS:
                location = parameter
            elif semantic in _PATCH_IDS:
                patch_id = parameter
            elif (
                semantic is None
                and isinstance(parameter_type, NamedType)
                and parameter_type.name in self.structs
            ):
                if constants is not None:
                    self.refuse("it takes more than one patch-constant structure")
                constants = parameter
            elif semantic is not None:
                self.refuse(
                    f"parameter '{parameter.name}' carries '{semantic}', which a Metal post-tessellation "
                    "vertex function does not read"
                )
            else:
                kept.append(parameter)
        self.func.parameters = kept
        return points, constants, location, patch_id

    def check_constant_reads(self, constants, factors):
        """Refuse reads of the tessellation factors, which the local patch-constant copy does not hold."""
        if not factors:
            return
        body = getattr(self.func, "body", None)
        member_objects = set()
        for node in body.walk() if body is not None else ():
            if isinstance(node, MemberAccessNode) and _is_name(
                node.object_expr, constants
            ):
                member_objects.add(id(node.object_expr))
                if node.member in factors:
                    self.refuse(
                        f"it reads the tessellation factor '{constants}.{node.member}', which Metal passes "
                        "to the tessellator in a host-bound buffer, not to the post-tessellation function"
                    )
        for node in body.walk() if body is not None else ():
            if _is_name(node, constants) and id(node) not in member_objects:
                self.refuse(
                    f"it uses the patch-constant structure '{constants}' as a whole, and Metal does not pass "
                    "its tessellation factors to the post-tessellation function"
                )

    def plan(self):
        patch_type = self.patch_type()
        points, constants, location, patch_id = self.classify()
        count = point_type = None
        point_members = ()
        if points is not None:
            point_type, count_node = _patch_object(points.param_type)
            count = _literal_int(count_node)
            if count is None:
                count = self.codegen.literal_int_value(
                    count_node, self.codegen.literal_int_constants
                )
            if count is None or count < 1:
                self.refuse(
                    "its control-point patch size is not a positive integer constant"
                )
            point_members, _ = self.members(
                getattr(point_type, "name", None), drop_factors=False
            )
        constant_members = ()
        if constants is not None:
            constant_members, factors = self.members(
                constants.param_type.name, drop_factors=True
            )
            self.check_constant_reads(constants.name, factors)
        # A patch or patch-constant parameter the body never names needs no local copy.
        used = self.used_names()
        if constants is not None and constants.name not in used:
            constants, constant_members = None, ()
        if points is not None and points.name not in used:
            points = None
        open_patch = self.func.get_annotation(OPEN_PATCH_ANNOTATION, False)
        return PostTessellationPlan(
            patch_type=patch_type,
            count=count,
            declared_count=None if open_patch else count,
            points=getattr(points, "name", None),
            point_type=point_type,
            point_members=point_members,
            constants=getattr(constants, "name", None),
            constants_type=getattr(constants, "param_type", None),
            constant_members=constant_members,
            location=None if location is None else (location.name, location.param_type),
            patch_id=None if patch_id is None else (patch_id.name, patch_id.param_type),
        )


def _is_name(node, name):
    return isinstance(node, IdentifierNode) and node.name == name


def _interface_struct_names(ast, lowered_ids):
    """Structures an entry other than the lowered ones takes or returns."""
    names = set()
    for _id, _stage, function in collect_stage_entry_records(
        ast, None, STAGE_QUALIFIER_NAMES
    ):
        if id(function) in lowered_ids:
            continue
        types = [getattr(function, "return_type", None)]
        types.extend(
            getattr(parameter, "param_type", None)
            for parameter in getattr(function, "parameters", None) or []
        )
        for type_node in types:
            patch = _patch_object(type_node)
            if patch is not None:
                type_node = patch[0]
            if isinstance(type_node, NamedType):
                names.add(type_node.name)
    return names


def _strip_plain_semantics(structs, names, codegen):
    for name in sorted(names):
        for member in getattr(structs.get(name), "members", None) or []:
            semantic = codegen.semantic_from_node(member)
            if semantic is None:
                continue
            member.attributes = [
                attribute
                for attribute in getattr(member, "attributes", None) or []
                if getattr(attribute, "name", None) != semantic
            ]


def _patch_parameters_as_arrays(ast, lowered_ids):
    """Give helpers that take a patch an array of its control points, which the lowered entries pass."""
    functions = list(getattr(ast, "functions", None) or [])
    for stage in (getattr(ast, "stages", None) or {}).values():
        functions.extend(getattr(stage, "local_functions", None) or [])
    for function in functions:
        if id(function) in lowered_ids:
            continue
        for parameter in getattr(function, "parameters", None) or []:
            patch = _patch_object(getattr(parameter, "param_type", None))
            if patch is not None:
                parameter.param_type = ArrayType(patch[0], patch[1])


def lower_metal_post_tessellation(ast, codegen):
    """Return ``ast`` with each tessellation evaluation entry planned as a Metal post-tessellation function.

    ``codegen`` is the Metal emitter, whose ``semantic_from_node`` and ``literal_int_value`` read semantics and
    patch sizes. Returns ``ast`` itself when it has no such entry; raises :class:`MetalPostTessellationError` for an
    entry Metal cannot write.
    """
    if not _entries(ast):
        return ast
    ast = deepcopy(ast)
    structs = _structs(ast)
    lowered_ids, plain = set(), set()
    for stage, function in _entries(ast):
        if _glsl_form(stage, function):
            from .metal_glsl_tessellation import rewrite_glsl_evaluation_entry

            for struct in rewrite_glsl_evaluation_entry(ast, stage, function):
                structs[struct.name] = struct
        plan = _EntryLowering(function, structs, codegen).plan()
        function.add_annotation(PLAN_ANNOTATION, plan)
        lowered_ids.add(id(function))
        for type_node in (plan.point_type, plan.constants_type):
            if isinstance(type_node, NamedType):
                plain.add(type_node.name)
    _strip_plain_semantics(
        structs, plain - _interface_struct_names(ast, lowered_ids), codegen
    )
    _patch_parameters_as_arrays(ast, lowered_ids)
    return ast


class MetalPostTessellationMixin:
    """Writes the interface of an entry :func:`lower_metal_post_tessellation` planned."""

    def unique_metal_post_tessellation_struct_name(self, base_name):
        used = set(self.structs_by_name)
        used.update(self.generated_return_wrapper_struct_names)
        used.update(getattr(self, "metal_generated_struct_names", set()))
        candidate, suffix = base_name, 2
        while candidate in used:
            candidate = f"{base_name}_{suffix}"
            suffix += 1
        self.generated_return_wrapper_struct_names.add(candidate)
        return candidate

    def metal_post_tessellation_attribute_type(self, function_name, owner, value):
        mapped = self.map_type(value.type_node)
        if not _ATTRIBUTE_TYPE.fullmatch(str(mapped)):
            raise MetalPostTessellationError(
                function_name,
                f"{owner} member '{value.name}' has type '{mapped}', and a Metal patch attribute is a scalar "
                "or vector of float, half or an integer type of 32 bits or fewer",
            )
        return mapped

    def metal_post_tessellation_interface(self, func, function_name, reserved_names):
        """``(structures, parameters, prefix, body preamble)`` of the planned entry ``func``."""
        plan = post_tessellation_plan(func)
        reserved = set(reserved_names) | self.metal_function_local_variable_names(func)
        structures, parameters, preamble = "", [], []
        stage_in_members, stage_in_name = [], None
        values = (plan.point_members if plan.points is not None else ()) + tuple(
            plan.constant_members
        )
        attributes = iter(_attribute_indices(function_name, values))
        if plan.points is not None:
            point_struct = self.unique_metal_post_tessellation_struct_name(
                f"{function_name}_PatchPoint"
            )
            structures += f"struct {point_struct} {{\n"
            for value in plan.point_members:
                mapped = self.metal_post_tessellation_attribute_type(
                    function_name, "control-point", value
                )
                structures += (
                    f"    {mapped} {value.name} [[attribute({next(attributes)})]];\n"
                )
            structures += "};\n"
            stage_in_members.append(f"patch_control_point<{point_struct}> points;")
        for value in plan.constant_members:
            mapped = self.metal_post_tessellation_attribute_type(
                function_name, "patch-constant", value
            )
            stage_in_members.append(
                f"{mapped} {value.name} [[attribute({next(attributes)})]];"
            )
        if stage_in_members:
            stage_in = self.unique_metal_post_tessellation_struct_name(
                f"{function_name}_PatchIn"
            )
            structures += f"struct {stage_in} {{\n"
            structures += "".join(f"    {member}\n" for member in stage_in_members)
            structures += "};\n"
            stage_in_name = self.unique_metal_generated_name("_crossglPatch", reserved)
            reserved.add(stage_in_name)
            parameters.append(f"{stage_in} {stage_in_name} [[stage_in]]")
        if plan.location is not None:
            name, type_node = plan.location
            parameters.append(
                f"{self.map_type(type_node)} {name} [[position_in_patch]]"
            )
            self.local_variable_types[name] = self.type_name_string(type_node)
        if plan.patch_id is not None:
            name, type_node = plan.patch_id
            parameters.append(f"{self.map_type(type_node)} {name} [[patch_id]]")
            self.local_variable_types[name] = self.type_name_string(type_node)
        if plan.points is not None:
            point_type = self.map_type(plan.point_type)
            index = self.unique_metal_generated_name("_crossglPoint", reserved)
            preamble.append(f"{point_type} {plan.points}[{plan.count}];")
            bound = f"{plan.count}u"
            if plan.declared_count is None:
                bound = f"min(uint({stage_in_name}.points.size()), {bound})"
            preamble.append(f"for (uint {index} = 0u; {index} < {bound}; ++{index}) {{")
            preamble.extend(
                f"    {plan.points}[{index}].{value.name} = "
                f"{stage_in_name}.points[{index}].{value.name};"
                for value in plan.point_members
            )
            preamble.append("}")
            self.local_variable_types[plan.points] = self.type_name_string(
                ArrayType(
                    plan.point_type, LiteralNode(plan.count, PrimitiveType("int"))
                )
            )
        if plan.constants is not None:
            preamble.append(f"{self.map_type(plan.constants_type)} {plan.constants};")
            preamble.extend(
                f"{plan.constants}.{value.name} = {stage_in_name}.{value.name};"
                for value in plan.constant_members
            )
            self.local_variable_types[plan.constants] = self.type_name_string(
                plan.constants_type
            )
        count = "" if plan.declared_count is None else f", {plan.declared_count}"
        prefix = f"[[patch({plan.patch_type}{count})]]\n"
        return structures, parameters, prefix, preamble
