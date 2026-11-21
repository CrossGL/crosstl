"""Intersection stages that return nothing, lowered to the result a Metal intersection function returns.

A GL_EXT_ray_tracing, SPIR-V or DXR intersection shader returns nothing. It reports each candidate hit with
``ReportHit(THit, HitKind, Attributes)`` (``reportIntersectionEXT(hitT, hitKind)``), which accepts the hit and
returns true when ``THit`` lies in the ray's current interval ``[RayTMin(), RayTCurrent()]``; an accepted hit makes
``THit`` the new ``RayTCurrent()``. Without an any-hit shader to ignore a hit, the stage's result is therefore the
closest hit it reports inside that interval (DirectX Raytracing functional specification, "ReportHit";
GL_EXT_ray_tracing, "reportIntersectionEXT"). A Metal bounding-box intersection function instead returns a structure
with a ``bool [[accept_intersection]]`` and a ``float [[distance]]`` member, and receives the ray's interval as its
``[[min_distance]]`` and ``[[max_distance]]`` parameters (Metal Shading Language Specification, sections 5.1.6,
5.2.3.7 and 5.2.3.8). The generator refused such a stage.

:func:`lower_metal_reported_hits` rewrites each void ``ray_intersection`` entry point for the Metal target:

- the entry returns a new structure ``CrossGLIntersectionHit { bool accept @accept_intersection; float distance
  @distance; }`` and is marked ``@bounding_box``: DXR and Vulkan run intersection shaders only for procedural
  (axis-aligned bounding box) geometry;
- it takes ``[[min_distance]]`` and ``[[max_distance]]`` parameters, or reuses the ones it declares;
- its body starts with a local result that rejects the hit at ``[[max_distance]]``, every ``return;`` returns that
  local, and so does the end of the body;
- each ``ReportHit`` call becomes a call of the helper ``crossgl_report_hit(hit, THit, minDistance)``, which accepts
  ``THit`` when ``minDistance <= THit <= hit.distance``, records it as the new distance and returns whether it did,
  as ``ReportHit`` returns.

Metal's intersection result has no hit kind or attribute members (section 5.2.3.8), and ``intersector<>::intersect``
returns the result to the kernel instead of invoking hit shaders (section 6.18). The hit kind and attributes are
dropped when they are plain values; a ``ReportHit`` whose hit kind or attributes have side effects is refused, and
so is the lowering when the program also has an any-hit or closest-hit stage with an ``@hit_attribute`` parameter,
which would expect the reported attributes. A GLSL ``hitAttributeEXT`` variable becomes a local of the entry
points that write it, and is refused when any other function uses it. A ``ReportHit`` outside the entry point and a void entry marked
``@triangle`` keep their existing refusals. The input AST is not modified.
"""

from __future__ import annotations

from copy import deepcopy

from ..ast import (
    AssignmentNode,
    AttributeNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    NamedType,
    ParameterNode,
    PrimitiveType,
    RayQueryOpNode,
    RayTracingOpNode,
    ReturnNode,
    UnaryOpNode,
)
from ..semantics.builtin_type_names import BUILTIN_TYPE_NAMES
from ..semantics.declared_types import declared_type_names
from ..semantics.rewrite import child_slots, get_child, set_child
from .stage_utils import collect_stage_entry_records

__all__ = ["lower_metal_reported_hits"]

INTERSECTION_STAGES = frozenset({"ray_intersection", "intersection"})
HIT_STAGES = frozenset({"ray_any_hit", "ray_closest_hit", "anyhit", "closesthit"})
_PRIMITIVE_ATTRIBUTES = frozenset(
    {
        "bounding_box",
        "boundingbox",
        "triangle",
        "triangles",
        "intersection",
        "primitive_type",
    }
)
_SIDE_EFFECT_NODES = (AssignmentNode, RayQueryOpNode, RayTracingOpNode)
_HIT_ATTRIBUTE_VARIABLES = frozenset(
    {"hitattributeext", "hitattributenv", "hit_attribute"}
)
_RESULT_STRUCT = "CrossGLIntersectionHit"
_HELPER = "crossgl_report_hit"
_HIT_LOCAL = "crossgl_hit"
_MIN_PARAMETER = "crossgl_min_distance"
_MAX_PARAMETER = "crossgl_max_distance"


def _walk(node):
    yield node
    for field, key in list(child_slots(node)):
        yield from _walk(get_child(node, field, key))


def _transform(node, visit):
    for field, key in list(child_slots(node)):
        child = get_child(node, field, key)
        _transform(child, visit)
        replacement = visit(child)
        if replacement is not None and replacement is not child:
            set_child(node, field, key, replacement)


def _attribute_names(node):
    return [
        str(getattr(attribute, "name", "")).strip().lower()
        for attribute in getattr(node, "attributes", None) or []
    ]


def _is_void(type_node):
    if type_node is None:
        return True
    name = getattr(type_node, "name", type_node)
    return str(name).strip() == "void"


def _is_report_hit(node):
    return isinstance(node, RayTracingOpNode) and node.operation == "ReportHit"


def _stage_entries(ast, stages):
    """``(stage, function)`` for each entry point of one of ``stages``; the records list every stage block."""
    return [
        (stage, function)
        for _entry_id, stage, function in collect_stage_entry_records(ast, None, stages)
        if stage in stages
    ]


def _void_intersection_entries(ast):
    entries = []
    seen = set()
    for _stage, function in _stage_entries(ast, INTERSECTION_STAGES):
        entry_id = id(function)
        if entry_id in seen or not _is_void(getattr(function, "return_type", None)):
            continue
        seen.add(entry_id)
        if {"triangle", "triangles"} & set(_attribute_names(function)):
            continue
        entries.append(function)
    return entries


def _program_names(ast):
    names = set()
    for node in _walk(ast):
        name = getattr(node, "name", None)
        if isinstance(name, str):
            names.add(name)
    return names


def _unique(base, taken):
    name = base
    index = 1
    while name in taken:
        name = f"{base}_{index}"
        index += 1
    taken.add(name)
    return name


def _has_side_effects(expression, type_names):
    """Whether evaluating ``expression`` may write state; calls of a type name construct a value."""
    for node in _walk(expression):
        if isinstance(node, FunctionCallNode):
            callee = getattr(node, "function", None)
            if isinstance(callee, IdentifierNode) and callee.name in type_names:
                continue
            if isinstance(callee, (PrimitiveType, NamedType)):
                continue
            return True
        if isinstance(node, _SIDE_EFFECT_NODES):
            return True
        if isinstance(node, UnaryOpNode) and node.operator in {"++", "--"}:
            return True
    return False


def _distance_parameter(function, semantic, default_name, taken):
    for parameter in getattr(function, "parameters", None) or []:
        if semantic in _attribute_names(parameter):
            return parameter.name
    name = _unique(default_name, taken)
    function.parameters.append(
        ParameterNode(
            name,
            PrimitiveType("float"),
            attributes=[AttributeNode(semantic)],
        )
    )
    return name


def _parse_support(struct_name, helper_name, hit_name, max_name):
    """The result structure, the helper and the entry prologue, parsed from CrossGL."""
    from ..lexer import Lexer
    from ..parser import Parser

    source = f"""
shader CrossGLReportedHits {{
    struct {struct_name} {{
        bool accept @ accept_intersection;
        float distance @ distance;
    }}
    bool {helper_name}(inout {struct_name} hit, float t, float minDistance) {{
        if (t >= minDistance && t <= hit.distance) {{
            hit.accept = true;
            hit.distance = t;
            return true;
        }}
        return false;
    }}
    void crossgl_prologue(float {max_name}) {{
        {struct_name} {hit_name};
        {hit_name}.accept = false;
        {hit_name}.distance = {max_name};
    }}
}}
"""
    program = Parser(Lexer(source).get_tokens()).parse()
    struct = program.structs[0]
    helper, prologue = program.functions[0], program.functions[1]
    return struct, helper, list(prologue.body.statements)


def _refusals(ast, entries):
    reasons = []
    type_names = BUILTIN_TYPE_NAMES | declared_type_names(ast)
    for function in entries:
        for node in _walk(function):
            if not _is_report_hit(node):
                continue
            if any(
                _has_side_effects(argument, type_names)
                for argument in node.arguments[1:]
            ):
                reasons.append(
                    "intersection stage: ReportHit hit kind or attributes with side effects: a Metal "
                    "intersection function returns only acceptance and distance, so the generator drops "
                    "them and cannot keep their side effects"
                )
                break
    if not any(
        _is_report_hit(node) for function in entries for node in _walk(function)
    ):
        return reasons
    for stage, function in _stage_entries(ast, HIT_STAGES):
        for parameter in getattr(function, "parameters", None) or []:
            if "hit_attribute" in _attribute_names(parameter):
                reasons.append(
                    f"intersection stage: ReportHit attributes read by the {stage} stage's @hit_attribute "
                    f"parameter '{parameter.name}': a Metal intersection function returns only acceptance and "
                    "distance, and intersector<>::intersect invokes no hit function with attributes "
                    "(MSL sections 5.2.3.8 and 6.18)"
                )
    return reasons


def _lower_entry(function, struct_name, helper_name, hit_name, min_name):
    def visit(node):
        if _is_report_hit(node) and node.arguments:
            return FunctionCallNode(
                IdentifierNode(helper_name),
                [IdentifierNode(hit_name), node.arguments[0], IdentifierNode(min_name)],
            )
        if isinstance(node, ReturnNode) and node.value is None:
            node.value = IdentifierNode(hit_name)
        return None

    _transform(function.body, visit)
    statements = function.body.statements
    if not statements or not isinstance(statements[-1], ReturnNode):
        statements.append(ReturnNode(IdentifierNode(hit_name)))
    function.return_type = NamedType(struct_name)
    if not _PRIMITIVE_ATTRIBUTES & set(_attribute_names(function)):
        function.attributes = list(function.attributes or []) + [
            AttributeNode("bounding_box")
        ]


def _names_used(function):
    return {node.name for node in _walk(function) if isinstance(node, IdentifierNode)}


def _program_functions(ast):
    functions = list(getattr(ast, "functions", None) or [])
    for stage in (getattr(ast, "stages", None) or {}).values():
        functions.extend(getattr(stage, "local_functions", None) or [])
        entry = getattr(stage, "entry_point", None)
        if entry is not None:
            functions.append(entry)
    return [function for function in functions if isinstance(function, FunctionNode)]


def _localize_hit_attribute_variables(ast, entries, refuse_lowering):
    """Make each GLSL ``hitAttributeEXT`` variable a local of the intersection entries that write it.

    The variable carries the attributes ``reportIntersectionEXT`` reports to the hit shaders. A Metal intersection
    function returns no attributes (MSL section 5.2.3.8), so the variable is only storage the entry writes; another
    function that uses it would read attributes that Metal never delivers, which is refused.
    """
    owners = [ast] + list((getattr(ast, "stages", None) or {}).values())
    entry_ids = {id(function) for function in entries}
    for owner in owners:
        field = "global_variables" if owner is ast else "local_variables"
        variables = list(getattr(owner, field, None) or [])
        kept = []
        for variable in variables:
            if not _HIT_ATTRIBUTE_VARIABLES & set(_attribute_names(variable)):
                kept.append(variable)
                continue
            users = [
                function
                for function in _program_functions(ast)
                if variable.name in _names_used(function)
            ]
            others = [function for function in users if id(function) not in entry_ids]
            if others:
                kept.append(variable)
                refuse_lowering(
                    f"intersection stage: hit attribute variable '{variable.name}' is used by "
                    f"'{others[0].name}': a Metal intersection function returns only acceptance and distance, "
                    "and intersector<>::intersect delivers no hit attributes (MSL sections 5.2.3.8 and 6.18)"
                )
                continue
            for function in users:
                local = deepcopy(variable)
                local.attributes = [
                    attribute
                    for attribute in local.attributes or []
                    if str(getattr(attribute, "name", "")).strip().lower()
                    not in _HIT_ATTRIBUTE_VARIABLES
                ]
                function.body.statements.insert(0, local)
        setattr(owner, field, kept)


def lower_metal_reported_hits(ast, refuse_lowering):
    """Return ``ast`` with each void intersection entry point returning a Metal bounding-box intersection result.

    ``refuse_lowering`` records a construct the lowering cannot keep. Returns ``ast`` itself when no entry point
    returns void.
    """
    if not _void_intersection_entries(ast):
        return ast
    ast = deepcopy(ast)
    entries = _void_intersection_entries(ast)
    for reason in _refusals(ast, entries):
        refuse_lowering(reason)
    _localize_hit_attribute_variables(ast, entries, refuse_lowering)
    taken = _program_names(ast)
    struct_name = _unique(_RESULT_STRUCT, taken)
    helper_name = _unique(_HELPER, taken)
    struct = helper = None
    for function in entries:
        hit_name = _unique(_HIT_LOCAL, taken)
        min_name = _distance_parameter(function, "min_distance", _MIN_PARAMETER, taken)
        max_name = _distance_parameter(function, "max_distance", _MAX_PARAMETER, taken)
        struct, helper, prologue = _parse_support(
            struct_name, helper_name, hit_name, max_name
        )
        _lower_entry(function, struct_name, helper_name, hit_name, min_name)
        function.body.statements[:0] = prologue
    ast.structs = [struct] + list(getattr(ast, "structs", None) or [])
    ast.functions = [helper] + list(getattr(ast, "functions", None) or [])
    return ast
