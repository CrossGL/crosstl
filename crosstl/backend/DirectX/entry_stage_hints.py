"""Entry-point name hints for HLSL stage inference.

HLSL does not mark an entry point in the source: the compiler is told the entry
name and the target profile (``-E VSMain -T vs_6_0``).  When neither is given,
the importer infers the stage from the function's attributes and semantics, and
falls back on the conventional two-letter profile prefixes (``VSMain``,
``ps_main``, ``CSMain``).  A prefix is weak evidence, so this module only lets it
decide when the function could be that stage's entry point at all:

* the prefix must be a whole word: ``VSMain``, ``vs_main``, ``vsMain``, ``VS`` and
  ``vsmain`` qualify, while ``assignValue``, ``AsFloat`` and ``msecs`` do not;
* compute, mesh and amplification entry points must carry ``[numthreads]``,
  geometry entry points ``[maxvertexcount]``, hull entry points
  ``[outputcontrolpoints]`` or ``[patchconstantfunc]`` and domain entry points
  ``[domain]`` (DXC rejects these stages without them), so a prefix without the
  attribute is ignored;
* a vertex or pixel entry point has a stage interface: a semantic on a
  parameter or on the return value, or a struct parameter or return type whose
  members carry semantics;
* an entry point is not called by another function of the translation unit.

A function that fails these checks keeps the stage its attributes and
semantics give it, which for an ordinary helper is none.
"""

from .DirectxAst import FunctionCallNode

ENTRY_NAME_STAGE_PREFIXES = {
    "vs": "vertex",
    "ps": "fragment",
    "cs": "compute",
    "gs": "geometry",
    "hs": "tessellation_control",
    "ds": "tessellation_evaluation",
    "ms": "mesh",
    "as": "task",
}

# Attributes HLSL requires on an entry point of each stage.  A stage absent
# from this table (vertex, pixel) has no mandatory entry attribute.
REQUIRED_ENTRY_ATTRIBUTES = {
    "compute": frozenset({"numthreads"}),
    "mesh": frozenset({"numthreads"}),
    "task": frozenset({"numthreads"}),
    "geometry": frozenset({"maxvertexcount"}),
    "tessellation_control": frozenset({"outputcontrolpoints", "patchconstantfunc"}),
    "tessellation_evaluation": frozenset({"domain"}),
}

INTERFACE_STAGES = frozenset({"vertex", "fragment"})


def entry_name_prefix_stage(name):
    """The stage a whole-word profile prefix of ``name`` names, if any."""
    text = str(name or "")
    prefix, rest = text[:2], text[2:]
    stage = ENTRY_NAME_STAGE_PREFIXES.get(prefix.lower())
    if stage is None or not (prefix.isupper() or prefix.islower()):
        return None
    if not rest or rest[0] == "_" or rest[0].isdigit() or rest[0].isupper():
        return stage
    if rest.lower().startswith("main"):
        return stage
    return None


def entry_name_stage_hint(name, attribute_names):
    """The stage a name prefix may decide, given the function's attributes."""
    stage = entry_name_prefix_stage(name)
    if stage is None:
        return None
    required = REQUIRED_ENTRY_ATTRIBUTES.get(stage)
    if required is not None and not (required & set(attribute_names)):
        return None
    return stage


def called_function_names(functions):
    """Names each function calls, keyed by the calling function's id."""
    calls = {}
    for function in functions:
        names = set()
        _collect_call_names(getattr(function, "body", None), names, set())
        calls[id(function)] = names
    return calls


def _collect_call_names(node, names, visited):
    if node is None or isinstance(node, (str, int, float, bool)):
        return
    node_id = id(node)
    if node_id in visited:
        return
    visited.add(node_id)
    if isinstance(node, FunctionCallNode):
        callee = getattr(node, "name", None)
        if isinstance(callee, str):
            names.add(callee)
    if isinstance(node, dict):
        children = node.values()
    elif isinstance(node, (list, tuple, set)):
        children = node
    elif hasattr(node, "__dict__"):
        children = vars(node).values()
    else:
        return
    for child in children:
        _collect_call_names(child, names, visited)


def demote_name_hinted_helpers(parser, functions, structs):
    """Drop name-prefix stages from functions that cannot be that entry point.

    ``parser`` supplies ``infer_function_qualifier`` (with ``use_name_hint``),
    ``parameter_struct_node`` and ``struct_member_stage_semantics``.
    """
    hinted = [
        function
        for function in functions
        if getattr(function, "qualifier", None)
        and function.qualifier == entry_name_prefix_stage(
            getattr(function, "name", None)
        )
    ]
    if not hinted:
        return
    structs_by_name = {
        struct.name: struct
        for struct in structs
        if getattr(struct, "name", None)
        and getattr(struct, "members", None) is not None
    }
    calls = called_function_names(functions)
    for function in hinted:
        fallback = parser.infer_function_qualifier(
            function.name,
            getattr(function, "attributes", []) or [],
            getattr(function, "params", []) or [],
            getattr(function, "semantic", None),
            getattr(function, "body", None),
            use_name_hint=False,
        )
        if fallback == function.qualifier:
            continue
        called = any(
            function.name in names
            for other_id, names in calls.items()
            if other_id != id(function)
        )
        lacks_interface = (
            function.qualifier in INTERFACE_STAGES
            and not _has_stage_interface(parser, function, structs_by_name)
        )
        if called or lacks_interface:
            _replace_stage_qualifier(function, fallback)


def _has_stage_interface(parser, function, structs_by_name):
    if getattr(function, "semantic", None):
        return True
    for param in getattr(function, "params", []) or []:
        if getattr(param, "semantic", None):
            return True
        struct = parser.parameter_struct_node(param, structs_by_name)
        if struct is not None and parser.struct_member_stage_semantics(struct):
            return True
    return_type = str(getattr(function, "return_type", "") or "").strip()
    struct = structs_by_name.get(return_type.split("<", 1)[0].strip())
    return struct is not None and bool(parser.struct_member_stage_semantics(struct))


def _replace_stage_qualifier(function, stage):
    previous = function.qualifier
    qualifiers = [
        qualifier
        for qualifier in getattr(function, "qualifiers", []) or []
        if qualifier != previous
    ]
    if stage:
        qualifiers.append(stage)
    function.qualifier = stage
    function.qualifiers = qualifiers
