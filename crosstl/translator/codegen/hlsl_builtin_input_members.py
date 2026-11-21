"""Entry input structures with members HLSL reads through an intrinsic.

A CrossGL stage can read a built-in through a member of its entry structure, such as ``bool helper @
gl_HelperInvocation`` in a fragment input. HLSL reads some built-ins only with an intrinsic, not through a
system-value semantic: ``IsHelperLane()`` (Shader Model 6.6) for ``gl_HelperInvocation``,
``GetRenderTargetSampleCount()`` for ``gl_NumSamples``, and ``WaveGetLaneCount()``, ``WaveGetLaneIndex()`` and the
lane masks built from it for the subgroup values. Such a member has no semantic HLSL can give it. Written as is, it
would become a user semantic that no stage writes, so the generator refuses it
(:func:`.hlsl_extension_builtins.refuse_unmapped_hlsl_builtin`).

An entry parameter with one of these built-ins already becomes a local initialised from the intrinsic
(:func:`.hlsl_extension_builtins.hlsl_builtin_input_parameter`). This pass does the same for a structure member. The
entry takes a signature structure that holds the structure's other members, in their order and with their
attributes, and its body starts by building the program's structure from it. Each built-in member is read through
the CrossGL built-in, which the generator writes as the intrinsic, or refuses where the stage or profile lacks it.
The program's structure loses the built-in attribute, so it becomes an ordinary value type that the rest of the
program can pass around.

A structure is rewritten only when every stage entry that uses it takes it as a plain input parameter outside a
compute stage (compute entries read these values through their own parameters). Any other use leaves the program
unchanged, and the refusal stands.
"""

from __future__ import annotations

from copy import deepcopy

from ..ast import (
    AssignmentNode,
    ExpressionStatementNode,
    IdentifierNode,
    MemberAccessNode,
    NamedType,
    PrimitiveType,
    VariableNode,
    VectorType,
)
from ..stage_utils import normalize_stage_name
from ..wave_operations import SUBGROUP_LANE_MASK_BUILTINS
from .specialization_constants import walk_nodes

__all__ = ["INTRINSIC_INPUT_BUILTINS", "lower_hlsl_builtin_input_members"]

# The built-ins the HLSL generator reads in a stage body through an intrinsic (hlsl_extension_builtins), with their
# CrossGL types as ``(scalar, vector size)``. A member of another type keeps the refusal: the parameter path refuses
# a type that cannot hold the value, and a member must not convert it silently.
_BUILTIN_TYPES = {
    "gl_HelperInvocation": ("bool", None),
    "gl_NumSamples": ("int", None),
    "gl_SubgroupSize": ("uint", None),
    "gl_SubgroupInvocationID": ("uint", None),
    **{name: ("uint", 4) for name in SUBGROUP_LANE_MASK_BUILTINS},
}
INTRINSIC_INPUT_BUILTINS = frozenset(_BUILTIN_TYPES)

_WRITING_QUALIFIERS = frozenset({"out", "inout"})


def _member_builtin(member):
    """The intrinsic-read built-in ``member`` is bound to, when that is its only attribute, else ``None``."""
    attributes = list(getattr(member, "attributes", None) or [])
    if len(attributes) != 1:
        return None
    attribute = attributes[0]
    name = getattr(attribute, "name", None)
    if name not in INTRINSIC_INPUT_BUILTINS or getattr(attribute, "arguments", None):
        return None
    if _type_key(getattr(member, "member_type", None)) != _BUILTIN_TYPES[name]:
        return None
    return name


def _type_key(type_node):
    if isinstance(type_node, PrimitiveType) and not type_node.size_bits:
        return type_node.name, None
    if isinstance(type_node, VectorType):
        element = type_node.element_type
        if isinstance(element, PrimitiveType) and not element.size_bits:
            return element.name, type_node.size
    return None


def _builtin_members(struct):
    """``{member name: built-in}`` for the members of ``struct`` HLSL reads through an intrinsic."""
    found = {}
    for member in getattr(struct, "members", None) or []:
        builtin = _member_builtin(member)
        if builtin is not None:
            found[member.name] = builtin
    return found


def _named_struct(type_node):
    if isinstance(type_node, NamedType) and not type_node.generic_args:
        return type_node.name
    return None


def _stage_entries(ast):
    """``(stage name, stage node, entry function)`` for every stage, in declaration order."""
    for key, stage in (getattr(ast, "stages", None) or {}).items():
        entry = getattr(stage, "entry_point", None)
        if entry is not None:
            yield normalize_stage_name(getattr(key, "value", key)), stage, entry


def _struct_lists(ast, stage):
    """The structure lists visible in ``stage``: its own first, then the program's."""
    return [
        getattr(stage, "local_structs", None) or [],
        getattr(ast, "structs", None) or [],
    ]


def _find_struct(ast, stage, name):
    for structs in _struct_lists(ast, stage):
        for struct in structs:
            if struct.name == name:
                return struct
    return None


def _candidates(ast):
    """``{id(struct): struct}`` for the structures whose entry uses all allow the rewrite."""
    candidates, blocked = {}, set()
    for stage_name, stage, entry in _stage_entries(ast):
        returned = _find_struct(ast, stage, _named_struct(entry.return_type))
        if returned is not None:
            blocked.add(id(returned))
        for parameter in entry.parameters or []:
            struct = _find_struct(ast, stage, _named_struct(parameter.param_type))
            if struct is None or not _builtin_members(struct):
                continue
            writes = _WRITING_QUALIFIERS & set(parameter.qualifiers or [])
            if stage_name in (None, "compute") or writes or struct.generic_params:
                blocked.add(id(struct))
            else:
                candidates[id(struct)] = struct
    return {key: struct for key, struct in candidates.items() if key not in blocked}


def _taken_names(ast):
    """Every name a declaration or an identifier of the program spells."""
    names = set()
    for node in walk_nodes([ast]):
        name = getattr(node, "name", None)
        if isinstance(name, str):
            names.add(name)
    return names


def _fresh(base, taken):
    candidate, index = base, 1
    while candidate in taken:
        index += 1
        candidate = f"{base}{index}"
    taken.add(candidate)
    return candidate


def _signature_struct(struct, builtins, taken):
    signature = deepcopy(struct)
    signature.name = _fresh(f"{struct.name}Signature", taken)
    signature.members = [m for m in signature.members if m.name not in builtins]
    signature.methods = []
    return signature


def _prologue(parameter_name, signature_name, struct, builtins):
    """``struct parameter_name;`` and one assignment per member, built-in members from the CrossGL built-in."""
    statements = [VariableNode(parameter_name, NamedType(struct.name))]
    for member in struct.members:
        builtin = builtins.get(member.name)
        value = (
            IdentifierNode(builtin)
            if builtin is not None
            else MemberAccessNode(IdentifierNode(signature_name), member.name)
        )
        target = MemberAccessNode(IdentifierNode(parameter_name), member.name)
        statements.append(ExpressionStatementNode(AssignmentNode(target, value)))
    return statements


def _rewrite_entry(ast, stage, entry, rewritten, taken):
    """Give ``entry`` the signature structure of every rewritten structure it takes.

    A structure with no other member needs no signature: its parameter leaves the entry.
    """
    prologue, parameters = [], []
    for parameter in entry.parameters or []:
        struct = _find_struct(ast, stage, _named_struct(parameter.param_type))
        if struct is None or id(struct) not in rewritten:
            parameters.append(parameter)
            continue
        signature, builtins = rewritten[id(struct)]
        signature_name = _fresh(f"{parameter.name}_signature", taken)
        prologue += _prologue(parameter.name, signature_name, struct, builtins)
        if signature is not None:
            parameter.name = signature_name
            parameter.param_type = NamedType(signature.name)
            parameters.append(parameter)
    if prologue:
        entry.parameters = parameters
        entry.body.statements[:0] = prologue


def lower_hlsl_builtin_input_members(ast):
    """``ast``, or a copy whose entry structures read their intrinsic-only built-in members in the entry body."""
    if not _candidates(ast):
        return ast
    ast = deepcopy(ast)
    candidates = _candidates(ast)
    taken = _taken_names(ast)
    rewritten = {}
    for _stage_name, stage, _entry in _stage_entries(ast):
        for structs in _struct_lists(ast, stage):
            for struct in list(structs):
                if id(struct) not in candidates or id(struct) in rewritten:
                    continue
                builtins = _builtin_members(struct)
                signature = None
                if len(builtins) < len(struct.members):
                    signature = _signature_struct(struct, builtins, taken)
                    structs.insert(structs.index(struct) + 1, signature)
                for member in struct.members:
                    if member.name in builtins:
                        member.attributes = []
                rewritten[id(struct)] = (signature, builtins)
    for _stage_name, stage, entry in _stage_entries(ast):
        _rewrite_entry(ast, stage, entry, rewritten, taken)
    return ast
