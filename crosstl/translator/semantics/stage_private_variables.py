"""Stage-scope private variables become module-scope ``static`` variables.

A mutable CrossGL variable declared at stage scope without a storage, interface or resource qualifier has one copy
per invocation of the stage, which every function of the stage reads and writes: SPIR-V ``Private`` storage, a WGSL
``var<private>``. CrossGL spells the same storage at module scope as ``static``, which is what the HLSL and GLSL
importers write, and the GLSL (a plain global), HLSL and Slang (``static``) and Metal (a ``thread`` variable passed
to the functions that use it) generators carry that form. A stage-scope declaration was instead dropped by the GLSL
and HLSL generators, declared as a uniform by the Slang generator, and made a local of the entry point that the
stage's helpers cannot see by the Metal generator.

For those targets :func:`hoist_stage_private_variables` moves each such declaration to module scope as a ``static``
variable. For Metal, only declarations that a helper of the stage uses move, because the Metal generator already
gives the entry point its own copy of the others. A name that another declaration at module or stage scope uses,
or that a function of the stage redeclares, is replaced by a fresh one in that stage's functions and initializers.
An initializer that is not a constant expression of module-scope values becomes an assignment at the start of the
entry point, where stage-scope initializers take effect, so it can still read the stage's other values; GLSL
requires module-scope initializers to be constant expressions (GLSL 4.60 section 4.3). Declarations whose types
are not plain values (opaque types, pointers, unsized arrays, stage-local structures) are left to the generators.
"""

from __future__ import annotations

import re

from .. import ast as A
from .static_locals import (
    _constant_names,
    _declared_names,
    _fresh_name,
    _is_constant,
    _Renamer,
)

__all__ = ["TARGETS", "hoist_stage_private_variables"]

_HELPER_ONLY_TARGETS = frozenset({"metal"})
TARGETS = (
    frozenset(
        {
            "opengl",
            "glsl",
            "webgl",
            "gles",
            "essl",
            "directx",
            "hlsl",
            "dx11",
            "dx12",
            "d3d11",
            "d3d12",
            "slang",
        }
    )
    | _HELPER_ONLY_TARGETS
)

_SCALAR_NAMES = frozenset({"bool", "int", "uint", "float", "half", "double"})
_SIZED_SCALAR = re.compile(r"u?int(?:8|16|32|64)_t|float(?:16|32|64)_t")
_PRIVATE_QUALIFIERS = frozenset({"static"})


def hoist_stage_private_variables(program, target) -> int:
    """Move stage-scope private variables to module scope for ``target``; returns how many moved."""
    target = str(target or "").lower()
    if not isinstance(program, A.ShaderNode) or target not in TARGETS:
        return 0
    stages = _stages(program)
    structs = {
        getattr(struct, "name", None): struct
        for struct in getattr(program, "structs", None) or []
    }
    module_names = _module_names(program)
    constants = _constant_names(program)
    taken = _declared_names(program)
    hoisted_names = set()
    hoisted = []
    plans = []
    for stage in stages:
        entry = getattr(stage, "entry_point", None)
        if entry is None:
            continue
        functions = [entry] + list(getattr(stage, "local_functions", None) or [])
        candidates = [
            variable
            for variable in getattr(stage, "local_variables", None) or []
            if _is_private(variable, structs)
            and (
                target not in _HELPER_ONLY_TARGETS
                or _helpers_use(functions[1:], variable.name)
            )
        ]
        if candidates:
            plans.append((stage, entry, functions, candidates))
    other_names = _stage_declaration_names(
        stages, {id(c) for *_, candidates in plans for c in candidates}
    )
    for stage, entry, functions, candidates in plans:
        initializations = []
        for variable in candidates:
            name = variable.name
            if (
                name in module_names
                or name in hoisted_names
                or name in other_names
                or any(_redeclares(function, name) for function in functions)
            ):
                new_name = _fresh_name(f"{_stage_name(stage)}_{name}", taken)
                _rename(stage, functions, name, new_name)
                variable.name = new_name
            hoisted_names.add(variable.name)
            value = getattr(variable, "initial_value", None)
            if value is not None and not _is_constant(value, constants):
                variable.initial_value = None
                initializations.append(
                    A.AssignmentNode(A.IdentifierNode(variable.name), value, "=")
                )
            variable.qualifiers = ["static"] + [
                qualifier
                for qualifier in getattr(variable, "qualifiers", None) or []
                if str(qualifier).lower() != "static"
            ]
            variable.parent = program
            hoisted.append(variable)
        candidate_ids = {id(variable) for variable in candidates}
        stage.local_variables = [
            variable
            for variable in stage.local_variables
            if id(variable) not in candidate_ids
        ]
        if initializations:
            _prepend(entry, initializations)
    if hoisted:
        program.global_variables = list(
            getattr(program, "global_variables", None) or []
        )
        program.global_variables.extend(hoisted)
    return len(hoisted)


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values()) if isinstance(stages, dict) else list(stages)


def _stage_name(stage):
    value = getattr(stage, "stage", None)
    return str(getattr(value, "value", value) or "stage").lower()


def _qualifiers(node):
    return {str(q).lower() for q in getattr(node, "qualifiers", None) or []}


def _is_private(variable, structs):
    """A mutable stage-scope value with no storage, interface or resource qualifier."""
    if type(variable) is not A.VariableNode or not getattr(variable, "name", None):
        return False
    if not getattr(variable, "is_mutable", True):
        return False
    if not _qualifiers(variable) <= _PRIVATE_QUALIFIERS:
        return False
    if (
        getattr(variable, "semantic", None)
        or getattr(variable, "attributes", None)
        or getattr(variable, "resource_qualifiers", None)
        or getattr(variable, "is_type_alias", False)
    ):
        return False
    return _is_value_type(getattr(variable, "var_type", None), structs, set())


def _is_value_type(type_node, structs, visiting):
    if isinstance(type_node, A.PrimitiveType):
        return type_node.name in _SCALAR_NAMES
    if isinstance(type_node, (A.VectorType, A.MatrixType)):
        return _is_value_type(type_node.element_type, structs, visiting)
    if isinstance(type_node, A.ArrayType):
        return type_node.size is not None and _is_value_type(
            type_node.element_type, structs, visiting
        )
    if isinstance(type_node, A.NamedType) and not type_node.generic_args:
        if _SIZED_SCALAR.fullmatch(type_node.name):
            return True
        struct = structs.get(type_node.name)
        if struct is None or type_node.name in visiting:
            return False
        visiting = visiting | {type_node.name}
        return all(
            _is_value_type(_member_type(member), structs, visiting)
            for member in getattr(struct, "members", None) or []
        )
    return False


def _member_type(member):
    for field in ("member_type", "var_type", "vtype"):
        value = getattr(member, field, None)
        if isinstance(value, A.TypeNode):
            return value
    return None


def _module_names(program):
    names = set()
    for field in ("global_variables", "constants", "functions", "structs"):
        for node in getattr(program, field, None) or []:
            names.add(getattr(node, "name", None))
    for cbuffer in getattr(program, "cbuffers", None) or []:
        names.add(getattr(cbuffer, "name", None))
        names.update(
            getattr(member, "name", None)
            for member in getattr(cbuffer, "members", None) or []
        )
    names.discard(None)
    return names


def _stage_declaration_names(stages, excluded_ids):
    """Names every stage declares, other than the declarations that move."""
    names = set()
    for stage in stages:
        entry = getattr(stage, "entry_point", None)
        if entry is not None:
            names.add(getattr(entry, "name", None))
        for field in ("local_variables", "local_functions", "local_structs"):
            for node in getattr(stage, field, None) or []:
                if id(node) not in excluded_ids:
                    names.add(getattr(node, "name", None))
        for cbuffer in getattr(stage, "local_cbuffers", None) or []:
            names.add(getattr(cbuffer, "name", None))
            names.update(
                getattr(member, "name", None)
                for member in getattr(cbuffer, "members", None) or []
            )
    names.discard(None)
    return names


def _parameters(function):
    return list(getattr(function, "parameters", None) or [])


def _redeclares(function, name):
    if any(getattr(p, "name", None) == name for p in _parameters(function)):
        return True
    return any(
        isinstance(node, A.VariableNode) and node.name == name
        for node in _walk(getattr(function, "body", None))
    )


def _helpers_use(helpers, name):
    return any(
        isinstance(node, A.IdentifierNode) and node.name == name
        for helper in helpers
        for node in _walk(getattr(helper, "body", None))
    )


def _walk(body):
    roots = body if isinstance(body, list) else [body]
    for root in roots:
        if isinstance(root, A.ASTNode):
            yield from root.walk()


def _rename(stage, functions, old, new):
    """Rename the uses of a stage variable in the stage's functions and stage-scope initializers."""
    renamer = _Renamer(None, old, new)
    for function in functions:
        if any(getattr(p, "name", None) == old for p in _parameters(function)):
            continue
        body = getattr(function, "body", None)
        if isinstance(body, list):
            renamer._sequence(body, active=True)
        elif isinstance(body, A.ASTNode):
            renamer._node(body, active=True)
    for variable in getattr(stage, "local_variables", None) or []:
        value = getattr(variable, "initial_value", None)
        if isinstance(value, A.ASTNode):
            renamer._node(value, active=True)


def _prepend(entry, statements):
    body = getattr(entry, "body", None)
    if isinstance(body, list):
        body[:0] = statements
    elif isinstance(body, A.BlockNode):
        body.statements[:0] = statements
    else:
        entry.body = A.BlockNode(list(statements) + ([body] if body else []))
