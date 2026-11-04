"""User functions renamed where their name collides with a module-scope value.

CrossGL keeps a function and a value of the same name apart by syntax: ``f(x)`` calls the function and ``f[0]``
reads the value. Sources produce such pairs because their scopes keep them apart. A GLSL block member is reached
through the block's instance (``ubo.brightnessContrast``) and may share its name with a function
``brightnessContrast``, but the CrossGL import reads the members of a uniform block as ``cbuffer`` members, which
are module-scope names. HLSL, GLSL and MSL each have one namespace for functions and variables at file scope, so a
generator that declares both writes a redeclaration that DXC, glslang and the Metal compiler reject.

:func:`rename_colliding_functions` renames every user function (each overload, and every call to it) whose name a
module-scope variable, constant, or constant-buffer member also declares, or that a stage's variables or constant
buffers declare. The values keep their names, because the host binds them by name; a function name is not part of
the host interface. Entry points are never renamed. The new name appends ``_fn`` (then a number) to the old one
and collides with no declared name.
"""

from __future__ import annotations

from .. import ast as A
from .hub_types import call_name

__all__ = [
    "names_called_with_other_arity",
    "rename_colliding_functions",
    "rename_functions",
    "user_function_names",
]


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values() if isinstance(stages, dict) else stages)


def _member_names(cbuffers):
    for cbuffer in cbuffers or []:
        for member in getattr(cbuffer, "members", None) or []:
            yield getattr(member, "name", None)


def _value_names(program):
    names = set()
    for variable in getattr(program, "global_variables", None) or []:
        names.add(getattr(variable, "name", None))
    for constant in getattr(program, "constants", None) or []:
        names.add(getattr(constant, "name", None))
    names.update(_member_names(getattr(program, "cbuffers", None)))
    for stage in _stages(program):
        for variable in getattr(stage, "local_variables", None) or []:
            names.add(getattr(variable, "name", None))
        names.update(_member_names(getattr(stage, "local_cbuffers", None)))
    names.discard(None)
    return names


def _functions(program):
    """``(function, is_entry_point)`` for every function of ``program``."""
    for function in getattr(program, "functions", None) or []:
        yield function, False
    for stage in _stages(program):
        for function in getattr(stage, "local_functions", None) or []:
            yield function, False
        entry = getattr(stage, "entry_point", None)
        if entry is not None:
            yield entry, True


def _declared_names(program):
    names = _value_names(program)
    names.update(getattr(function, "name", None) for function, _ in _functions(program))
    for struct in getattr(program, "structs", None) or []:
        names.add(getattr(struct, "name", None))
    names.discard(None)
    return names


def _fresh(name, taken):
    candidate = f"{name}_fn"
    index = 1
    while candidate in taken:
        index += 1
        candidate = f"{name}_fn{index}"
    taken.add(candidate)
    return candidate


def _walk(node, seen):
    if id(node) in seen:
        return
    seen.add(id(node))
    yield node
    for field, value in list(vars(node).items()):
        if field == "parent":
            continue
        if isinstance(value, A.ASTNode):
            yield from _walk(value, seen)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, A.ASTNode):
                    yield from _walk(item, seen)


def _rename_call(node, old_name, new_name):
    for field in ("function", "name"):
        callee = getattr(node, field, None)
        if isinstance(callee, A.IdentifierNode) and callee.name == old_name:
            callee.name = new_name
            if hasattr(callee, "identifier"):
                callee.identifier = new_name
        elif callee == old_name:
            setattr(node, field, new_name)


def rename_colliding_functions(program) -> dict:
    """Rename the user functions of ``program`` that share a name with a value, in place; returns old -> new."""
    if not isinstance(program, A.ShaderNode):
        return {}
    values = _value_names(program)
    functions = list(_functions(program))
    entry_names = {
        getattr(function, "name", None) for function, entry in functions if entry
    }
    colliding = sorted(
        {
            function.name
            for function, entry in functions
            if not entry
            and function.name in values
            and function.name not in entry_names
        }
    )
    return rename_functions(program, colliding)


def user_function_names(program):
    """The names of the user functions of ``program`` other than entry points."""
    functions = list(_functions(program))
    entry_names = {
        getattr(function, "name", None) for function, entry in functions if entry
    }
    return {
        function.name
        for function, entry in functions
        if not entry and function.name not in entry_names
    }


def _call_arity(node):
    arguments = getattr(node, "arguments", getattr(node, "args", None)) or []
    return len(arguments)


def _function_arities(functions, names):
    arities = {}
    for function, entry in functions:
        if not entry and function.name in names:
            parameters = getattr(
                function, "parameters", getattr(function, "params", [])
            )
            arities.setdefault(function.name, set()).add(len(parameters or []))
    return arities


def names_called_with_other_arity(program) -> set:
    """User function names that some call uses with an argument count none of their overloads takes.

    Such a call can only reach a built-in function of the same name, as ``clamp(x, 0.0, 1.0)`` beside a user
    ``clamp(float)`` does, so a generator that resolves calls by name alone must rename the user function first.
    """
    names = user_function_names(program)
    if not names:
        return set()
    functions = list(_functions(program))
    arities = _function_arities(functions, names)
    found = set()
    seen = set()
    for function, _ in functions:
        for node in _walk(function, seen):
            if isinstance(node, A.FunctionCallNode):
                name = call_name(node)
                if name in arities and _call_arity(node) not in arities[name]:
                    found.add(name)
    return found


def rename_functions(program, names, *, by_arity=False) -> dict:
    """Rename the user functions in ``names``, each overload and every call to it, in place; returns old -> new.

    A new name appends ``_fn`` (then a number) to the old one and collides with no declared name. With
    ``by_arity``, a call is redirected only when one of the renamed overloads takes its number of arguments, so a
    call that only a target built-in of the same name accepts keeps the name.
    """
    names = sorted(set(names) & user_function_names(program))
    if not names:
        return {}
    functions = list(_functions(program))
    taken = _declared_names(program)
    renamed = {name: _fresh(name, taken) for name in names}
    arities = _function_arities(functions, renamed)
    for function, entry in functions:
        if not entry and function.name in renamed:
            function.name = renamed[function.name]
    seen = set()
    for function, _ in functions:
        for node in _walk(function, seen):
            if isinstance(node, A.FunctionCallNode) and call_name(node) in renamed:
                old_name = call_name(node)
                if by_arity and _call_arity(node) not in arities[old_name]:
                    continue
                _rename_call(node, old_name, renamed[old_name])
    return renamed
