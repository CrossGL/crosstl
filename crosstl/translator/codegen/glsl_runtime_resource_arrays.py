"""Resource arrays declared without a size and indexed at run time, for the GLSL emitters.

GLSL sizes an unsized array implicitly from the largest constant index used with it (GLSL 4.60 section 4.1.9), so
``uniform sampler2D t[]`` read only as ``t[0]`` and ``t[2]`` declares ``t[3]``; the generator's resource-array
size hints compute that size. An array indexed with a value known only at run time has no implicit size: it is a
runtime descriptor array (HLSL ``Texture2D t[]``, SPIR-V ``OpTypeRuntimeArray`` under ``RuntimeDescriptorArray``).
Declaring it with any fixed size would put every element past that size out of bounds.

* Desktop GLSL declares it as ``t[]`` under GL_EXT_nonuniform_qualifier, the extension that allows a resource array
  to be runtime-sized and indexed with a variable (glslang rejects ``t[i]`` on an unsized sampler array without
  it), so :func:`runtime_indexed_resource_arrays` makes the generator enable the extension.
* GLSL ES has no such extension and indexes sampler arrays only with constant integral expressions, so
  :func:`unsized_resource_array_size` refuses the array for GLSL ES output.

This covers module-scope arrays that no constant index sizes and that are indexed in place. An array that constant
indices also size, or that is passed to a function, keeps the size the generator infers from its constant indices
and propagates into the function's parameters, because GLSL passes no runtime-sized array to a function.
"""

from __future__ import annotations

from ..ast import ArrayAccessNode, FunctionCallNode, IdentifierNode

__all__ = ["runtime_indexed_resource_arrays", "unsized_resource_array_size"]

_NON_UNIFORM = frozenset({"nonuniformEXT", "NonUniformResourceIndex"})
_CACHE = "_runtime_indexed_resource_arrays"


def _index_operand(codegen, index):
    """``index`` without a non-uniform qualifier, which does not change its value."""
    while (
        isinstance(index, FunctionCallNode)
        and codegen.function_call_name(index) in _NON_UNIFORM
        and len(getattr(index, "args", None) or []) == 1
    ):
        index = index.args[0]
    return index


def _parameter_names(function):
    return {
        getattr(parameter, "name", None)
        for parameter in getattr(function, "parameters", None) or []
    }


def _scan(codegen, function, unsized, user_functions, constants, indexed, passed):
    """Record the arrays of ``unsized`` that ``function`` indexes at run time or passes to a user function."""
    visible = unsized - _parameter_names(function)
    if not visible:
        return
    for node in codegen.walk_ast(getattr(function, "body", None)):
        if isinstance(node, FunctionCallNode):
            if codegen.function_call_name(node) in user_functions:
                for argument in getattr(node, "args", None) or []:
                    if (
                        isinstance(argument, IdentifierNode)
                        and argument.name in visible
                    ):
                        passed.add(argument.name)
            continue
        if not isinstance(node, ArrayAccessNode):
            continue
        array = getattr(node, "array_expr", None)
        if not isinstance(array, IdentifierNode) or array.name not in visible:
            continue
        index = _index_operand(codegen, getattr(node, "index_expr", None))
        if codegen.literal_int_value(index, constants) is None:
            indexed.add(array.name)


def runtime_indexed_resource_arrays(codegen, ast) -> frozenset:
    """Names of the unsized module-scope resource arrays that ``ast`` indexes in place with a non-constant value.

    Arrays that a constant index sizes, and arrays passed to a user function, are left out (see the module notes).
    """
    cached = getattr(codegen, _CACHE, None)
    if cached is not None and cached[0] is ast:
        return cached[1]
    hints = getattr(codegen, "resource_array_size_hints", None) or {}
    unsized = {
        name
        for name in codegen.collect_unsized_sampled_texture_globals(ast)
        if not hints.get(name)
    }
    indexed, passed = set(), set()
    if unsized:
        constants = getattr(codegen, "literal_int_constants", None) or {}
        functions = codegen.collect_functions(ast)
        user_functions = {getattr(function, "name", None) for function in functions}
        for function in functions:
            _scan(
                codegen, function, unsized, user_functions, constants, indexed, passed
            )
    result = frozenset(indexed - passed)
    setattr(codegen, _CACHE, (ast, result))
    return result


def _is_es_output(codegen):
    return codegen.glsl_version_is_es(
        getattr(codegen, "current_glsl_version_line", None)
    )


def unsized_resource_array_size(codegen, name) -> str:
    """The size of ``name``, an unsized resource array no constant index sizes: ``"1"``, or ``""`` for ``[]``."""
    cached = getattr(codegen, _CACHE, None)
    if cached is None or name not in cached[1]:
        # Unused, read only through constant index 0, or passed to a function.
        return "1"
    if _is_es_output(codegen):
        codegen.refuse_lowering(
            f"runtime-sized resource array '{name}' is indexed with a value known only at run time: GLSL ES "
            "indexes sampler and image arrays only with constant integral expressions and has no runtime-sized "
            "resource arrays"
        )
        return "1"
    return ""
