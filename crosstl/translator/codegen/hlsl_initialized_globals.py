"""HLSL declarations of initialized CrossGL program-scope variables.

A CrossGL variable declared at program scope with an initializer and without a storage qualifier starts from its
initializer: SPIR-V and WGSL output declare it ``Private`` with that initializer and Metal output a ``constant`` of
that value. Uniforms are spelled as ``cbuffer`` members in CrossGL, which is what the GLSL importer writes for a GLSL
``uniform float u = 2.0;``.

HLSL reads a module variable without ``static`` from the ``$Globals`` constant buffer, and DXC ignores its
initializer ("Initializer of external global will be ignored"), so the generator's ``W G;`` read whatever the host
bound there instead of the initializer: ``W G = {float[2](1.0, 2.0), 3.0};`` read ``G.m[1]`` as 0. Such a variable
is declared ``static`` with its initializer instead, which is HLSL's per-invocation private storage.

A variable the HLSL importer marks ``@hlsl_program_constant`` is a ``$Globals`` member of the source and keeps that
form, as does an uninitialized one, which has no value of its own for the generator to keep.
"""

from __future__ import annotations

from .hlsl_struct_initializers import hlsl_brace_struct_initializer

__all__ = ["hlsl_initialized_private_global", "hlsl_static_global_initializer"]

# Qualifiers under which a program-scope declaration is not private per-invocation storage.
_SHARED_QUALIFIERS = frozenset(
    {
        "uniform",
        "extern",
        "groupshared",
        "shared",
        "threadgroup",
        "workgroup",
        "in",
        "out",
        "inout",
        "buffer",
    }
)


def _is_program_constant(node):
    return any(
        str(getattr(attribute, "name", "")) == "hlsl_program_constant"
        for attribute in getattr(node, "attributes", None) or []
    )


def hlsl_initialized_private_global(node, qualifier):
    """Whether ``node``, a program-scope value declared with HLSL ``qualifier``, needs ``static`` to keep its
    initializer."""
    if getattr(node, "initial_value", None) is None:
        return False
    if qualifier.strip().startswith(("static", "groupshared")):
        return False
    qualifiers = {
        str(value).lower() for value in getattr(node, "qualifiers", None) or []
    }
    return not qualifiers & _SHARED_QUALIFIERS and not _is_program_constant(node)


def hlsl_static_global_initializer(codegen, node):
    """The initializer text of a ``static`` program-scope declaration.

    A structure whose array members are written as lists takes an HLSL initializer list, ``{{1.0, 2.0}, 3.0}``,
    because HLSL has no array literal to pass to the structure's constructor helper
    (:mod:`.hlsl_struct_initializers`).
    """
    value = node.initial_value
    braced = hlsl_brace_struct_initializer(codegen, value, node.var_type)
    if braced is not None:
        return braced
    return codegen.generate_expression_with_expected(value, node.var_type)
