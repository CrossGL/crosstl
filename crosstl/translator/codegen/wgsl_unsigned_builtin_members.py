"""Structure members bound to the unsigned scalar built-in inputs of WGSL.

WGSL declares the built-in inputs ``vertex_index``, ``instance_index``, ``local_invocation_index``, ``sample_index``
and ``primitive_index`` as ``u32`` and accepts no other type (WGSL section 13.3.1.1). CrossGL types ``gl_VertexID``,
``gl_InstanceID``, ``gl_SampleID`` and ``gl_PrimitiveID`` ``int``, as GLSL does, while HLSL and MSL use ``uint``.

An entry parameter of another integer type is received as ``u32`` by :mod:`.wgsl_integer_builtins`. A member of a
stage input structure is declared ``u32`` instead, and an ``int`` member is read through ``i32(...)``, so the shader
computes in the type it declared. A write to such a member is refused, since the WGSL member is ``u32``; a member
of a type other than ``int`` or ``uint`` is refused too.
"""

from __future__ import annotations

import re

from ..ast import MemberAccessNode, PrimitiveType
from .wgsl_integer_builtins import SCALAR_BUILTINS
from .wgsl_stage_io import WGSLStageIOUnsupportedError

__all__ = [
    "is_int_builtin_member",
    "refuse_int_builtin_member_write",
    "scalar_builtin_of",
    "unsigned_builtin_member_type",
]

_BUILTIN = re.compile(r"@builtin\((\w+)\)")
_SOURCE_TYPES = {"i32", "u32"}


def scalar_builtin_of(attributes):
    """The unsigned scalar built-in that rendered WGSL ``attributes`` bind, or ``None``."""
    match = _BUILTIN.search(attributes or "")
    if match is None or match.group(1) not in SCALAR_BUILTINS:
        return None
    return match.group(1)


def unsigned_builtin_member_type(codegen, struct_name, member, builtin):
    """The type of a structure member bound to ``builtin``, which is ``u32``.

    An ``int`` member is recorded so that its reads convert back; any other source type is refused.
    """
    type_name = codegen.type_name_string(member.member_type)
    if type_name not in _SOURCE_TYPES:
        raise WGSLStageIOUnsupportedError(
            f"WGSL target cannot bind member {struct_name}.{member.name} of type "
            f"{type_name} to {builtin}, which is u32; declare it int or uint"
        )
    if type_name == "i32":
        codegen._int_builtin_members[(struct_name, member.name)] = builtin
    return PrimitiveType("uint")


def _int_member(codegen, expr):
    """The ``(struct, member)`` of an access to an ``int`` member that WGSL declares ``u32``, else ``None``."""
    if not codegen._int_builtin_members or not isinstance(expr, MemberAccessNode):
        return None
    object_type = codegen.array_element_type(codegen.expression_type(expr.object_expr))
    key = (codegen.struct_type_name(object_type), expr.member)
    return key if key in codegen._int_builtin_members else None


def is_int_builtin_member(codegen, expr):
    """Whether ``expr`` accesses an ``int`` member that WGSL declares ``u32``."""
    return _int_member(codegen, expr) is not None


def refuse_int_builtin_member_write(codegen, target):
    """Refuse a write to an ``int`` member that WGSL declares ``u32``."""
    key = _int_member(codegen, target)
    if key is not None:
        builtin = codegen._int_builtin_members[key]
        raise WGSLStageIOUnsupportedError(
            f"WGSL target cannot write member {key[0]}.{key[1]}, the {builtin} "
            "input; copy it into a local variable first"
        )
