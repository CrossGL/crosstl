"""The WGSL ``primitive_index`` built-in input.

WGSL has the primitive index only as a ``u32`` fragment input (WGSL section
13.3.1.1.11), in a module that starts with ``enable primitive_index;`` (WGSL
section 4.1.1; WebGPU feature ``"primitive-index"``). CrossGL takes it from
``gl_PrimitiveID``, which GLSL types ``int``, from HLSL ``SV_PrimitiveID`` and
from MSL ``[[primitive_id]]``, which are ``uint``.

An ``int`` entry parameter is received as ``u32`` by
:mod:`.wgsl_integer_builtins`. An ``int`` member of a fragment input structure
is declared ``u32`` and read through ``i32(...)``, so the shader computes in the
type it declared; a write to it is refused. The built-in in another stage, as
an output or with another type is refused rather than handed to WGSL, which
rejects it.
"""

from __future__ import annotations

from ..ast import MemberAccessNode, PrimitiveType
from .wgsl_stage_io import WGSLStageIOUnsupportedError

__all__ = [
    "PRIMITIVE_INDEX_ENABLE_DIRECTIVE",
    "binds_primitive_index",
    "is_int_primitive_index_member",
    "primitive_index_member_type",
    "refuse_misplaced_primitive_index",
    "refuse_primitive_index_write",
]

PRIMITIVE_INDEX_ENABLE_DIRECTIVE = "enable primitive_index;"
_BUILTIN = "@builtin(primitive_index)"
_SOURCE_TYPES = {"i32", "u32"}


def binds_primitive_index(attributes):
    """Return whether rendered WGSL attributes bind the primitive index."""
    return _BUILTIN in attributes


def primitive_index_member_type(codegen, struct_name, member):
    """Return the type a struct member bound to the primitive index is declared with.

    The member is ``u32``; an ``int`` member is recorded so that reads convert
    it back. Any other source type is refused.
    """
    type_name = codegen.type_name_string(member.member_type)
    if type_name not in _SOURCE_TYPES:
        raise WGSLStageIOUnsupportedError(
            f"WGSL target cannot bind member {struct_name}.{member.name} of type "
            f"{type_name} to primitive_index, which is u32; declare it int or uint"
        )
    if type_name == "i32":
        codegen._primitive_index_int_members.add((struct_name, member.name))
    codegen._uses_primitive_index = True
    return PrimitiveType("uint")


def _int_member(codegen, expr):
    """The ``(struct, member)`` of an ``int`` primitive index member access, else ``None``."""
    if not codegen._primitive_index_int_members or not isinstance(
        expr, MemberAccessNode
    ):
        return None
    object_type = codegen.array_element_type(codegen.expression_type(expr.object_expr))
    key = (codegen.struct_type_name(object_type), expr.member)
    return key if key in codegen._primitive_index_int_members else None


def is_int_primitive_index_member(codegen, expr):
    """Return whether ``expr`` accesses an ``int`` member that WGSL declares ``u32``."""
    return _int_member(codegen, expr) is not None


def refuse_primitive_index_write(codegen, target):
    """Refuse a write to an ``int`` member that WGSL declares ``u32``."""
    key = _int_member(codegen, target)
    if key is not None:
        raise WGSLStageIOUnsupportedError(
            f"WGSL target cannot write member {key[0]}.{key[1]}, the primitive "
            "index input; copy it into a local variable first"
        )


def refuse_misplaced_primitive_index(codegen, entry_point, stage_name):
    """Refuse the primitive index outside fragment inputs or with a type other than ``u32``.

    Integer entry parameters are already ``u32`` here (:mod:`.wgsl_integer_builtins`).
    """
    structs = codegen._stage_interface_structs_by_name
    for parameter in getattr(entry_point, "parameters", []) or []:
        param_type = getattr(parameter, "param_type", None)
        struct = structs.get(codegen.struct_type_name(param_type))
        if struct is not None:
            _refuse_struct(codegen, struct, stage_name, "input")
            continue
        if not binds_primitive_index(codegen.wgsl_attributes(parameter.attributes)):
            continue
        context = f"{stage_name} entry-point parameter '{parameter.name}'"
        if stage_name != "fragment":
            raise _stage_error(context)
        type_name = codegen.type_name_string(param_type)
        if type_name != "u32":
            raise WGSLStageIOUnsupportedError(
                f"WGSL target cannot bind {context} of type {type_name} to "
                "primitive_index, which is u32; declare it int or uint"
            )
        codegen._uses_primitive_index = True
    struct = structs.get(
        codegen.struct_type_name(getattr(entry_point, "return_type", None))
    )
    if struct is not None:
        _refuse_struct(codegen, struct, stage_name, "output")


def _refuse_struct(codegen, struct, stage_name, direction):
    for member in getattr(struct, "members", []) or []:
        if not binds_primitive_index(codegen.wgsl_attributes(member.attributes)):
            continue
        if direction == "output" or stage_name != "fragment":
            raise _stage_error(
                f"{stage_name} {direction} member {struct.name}.{member.name}"
            )


def _stage_error(context):
    return WGSLStageIOUnsupportedError(
        f"WGSL target cannot bind {context} to primitive_index; WGSL has the "
        "primitive index only as a fragment shader input"
    )
