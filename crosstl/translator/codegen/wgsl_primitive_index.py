"""The WGSL ``primitive_index`` built-in input.

WGSL has the primitive index only as a ``u32`` fragment input (WGSL section
13.3.1.1.11), in a module that starts with ``enable primitive_index;`` (WGSL
section 4.1.1; WebGPU feature ``"primitive-index"``). CrossGL takes it from
``gl_PrimitiveID``, which GLSL types ``int``, from HLSL ``SV_PrimitiveID`` and
from MSL ``[[primitive_id]]``, which are ``uint``.

An ``int`` entry parameter is received as ``u32`` by
:mod:`.wgsl_integer_builtins`, and an ``int`` member of a fragment input
structure is declared ``u32`` by :mod:`.wgsl_unsigned_builtin_members`. The
built-in in another stage, as an output or with another type is refused rather
than handed to WGSL, which rejects it.
"""

from __future__ import annotations

from .wgsl_stage_io import WGSLStageIOUnsupportedError

__all__ = [
    "PRIMITIVE_INDEX_ENABLE_DIRECTIVE",
    "binds_primitive_index",
    "refuse_misplaced_primitive_index",
]

PRIMITIVE_INDEX_ENABLE_DIRECTIVE = "enable primitive_index;"
_BUILTIN = "@builtin(primitive_index)"


def binds_primitive_index(attributes):
    """Return whether rendered WGSL attributes bind the primitive index."""
    return _BUILTIN in attributes


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
