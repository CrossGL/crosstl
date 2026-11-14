"""SPIR-V for CrossGL's texture residency queries: the ``OpImageSparse*`` instructions.

Every SPIR-V image read a residency query may wrap has a sparse twin taking the same operands (SPIR-V
specification, section 3.37.10, "Image Instructions"): ``OpImageSparseSampleImplicitLod`` and its explicit-LOD and
depth-compare kin, ``OpImageSparseFetch``, ``OpImageSparseGather``, ``OpImageSparseDrefGather`` and
``OpImageSparseRead``. A sparse read returns a two-member struct, an ``int`` residency code and the texel, and
``OpImageSparseTexelsResident`` tests the code. All of them need the ``SparseResidency`` capability. A CrossGL
query::

    int code = textureResidency(texture(t, s, uv), texel);

lowers its read through the ordinary texture path, which emits the sampling instruction last, and replaces that
instruction with its sparse twin::

    %res  = OpImageSparseSampleImplicitLod %ResidencyStruct %sampled %uv
    %read = OpCompositeExtract %v4float %res 1
    %code = OpCompositeExtract %int %res 0

The texel keeps the read's result id and is stored to the texel operand through an ordinary assignment, which
converts it to the operand's type; the code is the query's value. ``textureResident(code)`` is
``OpImageSparseTexelsResident``. A read whose lowering does not end in an image instruction with a sparse twin is
refused with :class:`~crosstl.translator.codegen.texture_residency.TextureResidencyUnsupportedError`.
"""

from __future__ import annotations

import re

from .spirv_exponent_builtins import _store_through_assignment
from .texture_residency import (
    TEXTURE_RESIDENCY_OPERATION,
    TEXTURE_RESIDENT_OPERATION,
    TextureResidencyUnsupportedError,
    _has_pending_refusal,
    _resident_code,
    split_texture_residency_call,
)

__all__ = [
    "SPIRV_SPARSE_OPCODES",
    "spirv_texture_residency_call",
    "spirv_texture_resident_call",
]

_TARGET = "vulkan"
# Image instruction: its sparse twin (SPIR-V specification, section 3.37.10). The projective sampling instructions
# have reserved sparse twins only, and CrossGL refuses projected reads in a residency query.
SPIRV_SPARSE_OPCODES = {
    "OpImageSampleImplicitLod": "OpImageSparseSampleImplicitLod",
    "OpImageSampleExplicitLod": "OpImageSparseSampleExplicitLod",
    "OpImageSampleDrefImplicitLod": "OpImageSparseSampleDrefImplicitLod",
    "OpImageSampleDrefExplicitLod": "OpImageSparseSampleDrefExplicitLod",
    "OpImageFetch": "OpImageSparseFetch",
    "OpImageGather": "OpImageSparseGather",
    "OpImageDrefGather": "OpImageSparseDrefGather",
    "OpImageRead": "OpImageSparseRead",
}
_INSTRUCTION = re.compile(r"%(\d+) = (OpImage\w+) %(\d+) (.+)", re.DOTALL)
_INTEGER_COMPONENTS = frozenset({"int", "uint"})


def _refuse(codegen, reason, call_node, placeholder=None):
    """Raise, unless the read was itself refused: its pending refusal already names the actual cause."""
    if _has_pending_refusal(codegen):
        return placeholder
    raise TextureResidencyUnsupportedError(
        _TARGET,
        TEXTURE_RESIDENCY_OPERATION,
        reason,
        source_location=getattr(call_node, "source_location", None),
    )


def _process_without_expected_type(codegen, expression):
    """``expression`` evaluated outside the query's expected-type context, so that no conversion follows the read."""
    saved = getattr(codegen, "current_expression_expected_type", None)
    codegen.current_expression_expected_type = None
    try:
        return codegen.process_expression(expression)
    finally:
        codegen.current_expression_expected_type = saved


def _residency_struct(codegen, texel_type):
    """The ``{int code; texel}`` struct a sparse read of ``texel_type`` returns."""
    name = f"_crosstl_sparse_residency_{texel_type.type.base_type}"
    if name in codegen.struct_types:
        return codegen.struct_types[name]
    code_type = codegen.register_primitive_type("int")
    return codegen.register_struct_type(
        name, [(code_type, "code"), (texel_type, "texel")]
    )


def spirv_texture_residency_call(codegen, arguments, call_node=None):
    """Lower ``textureResidency(read, texel)``; the arguments are CrossGL expressions, not yet evaluated."""
    read, texel = split_texture_residency_call(_TARGET, arguments, call_node)
    result = _process_without_expected_type(codegen, read)
    if result is None:
        return _refuse(codegen, "its texture read could not be lowered", call_node)
    lines = codegen.code_lines
    match = _INSTRUCTION.fullmatch(str(lines[-1])) if lines else None
    texel_type = codegen.value_types.get(result.id)
    if (
        match is None
        or int(match.group(1)) != result.id
        or match.group(2) not in SPIRV_SPARSE_OPCODES
        or texel_type is None
        or int(match.group(3)) != texel_type.id
    ):
        last = str(lines[-1]) if lines else "nothing"
        return _refuse(
            codegen,
            f"the read did not lower to a single image instruction with a sparse form "
            f"({', '.join(sorted(SPIRV_SPARSE_OPCODES))}); its last instruction is '{last}'",
            call_node,
            placeholder=result,
        )
    lines.pop()
    struct_type = _residency_struct(codegen, texel_type)
    struct_id = codegen.get_id()
    codegen.emit(
        f"%{struct_id} = {SPIRV_SPARSE_OPCODES[match.group(2)]} "
        f"%{struct_type.id} {match.group(4)}"
    )
    codegen.value_types[struct_id] = struct_type
    composite = type(result)(struct_id, struct_type.type)
    # The texel keeps the read's result id, so every record keyed by it stays valid.
    codegen.emit(f"%{result.id} = OpCompositeExtract %{texel_type.id} %{struct_id} 1")
    code = codegen.composite_extract(
        composite, codegen.register_primitive_type("int"), 0
    )
    codegen.require_capability("SparseResidency")
    _store_through_assignment(codegen, texel, result, texel_type)
    return code


def spirv_texture_resident_call(codegen, arguments, call_node=None):
    """Lower ``textureResident(code)`` to ``OpImageSparseTexelsResident``."""
    code = codegen.process_expression(_resident_code(_TARGET, arguments, call_node))
    if code is None:
        return None
    code_type = codegen.registered_value_type(code)
    component = (
        codegen.normalize_primitive_name(code_type.type.base_type)
        if code_type is not None
        else None
    )
    if component not in _INTEGER_COMPONENTS:
        raise TextureResidencyUnsupportedError(
            _TARGET,
            TEXTURE_RESIDENT_OPERATION,
            f"its operand must be the scalar integer residency code, got {component or 'an untyped value'}",
            source_location=getattr(call_node, "source_location", None),
        )
    code = codegen.convert_value_to_type(code, codegen.register_primitive_type("int"))
    bool_type = codegen.register_primitive_type("bool")
    id_value = codegen.get_id()
    codegen.emit(
        f"%{id_value} = OpImageSparseTexelsResident %{bool_type.id} %{code.id}"
    )
    codegen.value_types[id_value] = bool_type
    codegen.require_capability("SparseResidency")
    return type(code)(id_value, bool_type.type)
