"""Minimum level-of-detail clamps on texture sampling: CrossGL's ``texture*Clamp`` operations and their lowering.

A sample may clamp the level of detail it computes from below, so that it never reads a mip level finer than a
given one (used to keep reads inside the resident mip tail of a sparse or streamed texture). Every target of the
graphics triad expresses this per sample:

* HLSL: the ``Clamp`` parameter of ``Sample``, ``SampleBias``, ``SampleGrad``, ``SampleCmp`` and ``SampleCmpGrad``
  (Shader Model 5 tiled-resource overloads, e.g. ``Texture2D.Sample(S, Location, Offset, Clamp)``); the parameter
  follows an offset, which cube textures do not have.
* MSL: the ``min_lod_clamp(float)`` sample option of ``sample`` and ``sample_compare`` (Metal Shading Language
  specification, section 6.12, MSL 2.2 and later), alone or after ``bias`` or ``gradient*``, before the offset.
* GLSL: ``textureClampARB``, ``textureOffsetClampARB``, ``textureGradClampARB`` and ``textureGradOffsetClampARB``
  (GL_ARB_sparse_texture_clamp; desktop OpenGL only, so GLSL ES and WebGL 2 refuse it).
* SPIR-V: the ``MinLod`` image operand, which needs the ``MinLod`` capability and is valid only on implicit-LOD
  or ``Grad`` sampling (SPIR-V specification, section 3.14 "Image Operands").

CrossGL spells the clamp after every other operand except a bias, in GL_ARB_sparse_texture_clamp order::

    textureClamp(t[, s], P, clamp[, bias])           textureCompareClamp(t[, s], P, ref, clamp)
    textureOffsetClamp(t[, s], P, offset, clamp[, bias])
    textureGradClamp(t[, s], P, dPdx, dPdy, clamp)  textureCompareGradClamp(t[, s], P, ref, dPdx, dPdy, clamp)
    textureGradOffsetClamp(t[, s], P, dPdx, dPdy, offset, clamp)
    textureCompareOffsetClamp(t[, s], P, ref, offset, clamp)
    textureCompareGradOffsetClamp(t[, s], P, ref, dPdx, dPdy, offset, clamp)

Each operation is its unclamped counterpart (``texture``, ``textureOffset``, ... ``textureCompareGradOffset``)
plus the clamp. An emitter lowers the counterpart through its ordinary texture path, so the sampler, coordinate,
offset and resource checks stay those of the unclamped operation, and then adds the clamp to the call it produced.
When that call is not the plain sample the target's clamp attaches to (for example a stage without implicit
derivatives turned an implicit-LOD sample into an explicit-LOD one), the clamp cannot be carried and the operation
is refused with :class:`TextureLodClampUnsupportedError` rather than emitted without it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..ast import FunctionCallNode, IdentifierNode
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = [
    "GLSL_LOD_CLAMP_EXTENSION",
    "GLSL_LOD_CLAMP_FUNCTIONS",
    "TEXTURE_COMPARE_LOD_CLAMP_INTRINSIC_NAMES",
    "TEXTURE_LOD_CLAMP_FORMS",
    "TEXTURE_LOD_CLAMP_INTRINSIC_NAMES",
    "TEXTURE_SAMPLE_LOD_CLAMP_INTRINSIC_NAMES",
    "TextureLodClampCall",
    "TextureLodClampForm",
    "TextureLodClampUnsupportedError",
    "base_texture_call_node",
    "clamped_glsl_call",
    "clamped_hlsl_call",
    "clamped_metal_call",
    "clamped_spirv_instruction",
    "glsl_texture_lod_clamp_call",
    "hlsl_texture_lod_clamp_call",
    "is_texture_lod_clamp_operation",
    "join_call_text",
    "metal_texture_lod_clamp_call",
    "split_call_text",
    "spirv_texture_lod_clamp_call",
    "split_texture_lod_clamp_call",
]


@dataclass(frozen=True)
class TextureLodClampForm:
    """One clamped operation: its unclamped counterpart and the operands between the coordinate and the clamp."""

    name: str
    base: str
    operands: int
    compare: bool = False
    gradient: bool = False
    offset: bool = False
    accepts_bias: bool = False


TEXTURE_LOD_CLAMP_FORMS = {
    form.name: form
    for form in (
        TextureLodClampForm("textureClamp", "texture", 0, accepts_bias=True),
        TextureLodClampForm(
            "textureOffsetClamp", "textureOffset", 1, offset=True, accepts_bias=True
        ),
        TextureLodClampForm("textureGradClamp", "textureGrad", 2, gradient=True),
        TextureLodClampForm(
            "textureGradOffsetClamp",
            "textureGradOffset",
            3,
            gradient=True,
            offset=True,
        ),
        TextureLodClampForm("textureCompareClamp", "textureCompare", 1, compare=True),
        TextureLodClampForm(
            "textureCompareOffsetClamp",
            "textureCompareOffset",
            2,
            compare=True,
            offset=True,
        ),
        TextureLodClampForm(
            "textureCompareGradClamp",
            "textureCompareGrad",
            3,
            compare=True,
            gradient=True,
        ),
        TextureLodClampForm(
            "textureCompareGradOffsetClamp",
            "textureCompareGradOffset",
            4,
            compare=True,
            gradient=True,
            offset=True,
        ),
    )
}
TEXTURE_LOD_CLAMP_INTRINSIC_NAMES = frozenset(TEXTURE_LOD_CLAMP_FORMS)
TEXTURE_COMPARE_LOD_CLAMP_INTRINSIC_NAMES = frozenset(
    name for name, form in TEXTURE_LOD_CLAMP_FORMS.items() if form.compare
)
TEXTURE_SAMPLE_LOD_CLAMP_INTRINSIC_NAMES = (
    TEXTURE_LOD_CLAMP_INTRINSIC_NAMES - TEXTURE_COMPARE_LOD_CLAMP_INTRINSIC_NAMES
)

GLSL_LOD_CLAMP_EXTENSION = "GL_ARB_sparse_texture_clamp"
# The GLSL call an unclamped operation renders to, and the GL_ARB_sparse_texture_clamp function that adds a clamp.
GLSL_LOD_CLAMP_FUNCTIONS = {
    "texture": "textureClampARB",
    "textureOffset": "textureOffsetClampARB",
    "textureGrad": "textureGradClampARB",
    "textureGradOffset": "textureGradOffsetClampARB",
}

_HLSL_METHODS = {
    ("texture", False): "Sample",
    ("texture", True): "SampleBias",
    ("textureOffset", False): "Sample",
    ("textureOffset", True): "SampleBias",
    ("textureGrad", False): "SampleGrad",
    ("textureGradOffset", False): "SampleGrad",
    ("textureCompare", False): "SampleCmp",
    ("textureCompareOffset", False): "SampleCmp",
    ("textureCompareGrad", False): "SampleCmpGrad",
    ("textureCompareGradOffset", False): "SampleCmpGrad",
}
_METAL_SAMPLE_OPTIONS = ("bias(", "level(", "gradient", "min_lod_clamp(")
_CALLEE_CHARACTER = re.compile(r"[A-Za-z0-9_.:]")
_OPENING = {"(": ")", "[": "]", "{": "}"}
_CLOSING = {value: key for key, value in _OPENING.items()}


def is_texture_lod_clamp_operation(operation) -> bool:
    """Whether ``operation`` names a clamped texture operation."""
    return operation in TEXTURE_LOD_CLAMP_FORMS


class TextureLodClampUnsupportedError(TranslationDiagnostic):
    """A minimum-LOD clamp the target cannot attach to the sample it lowers to."""

    def __init__(self, target, operation, reason, *, source_location=None):
        self.operation = operation
        self.source_location = source_location
        self.missing_capabilities = (f"{target}.texture-lod-clamp",)
        super().__init__(
            f"texture operation '{operation}' clamps the minimum level of detail, which {target} cannot carry "
            f"here: {reason}",
            target=target,
            code=f"{CODE_PREFIX}{target}-texture-lod-clamp-unsupported",
        )


@dataclass
class TextureLodClampCall:
    """A clamped call split into its unclamped counterpart's arguments, the clamp and whether a bias follows."""

    form: TextureLodClampForm
    base_arguments: list
    clamp: object
    has_bias: bool


def split_texture_lod_clamp_call(backend_name, operation, args, explicit_sampler):
    """Split ``operation(args)``; raises ``ValueError`` when the argument count fits no form of the operation."""
    form = TEXTURE_LOD_CLAMP_FORMS[operation]
    args = list(args)
    coordinate_index = 2 if explicit_sampler else 1
    trailing = args[coordinate_index + 1 :]
    expected = [form.operands + 1] + ([form.operands + 2] if form.accepts_bias else [])
    if len(args) <= coordinate_index or len(trailing) not in expected:
        counts = " or ".join(str(coordinate_index + 1 + count) for count in expected)
        raise ValueError(
            f"{backend_name} texture operation '{operation}' accepts {counts} argument(s), got {len(args)}"
        )
    has_bias = len(trailing) == form.operands + 2
    clamp = trailing[form.operands]
    base_arguments = args[: coordinate_index + 1] + trailing[: form.operands]
    if has_bias:
        base_arguments.append(trailing[-1])
    return TextureLodClampCall(form, base_arguments, clamp, has_bias)


def base_texture_call_node(call, call_node=None):
    """The unclamped counterpart as a CrossGL call node carrying ``call_node``'s location and type."""
    return FunctionCallNode(
        IdentifierNode(call.form.base),
        list(call.base_arguments),
        source_location=getattr(call_node, "source_location", None),
        expression_type=getattr(call_node, "expression_type", None),
    )


def split_call_text(text):
    """``(callee, arguments)`` when ``text`` is exactly one call ``callee(a, b, ...)``, otherwise ``None``.

    The callee may be a name, a member path (``t.Sample``, ``ns::f``) or contain balanced subscripts; arguments are
    split at top-level commas.
    """
    text = str(text or "").strip()
    if not text.endswith(")"):
        return None
    stack = []
    open_index = None
    for index in range(len(text) - 1, -1, -1):
        character = text[index]
        if character in _CLOSING:
            stack.append(character)
        elif character in _OPENING:
            if not stack or _OPENING[character] != stack[-1]:
                return None
            stack.pop()
            if not stack:
                open_index = index
                break
    if open_index is None or open_index == 0:
        return None
    callee = text[:open_index]
    depth = 0
    for character in callee:
        if character in "[(":
            depth += 1
        elif character in "])":
            depth -= 1
            if depth < 0:
                return None
        elif depth == 0 and not _CALLEE_CHARACTER.fullmatch(character):
            return None
    if depth != 0 or not callee or callee[-1] in ".:":
        return None
    return callee, _split_arguments(text[open_index + 1 : -1])


def _split_arguments(body):
    arguments, depth, start = [], 0, 0
    for index, character in enumerate(body):
        if character in _OPENING:
            depth += 1
        elif character in _CLOSING:
            depth -= 1
        elif character == "," and depth == 0:
            arguments.append(body[start:index].strip())
            start = index + 1
    tail = body[start:].strip()
    if tail or arguments:
        arguments.append(tail)
    return arguments


def join_call_text(callee, arguments):
    return f"{callee}({', '.join(arguments)})"


def _method(callee):
    return callee.rsplit(".", 1)[-1] if "." in callee else None


def clamped_hlsl_call(
    operation, call, base_text, clamp_text, *, cube_family, zero_offset
):
    """Add ``Clamp`` to the HLSL sample ``base_text``; a non-cube sample without an offset gets ``zero_offset``."""
    form = call.form
    expected = _HLSL_METHODS.get((form.base, call.has_bias))
    parts = split_call_text(base_text)
    expected_count = 2 + form.operands + (1 if call.has_bias else 0)
    if parts is None or _method(parts[0]) != expected:
        raise TextureLodClampUnsupportedError(
            "directx",
            operation,
            f"the sample lowered to '{base_text}', not to {expected}, which is where HLSL takes a Clamp operand",
        )
    callee, arguments = parts
    if len(arguments) != expected_count:
        raise TextureLodClampUnsupportedError(
            "directx",
            operation,
            f"the sample lowered to '{base_text}', whose operands do not match {expected}(S, Location, ...)",
        )
    if not form.offset and not cube_family:
        if zero_offset is None:
            raise TextureLodClampUnsupportedError(
                "directx",
                operation,
                "HLSL takes the Clamp operand after an offset, and the resource has no offset dimension",
            )
        arguments.append(zero_offset)
    arguments.append(clamp_text)
    return join_call_text(callee, arguments)


def clamped_metal_call(operation, call, base_text, clamp_text):
    """Add ``min_lod_clamp`` to the MSL sample ``base_text``, before its offset when it has one."""
    expected = "sample_compare" if call.form.compare else "sample"
    parts = split_call_text(base_text)
    if parts is None or _method(parts[0]) != expected:
        raise TextureLodClampUnsupportedError(
            "metal",
            operation,
            f"the sample lowered to '{base_text}', not to {expected}, which is where MSL takes min_lod_clamp",
        )
    callee, arguments = parts
    if any(argument.startswith("level(") for argument in arguments):
        raise TextureLodClampUnsupportedError(
            "metal",
            operation,
            f"the sample lowered to an explicit level ('{base_text}'), and MSL clamps only computed levels",
        )
    option = f"min_lod_clamp({clamp_text})"
    if call.form.offset:
        if not arguments or arguments[-1].startswith(_METAL_SAMPLE_OPTIONS):
            raise TextureLodClampUnsupportedError(
                "metal",
                operation,
                f"the sample lowered to '{base_text}', which has no trailing offset",
            )
        arguments.insert(len(arguments) - 1, option)
    else:
        arguments.append(option)
    return join_call_text(callee, arguments)


def clamped_glsl_call(operation, call, base_text, clamp_text):
    """Rename the GLSL sample ``base_text`` to its GL_ARB_sparse_texture_clamp function and add the clamp."""
    parts = split_call_text(base_text)
    function = GLSL_LOD_CLAMP_FUNCTIONS.get(parts[0]) if parts else None
    if function is None:
        raise TextureLodClampUnsupportedError(
            "opengl",
            operation,
            f"the sample lowered to '{base_text}', which GL_ARB_sparse_texture_clamp has no clamped form of",
        )
    arguments = parts[1]
    if call.has_bias:
        arguments.insert(len(arguments) - 1, clamp_text)
    else:
        arguments.append(clamp_text)
    return join_call_text(function, arguments)


_SPIRV_IMAGE_OPERAND_WORDS = {
    "Bias": 1,
    "Lod": 1,
    "Grad": 2,
    "ConstOffset": 1,
    "Offset": 1,
    "ConstOffsets": 1,
    "Sample": 1,
    "MinLod": 1,
}


def clamped_spirv_instruction(operation, line, clamp_id, image_operands):
    """Add ``MinLod %clamp_id`` to the sampling instruction ``line``; ``image_operands`` orders the operand mask.

    ``line`` is ``%r = OpImageSample... %type %image %coordinate [%dref] [Mask ids...]``.
    """
    words = str(line).split()
    if len(words) < 6 or words[1] != "=" or not words[2].startswith("OpImageSample"):
        raise TextureLodClampUnsupportedError(
            "vulkan",
            operation,
            f"the sample lowered to '{line}', not to an OpImageSample instruction",
        )
    fixed = 7 if "Dref" in words[2] else 6
    if len(words) < fixed:
        raise TextureLodClampUnsupportedError(
            "vulkan", operation, f"the sample '{line}' has too few operands"
        )
    head, tail = words[:fixed], words[fixed:]
    operands = []
    if tail:
        masks = tail[0].split("|")
        values = tail[1:]
        for mask in masks:
            count = _SPIRV_IMAGE_OPERAND_WORDS.get(mask)
            if count is None or len(values) < count:
                raise TextureLodClampUnsupportedError(
                    "vulkan",
                    operation,
                    f"the sample lowered to '{line}', whose image operands are not understood",
                )
            operands.append(" ".join([mask, *values[:count]]))
            values = values[count:]
        if values:
            raise TextureLodClampUnsupportedError(
                "vulkan",
                operation,
                f"the sample lowered to '{line}', whose image operands are not understood",
            )
    if "ExplicitLod" in words[2] and not any(
        operand.startswith("Grad ") for operand in operands
    ):
        raise TextureLodClampUnsupportedError(
            "vulkan",
            operation,
            "the stage has no implicit derivatives, so the sample lowered to an explicit Lod, and SPIR-V allows "
            "MinLod only on implicit-LOD or Grad sampling (SPIR-V 3.14, Image Operands)",
        )
    if any(operand.startswith("MinLod ") for operand in operands):
        raise TextureLodClampUnsupportedError(
            "vulkan", operation, f"the sample '{line}' already has a MinLod operand"
        )
    operands.append(f"MinLod %{clamp_id}")
    return " ".join(head + [image_operands(*operands)])


def _has_pending_refusal(codegen):
    refusals = getattr(codegen, "refusals", None)
    return bool(getattr(refusals, "reasons", None))


def _lowered_base(codegen, backend_name, operation, args, call_node):
    """Split the call and render its unclamped counterpart through the emitter's ordinary texture path."""
    call = split_texture_lod_clamp_call(
        backend_name, operation, args, codegen.is_explicit_sampler_argument(args)
    )
    base_text = codegen.generate_expression(base_texture_call_node(call, call_node))
    return call, base_text


def _clamped(codegen, call_node, base_text, lower):
    """``lower()``, or ``base_text`` when the counterpart was itself refused.

    A counterpart the emitter refused renders as a placeholder that no clamp attaches to; the pending refusal
    already keeps the program from being returned, and it names the actual cause.
    """
    try:
        return lower()
    except TextureLodClampUnsupportedError as error:
        if _has_pending_refusal(codegen):
            return base_text
        if error.source_location is None:
            error.source_location = getattr(call_node, "source_location", None)
        raise error from None


def hlsl_texture_lod_clamp_call(codegen, operation, args, call_node=None):
    """HLSL for a clamped CrossGL texture operation (``Sample(S, Location, Offset, Clamp)`` and its kin)."""
    call, base_text = _lowered_base(codegen, "DirectX", operation, args, call_node)
    texture_type = codegen.texture_resource_type(args[0])
    shape = codegen.sampled_texture_shape_type(texture_type)
    offset_operation = "textureCompareOffset" if call.form.compare else "textureOffset"
    dimension = codegen.resource_offset_dimension(offset_operation, texture_type)
    zero_offset = (
        codegen.hlsl_render_texture_offset_value((0,) * dimension)
        if dimension
        else None
    )
    return _clamped(
        codegen,
        call_node,
        base_text,
        lambda: clamped_hlsl_call(
            operation,
            call,
            base_text,
            codegen.generate_expression(call.clamp),
            cube_family=shape in {"TextureCube", "TextureCubeArray"},
            zero_offset=zero_offset,
        ),
    )


def _metal_clamp(codegen, operation, call, args, base_text):
    texture_type = codegen.texture_argument_resource_type(args[0])
    if texture_type and codegen.is_texture1d_sample_resource(texture_type):
        raise TextureLodClampUnsupportedError(
            "metal",
            operation,
            "MSL 1D textures have no mipmaps and take no sample options (MSL specification, section 6.12.1)",
        )
    return clamped_metal_call(
        operation, call, base_text, codegen.generate_expression(call.clamp)
    )


def metal_texture_lod_clamp_call(codegen, operation, args, call_node=None):
    """MSL for a clamped CrossGL texture operation (the ``min_lod_clamp`` sample option)."""
    call, base_text = _lowered_base(codegen, "Metal", operation, args, call_node)
    return _clamped(
        codegen,
        call_node,
        base_text,
        lambda: _metal_clamp(codegen, operation, call, args, base_text),
    )


def glsl_texture_lod_clamp_call(codegen, operation, args, call_node=None):
    """GLSL for a clamped CrossGL texture operation (the GL_ARB_sparse_texture_clamp functions)."""
    call, base_text = _lowered_base(codegen, "GLSL", operation, args, call_node)
    return _clamped(
        codegen,
        call_node,
        base_text,
        lambda: clamped_glsl_call(
            operation, call, base_text, codegen.generate_expression(call.clamp)
        ),
    )


def spirv_texture_lod_clamp_call(codegen, operation, args, explicit_sampler):
    """The SPIR-V result id for a clamped texture operation whose operands ``args`` are already evaluated.

    The unclamped counterpart emits its sampling instruction last; the ``MinLod`` image operand is then added to it.
    """
    call = split_texture_lod_clamp_call("SPIR-V", operation, args, explicit_sampler)
    if not codegen.is_scalar_numeric_value(call.clamp):
        codegen.refuse_lowering(f"{operation} requires a scalar numeric LOD clamp")
        return None
    # MinLod is a 32-bit float scalar; convert before the sample so that the sample stays the last instruction.
    clamp = codegen.convert_scalar_to_type(
        call.clamp, codegen.register_primitive_type("float")
    )
    result = codegen.call_resource_function(call.form.base, call.base_arguments)
    if result is None:
        return result
    lines = codegen.code_lines
    try:
        if not lines or not str(lines[-1]).startswith(f"%{result.id} = "):
            raise TextureLodClampUnsupportedError(
                "vulkan",
                operation,
                f"the sample of '{call.form.base}' is not a single sampling instruction",
            )
        lines[-1] = clamped_spirv_instruction(
            operation, lines[-1], clamp.id, codegen.image_operands
        )
    except TextureLodClampUnsupportedError:
        if _has_pending_refusal(codegen):
            return result
        raise
    codegen.require_capability("MinLod")
    return result
