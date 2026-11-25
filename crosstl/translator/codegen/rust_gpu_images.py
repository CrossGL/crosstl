"""Images and samplers in the rust-gpu output: ``spirv_std::Image!`` bindings and the methods that read them.

rust-gpu binds an image as an entry-point parameter ``#[spirv(descriptor_set = S, binding = B)] t: &Image!(...)``
in the UniformConstant storage class, and a sampler as ``s: &Sampler``. The ``Image!`` arguments spell the SPIR-V
image type (SPIR-V 1.6 section 3.3 ``OpTypeImage``): its dimensionality, sampled type or storage format, and whether
it is a depth, arrayed, multisampled, sampled or storage image. This module maps each CrossGL image type to that
spelling and each CrossGL image built-in to the spirv-std method that issues the same SPIR-V instruction:

- ``texture``, ``textureLod``, ``textureGrad``, ``textureProj*``, ``textureGather`` and ``textureQueryLod`` take a
  sampler: ``t.sample(*s, P)``, ``t.sample_by_lod(*s, P, lod)``, ``t.sample_depth_reference(*s, P, ref)`` for the
  depth comparisons :mod:`~crosstl.translator.semantics.shadow_samples` writes shadow samples as, and so on;
- ``texelFetch``, ``textureSize``, ``textureQueryLevels`` and ``textureSamples`` read the image alone:
  ``t.fetch_with_lod(P, lod)``, ``t.query_size_lod::<UVec2>(lod).as_ivec2()``;
- ``imageLoad``, ``imageStore``, ``imageSize`` and ``subpassLoad`` read and write storage images and input
  attachments: ``t.read(P)``, ``unsafe { t.write(P, v) }``, ``t.read_subpass(IVec2::ZERO)``.

spirv-std's combined image-sampler type (``SampledImage``) has only ``sample``, ``sample_by_lod`` and the size
queries, so a CrossGL combined sampler (``sampler2D t`` read without an explicit sampler) is split, as the HLSL and
MSL outputs split it: :func:`prepare_rust_gpu_images` declares a sampler ``t_sampler`` beside it, bound with the next
free binding of the same descriptor set, passes it to every read that needs one, and gives each helper parameter
that such a read uses its own sampler parameter. A texture array shares one sampler, as in those outputs.

A storage image's format becomes part of its CrossGL type here (``image2D<rgba32f>``), so the format reaches the
helpers that receive the image. spirv-std's ``read`` and ``write`` take the format's component count, so a load from a
one-, two- or three-component format is widened to the four components CrossGL's ``imageLoad`` returns (the missing
components are 0 and alpha is 1, as GLSL 4.60 section 8.12 defines) and a store writes the leading components.

Everything else is refused with its reason: texel offsets (spirv-std's image operands have no ``Offset``), gathers
with a depth reference or a component that is not a constant, storage images without a format, multisampled
storage and subpass reads, and resource arrays of more than one dimension.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass

from ..ast import (
    ArrayAccessNode,
    ArrayType,
    AttributeNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    LiteralNode,
    NamedType,
    ParameterNode,
    PrimitiveType,
    VariableNode,
)
from .ast_walk import walk_ast_values

__all__ = [
    "SAMPLER_TYPE_NAMES",
    "add_rust_gpu_image_imports",
    "generate_rust_gpu_image_call",
    "image_shape",
    "input_attachment_attributes",
    "rust_gpu_image_kind",
    "formatless_image_type",
    "prepare_rust_gpu_images",
    "rust_gpu_image_declaration_refusal",
    "rust_gpu_image_parameter_type",
    "rust_gpu_image_result_type",
]

# The CrossGL sampler types: separate sampler state, which rust-gpu binds as ``&Sampler``.
SAMPLER_TYPE_NAMES = frozenset(
    {
        "sampler",
        "samplerShadow",
        "comparison_sampler",
        "SamplerState",
        "SamplerComparisonState",
    }
)

_IMAGE_TYPE = re.compile(
    r"^(?P<sign>[iu]?)(?P<kind>sampler|image)(?P<dim>1D|2D|3D|Cube|Buffer)(?P<ms>MS)?(?P<arrayed>Array)?"
    r"(?P<shadow>Shadow)?(?:<(?P<format>\w+)>)?$"
)
_SUBPASS_TYPE = re.compile(r"^(?P<sign>[iu]?)subpassInput(?P<ms>MS)?$")
_DIMENSIONS = {"1D": "1D", "2D": "2D", "3D": "3D", "Cube": "cube", "Buffer": "buffer"}
_SCALARS = {"": "f32", "i": "i32", "u": "u32"}
# The coordinate components of each dimensionality, without an array layer.
_COORDINATES = {"1D": 1, "2D": 2, "3D": 3, "cube": 3, "buffer": 1, "subpass": 2}
# The components textureSize and imageSize return (a cube's faces are square, so its size has two).
_SIZE_COMPONENTS = {"1D": 1, "2D": 2, "3D": 3, "cube": 2, "buffer": 1}
_STORAGE_FORMATS = frozenset(
    "rgba32f rgba16f r32f rgba8 rgba8_snorm rg32f rg16f r11f_g11f_b10f r16f rgba16 rgb10_a2 rg16 rg8 r16 r8 "
    "rgba16_snorm rg16_snorm rg8_snorm r16_snorm r8_snorm rgba32i rgba16i rgba8i r32i rg32i rg16i rg8i r16i r8i "
    "rgba32ui rgba16ui rgba8ui r32ui rgb10_a2ui rg32ui rg16ui rg8ui r16ui r8ui".split()
)
_GLAM_VECTORS = {"f32": "Vec", "i32": "IVec", "u32": "UVec"}
_CROSSGL_VECTORS = {"f32": "vec", "i32": "ivec", "u32": "uvec"}
_CROSSGL_SCALARS = {"f32": "float", "i32": "int", "u32": "uint"}
_ZERO = {"f32": "0.0", "i32": "0", "u32": "0"}
_ONE = {"f32": "1.0", "i32": "1", "u32": "1"}

# The CrossGL image built-ins this module lowers.
_SAMPLED_READS = frozenset(
    {
        "texture",
        "textureLod",
        "textureGrad",
        "textureProj",
        "textureProjLod",
        "textureProjGrad",
        "textureGather",
        "textureQueryLod",
    }
)
# The depth comparisons semantics/shadow_samples.py writes shadow samples as: the reference is its own operand, and a
# projective coordinate keeps q as its last component, as spirv-std's project-coordinate methods take it. The value
# is the method and the operands after the sampler.
_COMPARE_METHODS = {
    "textureCompare": ("sample_depth_reference", 2),
    "textureCompareLod": ("sample_depth_reference_by_lod", 3),
    "textureCompareGrad": ("sample_depth_reference_by_gradient", 4),
    "textureCompareProj": ("sample_depth_reference_with_project_coordinate", 2),
    "textureCompareProjLod": (
        "sample_depth_reference_with_project_coordinate_by_lod",
        3,
    ),
    "textureCompareProjGrad": (
        "sample_depth_reference_with_project_coordinate_by_gradient",
        4,
    ),
}
_IMAGE_ONLY_CALLS = frozenset(
    {
        "texelFetch",
        "textureSize",
        "textureDimensions",
        "textureQueryLevels",
        "textureSamples",
        "imageLoad",
        "imageStore",
        "imageSize",
        "imageSamples",
        "subpassLoad",
    }
)
# The CrossGL image built-ins with no spirv-std form: spirv-std's image operands have no Offset (sample_with.rs).
_OFFSET_CALLS = frozenset(
    {
        "textureOffset",
        "textureLodOffset",
        "textureGradOffset",
        "textureProjOffset",
        "textureProjLodOffset",
        "textureProjGradOffset",
        "texelFetchOffset",
        "textureGatherOffset",
        "textureGatherOffsets",
        "textureCompareOffset",
        "textureCompareLodOffset",
        "textureCompareGradOffset",
        "textureCompareProjOffset",
        "textureCompareProjLodOffset",
        "textureCompareProjGradOffset",
        "textureGatherCompareOffset",
        "textureGatherCompareOffsets",
    }
)
# spirv-std has no depth-reference gather (OpImageDrefGather) method.
_UNLOWERED_CALLS = frozenset({"textureGatherCompare"})
_SAMPLER_CALLS = _SAMPLED_READS | frozenset(_COMPARE_METHODS)
RUST_GPU_IMAGE_CALLS = (
    _SAMPLER_CALLS | _IMAGE_ONLY_CALLS | _OFFSET_CALLS | _UNLOWERED_CALLS
)


@dataclass(frozen=True)
class ImageShape:
    """The SPIR-V image type of a CrossGL image type."""

    dim: str
    scalar: str
    arrayed: bool = False
    multisampled: bool = False
    depth: bool = False
    sampled: bool = True
    format: str | None = None

    @property
    def coordinates(self) -> int:
        return _COORDINATES[self.dim] + int(self.arrayed)

    @property
    def needs_sampler(self) -> bool:
        """Whether the CrossGL type is a combined sampler that sampling reads need a sampler for."""
        return self.sampled and self.dim != "buffer" and not self.multisampled

    @property
    def components(self) -> int:
        """The components spirv-std's read and write take: the format's, or 4 for a sampled image."""
        if self.format is None:
            return 4
        if self.format.startswith(("rgba", "rgb10_a2")):
            return 4
        if self.format == "r11f_g11f_b10f":
            return 3
        return 2 if self.format.startswith("rg") else 1

    def spelling(self) -> str:
        parts = [self.dim]
        if self.sampled:
            parts += [f"type={self.scalar}", "sampled"]
        else:
            parts += [
                f"format={self.format}" if self.format else f"type={self.scalar}",
                "sampled=false",
            ]
        if self.depth:
            parts.append("depth")
        if self.arrayed:
            parts.append("arrayed")
        if self.multisampled:
            parts.append("multisampled")
        return f"Image!({', '.join(parts)})"


def rust_gpu_image_kind(type_name):
    """The resource kind of a CrossGL image type: ``"texture"`` for a sampled one, ``"image"`` otherwise, else
    ``None``."""
    shape = image_shape(type_name)
    if shape is None:
        return None
    return "texture" if shape.sampled else "image"


def formatless_image_type(type_name):
    """The CrossGL image type of a storage image type with its format (``image2D`` for ``image2D<rgba32f>``)."""
    name = str(type_name or "")
    shape = image_shape(name) if "<" in name else None
    return name.split("<", 1)[0] if shape is not None and shape.format else None


def _format_scalar(format_name):
    if format_name.endswith("ui"):
        return "u32"
    if format_name.endswith("i"):
        return "i32"
    return "f32"


def image_shape(type_name) -> ImageShape | None:
    """The image type a CrossGL image type name (``sampler2DArray``, ``image2D<rgba32f>``) spells, else ``None``."""
    name = str(type_name or "").strip()
    match = _SUBPASS_TYPE.match(name)
    if match is not None:
        return ImageShape(
            "subpass",
            _SCALARS[match["sign"]],
            multisampled=bool(match["ms"]),
            sampled=False,
        )
    match = _IMAGE_TYPE.match(name)
    if match is None:
        return None
    sampled = match["kind"] == "sampler"
    dim = _DIMENSIONS[match["dim"]]
    if (match["shadow"] and not sampled) or (match["format"] and sampled):
        return None
    if dim == "buffer" and (match["ms"] or match["arrayed"] or match["shadow"]):
        return None
    scalar = _SCALARS[match["sign"]]
    if match["format"]:
        if match["format"] not in _STORAGE_FORMATS:
            return None
        if _format_scalar(match["format"]) != scalar:
            return None
    return ImageShape(
        dim,
        scalar,
        arrayed=bool(match["arrayed"]),
        multisampled=bool(match["ms"]),
        depth=bool(match["shadow"]),
        sampled=sampled,
        format=match["format"],
    )


def _array_parts(generator, type_name):
    """``(base type, sizes)`` of a CrossGL type string; ``sizes`` is empty for a type that is not an array."""
    type_name = generator.type_name_string(type_name)
    if not generator.is_array_type_name(type_name):
        return type_name, []
    return generator.c_array_type_parts(type_name)


def rust_gpu_image_parameter_type(generator, type_name):
    """The rust-gpu parameter type of a CrossGL image or sampler, or of an array of them, else ``None``."""
    base, sizes = _array_parts(generator, type_name)
    if base in SAMPLER_TYPE_NAMES:
        inner = "Sampler"
    else:
        shape = image_shape(base)
        if shape is None:
            return None
        inner = shape.spelling()
    if not sizes:
        return f"&{inner}"
    if len(sizes) != 1:
        return None
    if sizes[0] is None:
        return f"&RuntimeArray<{inner}>"
    return f"&[{inner}; {sizes[0]}]"


def rust_gpu_image_declaration_refusal(generator, node, type_name):
    """Why the image or sampler declaration ``node`` has no rust-gpu binding, ``""`` when it has one, or ``None``
    when ``type_name`` is not an image or sampler type."""
    base, sizes = _array_parts(generator, type_name)
    name = getattr(node, "name", "?")
    if base not in SAMPLER_TYPE_NAMES:
        shape = image_shape(base)
        if shape is None:
            if not _IMAGE_TYPE.match(base) and not _SUBPASS_TYPE.match(base):
                return None
            return f"resource: '{name}' ({base}) has no spirv-std Image! type"
        if not shape.sampled and shape.dim != "subpass" and shape.format is None:
            return (
                f"resource: storage image '{name}' ({base}) has no format layout qualifier, and rust-gpu "
                "storage images without a format need the StorageImage*WithoutFormat capabilities"
            )
    if len(sizes) > 1:
        return f"resource: '{name}' is an array of resource arrays, which rust-gpu does not bind"
    if getattr(node, "initial_value", getattr(node, "value", None)) is not None:
        return f"resource: '{name}' has an initializer, and a rust-gpu resource binding has none"
    return ""


def input_attachment_attributes(node):
    """The ``input_attachment_index = N`` attribute argument of a subpass input declaration, as a list."""
    for attribute in getattr(node, "attributes", None) or []:
        if str(getattr(attribute, "name", "")) != "input_attachment_index":
            continue
        arguments = list(getattr(attribute, "arguments", None) or [])
        if arguments and isinstance(getattr(arguments[0], "value", None), int):
            return [f"input_attachment_index = {arguments[0].value}"]
    return []


# --- Splitting combined samplers ----------------------------------------------------------------------------------


def _type_base_name(type_node):
    while isinstance(type_node, ArrayType):
        type_node = type_node.element_type
    if isinstance(type_node, (NamedType, PrimitiveType)):
        return type_node.name
    return str(type_node) if type_node is not None else None


def _callee_name(call):
    callee = getattr(call, "function", None)
    if isinstance(callee, IdentifierNode):
        return callee.name
    return callee if isinstance(callee, str) else None


def _root_identifier(expr):
    while isinstance(expr, ArrayAccessNode):
        expr = expr.array_expr
    return expr.name if isinstance(expr, IdentifierNode) else None


def _module_declarations(ast):
    """``(owner list, declaration)`` for every module- and stage-scope variable declaration."""
    owners = [getattr(ast, "global_variables", None)]
    for stage in (getattr(ast, "stages", None) or {}).values():
        owners.append(getattr(stage, "local_variables", None))
    for owner in owners:
        for node in list(owner or []):
            if isinstance(node, VariableNode):
                yield owner, node


def _functions(ast):
    seen = set()
    candidates = list(getattr(ast, "functions", None) or [])
    for stage in (getattr(ast, "stages", None) or {}).values():
        candidates.append(getattr(stage, "entry_point", None))
        candidates.extend(getattr(stage, "local_functions", None) or [])
    for function in candidates:
        if isinstance(function, FunctionNode) and id(function) not in seen:
            seen.add(id(function))
            yield function


def _calls(function):
    for value in walk_ast_values(getattr(function, "body", None)):
        if isinstance(value, FunctionCallNode):
            yield value


class _SamplerSplit:
    """The combined samplers of a program, and the sampler each gets (see the module docstring)."""

    def __init__(self, ast):
        self.ast = ast
        self.functions = list(_functions(ast))
        self.resources = {}
        self.samplers = set()
        for _owner, node in _module_declarations(ast):
            base = _type_base_name(node.var_type)
            if base in SAMPLER_TYPE_NAMES:
                self.samplers.add(node.name)
            shape = image_shape(base)
            if shape is not None and shape.needs_sampler:
                self.resources.setdefault(node.name, node)
        self.by_name = {}
        for function in self.functions:
            self.by_name.setdefault(function.name, []).append(function)
        self.needs = set()
        self.refusals = []

    def _texture_params(self, function):
        return {
            param.name: index
            for index, param in enumerate(getattr(function, "parameters", None) or [])
            if (shape := image_shape(_type_base_name(param.param_type))) is not None
            and shape.needs_sampler
        }

    def _sampler_names(self, function):
        params = {
            param.name
            for param in getattr(function, "parameters", None) or []
            if _type_base_name(param.param_type) in SAMPLER_TYPE_NAMES
        }
        return params | (self.samplers - self._shadowing_names(function))

    def _shadowing_names(self, function):
        return {param.name for param in getattr(function, "parameters", None) or []}

    def symbol(self, function, name):
        """``(function name, parameter index)`` for a texture parameter, ``name`` for a resource, else ``None``."""
        params = self._texture_params(function)
        if name in params:
            return (function.name, params[name])
        if name in self.resources and name not in self._shadowing_names(function):
            return name
        return None

    def implicit_reads(self, function):
        """The calls of ``function`` that read a combined sampler without an explicit sampler argument."""
        samplers = self._sampler_names(function)
        for call in _calls(function):
            arguments = list(getattr(call, "arguments", None) or [])
            if (
                _callee_name(call)
                not in _SAMPLER_CALLS | _OFFSET_CALLS | _UNLOWERED_CALLS
                or not arguments
            ):
                continue
            if len(arguments) > 1 and _root_identifier(arguments[1]) in samplers:
                continue
            symbol = self.symbol(function, _root_identifier(arguments[0]))
            if symbol is not None:
                yield call, symbol

    def texture_arguments(self, function):
        """``(call, argument index, symbol)`` for each texture argument of a call to a helper."""
        for call in _calls(function):
            callees = self.by_name.get(_callee_name(call), [])
            arguments = list(getattr(call, "arguments", None) or [])
            for callee in callees:
                for index in sorted(self._texture_params(callee).values()):
                    if index < len(arguments):
                        yield call, index, (callee.name, index), arguments[index]

    def collect(self):
        for function in self.functions:
            for _call, symbol in self.implicit_reads(function):
                self.needs.add(symbol)
        changed = True
        while changed:
            changed = False
            for function in self.functions:
                for _call, _index, param, argument in self.texture_arguments(function):
                    if param not in self.needs:
                        continue
                    symbol = self.symbol(function, _root_identifier(argument))
                    if symbol is not None and symbol not in self.needs:
                        self.needs.add(symbol)
                        changed = True


def _unique_name(base, taken):
    name = base
    suffix = 1
    while name in taken:
        name = f"{base}_{suffix}"
        suffix += 1
    taken.add(name)
    return name


def _sampler_declaration(texture, name):
    """A sampler beside ``texture``, in the same descriptor set and with the next free binding."""
    attributes = []
    for attribute in getattr(texture, "attributes", None) or []:
        if str(getattr(attribute, "name", "")).lower() in {"set", "group", "space"}:
            attributes.append(deepcopy(attribute))
    for attribute in getattr(texture, "attributes", None) or []:
        if str(getattr(attribute, "name", "")).lower() == "register":
            arguments = list(getattr(attribute, "arguments", None) or [])
            attributes.extend(AttributeNode("space", [arg]) for arg in arguments[1:2])
    return VariableNode(
        name, NamedType("sampler"), attributes=attributes, qualifiers=["uniform"]
    )


def _split_combined_samplers(generator, ast):
    split = _SamplerSplit(ast)
    split.collect()
    if not split.needs:
        return ast
    ast = deepcopy(ast)
    split = _SamplerSplit(ast)
    split.collect()
    taken = {value.name for value in walk_ast_values(ast) if hasattr(value, "name")}
    taken = {name for name in taken if isinstance(name, str)}
    sampler_of = {}
    for owner, node in _module_declarations(ast):
        if node.name in split.needs and node.name not in sampler_of:
            sampler_of[node.name] = _unique_name(f"{node.name}_sampler", taken)
            owner.insert(
                owner.index(node) + 1,
                _sampler_declaration(node, sampler_of[node.name]),
            )
    for function in split.functions:
        for name, index in split._texture_params(function).items():
            if (function.name, index) in split.needs:
                sampler_of[(function.name, index)] = _unique_name(
                    f"{name}_sampler", taken
                )
    _rewrite_calls(generator, split, sampler_of)
    for function in split.functions:
        params = list(getattr(function, "parameters", None) or [])
        for name, index in sorted(
            split._texture_params(function).items(), key=lambda item: -item[1]
        ):
            sampler = sampler_of.get((function.name, index))
            if sampler is not None:
                params.insert(index + 1, ParameterNode(sampler, NamedType("sampler")))
        function.parameters = params
        if hasattr(function, "params"):
            function.params = params
    return ast


def _rewrite_calls(generator, split, sampler_of):
    """Pass each combined sampler's sampler to the reads and helper calls that need it."""
    for function in split.functions:
        inserts = []
        for call, symbol in split.implicit_reads(function):
            inserts.append((call, 0, sampler_of[symbol]))
        for call, index, param, argument in split.texture_arguments(function):
            if param not in split.needs:
                continue
            symbol = split.symbol(function, _root_identifier(argument))
            if symbol is None or symbol not in sampler_of:
                generator.refuse_lowering(
                    f"texture argument: argument {index + 1} of '{_callee_name(call)}' is sampled through a "
                    "combined sampler that is not a texture variable, so its rust-gpu sampler cannot be passed"
                )
                continue
            inserts.append((call, index, sampler_of[symbol]))
        # Insert from the last argument back, so earlier positions stay valid.
        for call, index, sampler in sorted(
            inserts, key=lambda item: (id(item[0]), -item[1])
        ):
            arguments = list(call.arguments)
            arguments.insert(index + 1, IdentifierNode(sampler))
            call.arguments = arguments
            call.args = arguments


_STORAGE_IMAGE_TYPE = re.compile(r"^[iu]?image(1D|2D|3D|Cube|Buffer)")


def _element_type(type_node):
    while isinstance(type_node, ArrayType):
        type_node = type_node.element_type
    return type_node


def _storage_format(type_node):
    """The format a storage image type spells (``rgba32f`` for ``image2D<rgba32f>``), ``""`` for a storage image
    type without one, or ``None`` for any other type."""
    if not _STORAGE_IMAGE_TYPE.match(str(_type_base_name(type_node) or "")):
        return None
    arguments = getattr(_element_type(type_node), "generic_args", None) or []
    return str(getattr(arguments[0], "name", arguments[0])) if arguments else ""


def _with_format(type_node, format_name):
    """``type_node``, a storage image type or an array of one, with ``format_name`` spelled into its image type."""
    typed = NamedType(_type_base_name(type_node), [NamedType(str(format_name))])
    if not isinstance(type_node, ArrayType):
        return typed
    type_node = deepcopy(type_node)
    inner = type_node
    while isinstance(inner.element_type, ArrayType):
        inner = inner.element_type
    inner.element_type = typed
    return type_node


def _parameters(function):
    return list(getattr(function, "parameters", None) or [])


def _set_parameter_type(param, type_node):
    param.param_type = type_node
    if hasattr(param, "vtype"):
        param.vtype = type_node


def _received_formats(ast, functions):
    """``{(function, parameter index): formats}``: the formats of the images each caller passes to a storage image
    parameter that has none."""
    by_name = {}
    for function in functions:
        by_name.setdefault(function.name, []).append(function)
    module_formats = {
        node.name: _storage_format(node.var_type)
        for _owner, node in _module_declarations(ast)
    }
    received = {}
    for function in functions:
        scope = dict(module_formats)
        scope.update(
            (param.name, _storage_format(param.param_type))
            for param in _parameters(function)
        )
        for call in _calls(function):
            arguments = list(getattr(call, "arguments", None) or [])
            for callee in by_name.get(_callee_name(call), []):
                params = _parameters(callee)
                if len(params) != len(arguments):
                    continue
                for index, (param, argument) in enumerate(zip(params, arguments)):
                    format_name = scope.get(_root_identifier(argument))
                    if _storage_format(param.param_type) == "" and format_name:
                        received.setdefault((callee, index), set()).add(format_name)
    return received


def _propagate_parameter_formats(generator, ast):
    """Give each storage image parameter of a helper the format of the images its callers pass it.

    rust-gpu spells the format into the ``Image!`` type, so a helper parameter has to name the format of its
    arguments. A parameter that receives images of two formats, or none with a format, is refused.
    """
    functions = list(_functions(ast))
    while True:
        received = _received_formats(ast, functions)
        settled = [
            (key, formats) for key, formats in received.items() if len(formats) == 1
        ]
        for (function, index), formats in settled:
            param = _parameters(function)[index]
            _set_parameter_type(
                param, _with_format(param.param_type, next(iter(formats)))
            )
        if not settled:
            break
    for function in functions:
        for index, param in enumerate(_parameters(function)):
            if _storage_format(param.param_type) != "":
                continue
            formats = sorted(received.get((function, index), ()))
            if formats:
                reason = (
                    f"receives images of the formats {', '.join(formats)}, and a rust-gpu Image! type has one "
                    "format"
                )
            else:
                reason = (
                    "receives no image with a format layout qualifier, and rust-gpu storage images without a "
                    "format need the StorageImage*WithoutFormat capabilities"
                )
            generator.refuse_lowering(
                f"texture argument: storage image parameter '{param.name}' of '{function.name}' {reason}"
            )


def _encode_storage_formats(generator, ast):
    """Spell each storage image declaration's format into its type (``image2D<rgba32f>``), and the formats of the
    images each helper receives into its parameters."""
    for _owner, node in _module_declarations(ast):
        if _storage_format(node.var_type) != "":
            continue
        format_name = generator.storage_image_format_attribute(
            getattr(node, "attributes", None)
        )
        if format_name is not None:
            node.var_type = _with_format(node.var_type, format_name)
    _propagate_parameter_formats(generator, ast)
    return ast


def _has_storage_images(ast):
    declarations = [node.var_type for _owner, node in _module_declarations(ast)]
    parameters = [
        param.param_type
        for function in _functions(ast)
        for param in _parameters(function)
    ]
    return any(
        _storage_format(type_node) is not None
        for type_node in declarations + parameters
    )


def prepare_rust_gpu_images(generator, ast):
    """``ast`` with its combined samplers split and its storage image formats in their types (a copy when changed)."""
    ast = _split_combined_samplers(generator, ast)
    if _has_storage_images(ast):
        ast = _encode_storage_formats(generator, deepcopy(ast))
    return ast


# --- Calls ----------------------------------------------------------------------------------------------------------


def _operand_shape(generator, expr):
    """The image shape of the CrossGL expression ``expr``, or ``None`` when it is not an image."""
    return image_shape(
        _array_parts(generator, generator.expression_result_type(expr))[0]
    )


def _image_operand(generator, expr):
    """``(image, shape, reference)``: the Rust expression of an image (or of an element of an image array), its
    shape, and an expression of a reference to it, or ``(None, None, None)``."""
    shape = _operand_shape(generator, expr)
    if shape is None:
        return None, None, None
    if isinstance(expr, ArrayAccessNode):
        array_type = generator.expression_result_type(expr.array_expr)
        sizes = _array_parts(generator, array_type)[1]
        array = generator.generate_expression(expr.array_expr)
        index = generator.generate_expression(expr.index_expr)
        if sizes == [None]:
            element = f"unsafe {{ {array}.index(({index}) as usize) }}"
            return element, shape, element
        element = f"{array}[({index}) as usize]"
        return element, shape, f"&{element}"
    image = generator.generate_expression(expr)
    # Entry-point and helper parameters hold images by reference.
    return image, shape, image


def _sampler_operand(generator, expr):
    """The ``Sampler`` value of a sampler reference (``*s``) or of an element of a sampler array (``s[i]``)."""
    if isinstance(expr, ArrayAccessNode):
        return generator.generate_expression(expr)
    if isinstance(expr, IdentifierNode):
        return f"*{generator.generate_expression(expr)}"
    return None


def _is_sampler(generator, expr):
    base = _array_parts(generator, generator.expression_result_type(expr))[0]
    return base in SAMPLER_TYPE_NAMES


def _refuse(generator, name, reason):
    return generator.refuse_lowering(
        f"texture operation: '{name}' {reason}", "Default::default()"
    )


def _integer(generator, expr, rust_type="u32"):
    return f"({generator.generate_expression(expr)}) as {rust_type}"


def _float_vector(size):
    return "f32" if size == 1 else f"Vec{size}"


_LANES = "xyzw"


_SHADOW_METHODS = {
    ("texture", 0): "sample_depth_reference",
    ("textureLod", 1): "sample_depth_reference_by_lod",
    ("textureGrad", 2): "sample_depth_reference_by_gradient",
}


def _packed_shadow_call(generator, name, image, shape, sampler, arguments):
    """``texture``, ``textureLod`` or ``textureGrad`` of a shadow sampler, whose coordinate carries the depth
    reference: the third component for 1D, 1D-array and 2D shadow samplers, the fourth for 2D-array and cube
    ones, and a separate operand for a cube-map array (GLSL 4.60 section 8.9.2). spirv-std has no biased
    depth-reference sample, so ``texture`` with a bias is refused."""
    if shape.dim == "cube" and shape.arrayed:
        if len(arguments) < 2:
            return None
        coordinate = generator.generate_expression(arguments[0])
        reference = generator.generate_expression(arguments[1])
        rest = arguments[2:]
        method = _SHADOW_METHODS.get((name, len(rest)))
        if method is None:
            return None
        operands = ", ".join(generator.generate_expression(arg) for arg in rest)
        return f"{image}.{method}({sampler}, {coordinate}, {reference}{', ' if operands else ''}{operands})"
    rest = arguments[1:]
    method = _SHADOW_METHODS.get((name, len(rest)))
    count = shape.coordinates
    info = generator.vector_type_info(generator.expression_result_type(arguments[0]))
    if method is None or (info or {}).get("size") != max(count + 1, 3):
        return None
    point = "xyz"[:count]
    point = f"crosstl_p.{point}" if count == 1 else f"crosstl_p.{point}()"
    reference = "crosstl_p.z" if count + 1 <= 3 else "crosstl_p.w"
    operands = "".join(", " + generator.generate_expression(arg) for arg in rest)
    coordinate = generator.generate_expression(arguments[0])
    # The block reads the coordinate once, so a coordinate with side effects runs once.
    return (
        f"{{ let crosstl_p = {coordinate}; "
        f"{image}.{method}({sampler}, {point}, {reference}{operands}) }}"
    )


def _sample_call(generator, name, image, shape, sampler, arguments):
    """A sampling read with an implicit level of detail, a bias, an explicit level or explicit gradients."""
    if shape.depth:
        return _packed_shadow_call(generator, name, image, shape, sampler, arguments)
    coordinate = generator.generate_expression(arguments[0])
    rest = [generator.generate_expression(arg) for arg in arguments[1:]]
    if name == "texture":
        if len(rest) == 1:
            return f"{image}.sample_bias({sampler}, {coordinate}, {rest[0]})"
        return f"{image}.sample({sampler}, {coordinate})" if not rest else None
    if name == "textureLod" and len(rest) == 1:
        return f"{image}.sample_by_lod({sampler}, {coordinate}, {rest[0]})"
    if name == "textureGrad" and len(rest) == 2:
        return (
            f"{image}.sample_by_gradient({sampler}, {coordinate}, {rest[0]}, {rest[1]})"
        )
    return None


def _compare_call(generator, name, image, shape, sampler, arguments):
    """A depth comparison: the coordinate, the reference, then the level or the gradients."""
    method, count = _COMPARE_METHODS[name]
    if not shape.depth or len(arguments) != count:
        return None
    if "Proj" in name and (shape.arrayed or shape.dim not in {"1D", "2D"}):
        return None
    operands = ", ".join(generator.generate_expression(arg) for arg in arguments)
    return f"{image}.{method}({sampler}, {operands})"


def _projective_call(generator, name, image, shape, sampler, arguments):
    """A projective read: the coordinate's last component divides the others (GLSL 4.60 section 8.9.2)."""
    if shape.depth or shape.arrayed or shape.dim not in {"1D", "2D", "3D"}:
        return None
    coordinate = generator.generate_expression(arguments[0])
    info = generator.vector_type_info(generator.expression_result_type(arguments[0]))
    width = (info or {}).get("size", 1)
    rest = [generator.generate_expression(arg) for arg in arguments[1:]]
    count = shape.coordinates
    if width == count + 1:
        projected = coordinate
    elif width == 4:
        # textureProj(sampler1D/2D, vec4 P) reads (P.x[, P.y]) / P.w: spirv-std takes (s[, t], q).
        projected = f"({coordinate}).{'xw' if count == 1 else 'xyw'}()"
    else:
        return None
    args = f"{sampler}, {projected}"
    method = "sample_with_project_coordinate"
    if name == "textureProj":
        if len(rest) == 1:
            method += "_with"
            args += f", spirv_std::image::sample_with::bias({rest[0]})"
        elif rest:
            return None
    elif name == "textureProjLod" and len(rest) == 1:
        method += "_by_lod"
        args += f", {rest[0]}"
    elif name == "textureProjGrad" and len(rest) == 2:
        method += "_by_gradient"
        args += f", {rest[0]}, {rest[1]}"
    else:
        return None
    return f"{image}.{method}({args})"


def _gather_call(generator, image, shape, sampler, arguments):
    if shape.depth or shape.dim not in {"2D", "cube"}:
        return None
    coordinate = generator.generate_expression(arguments[0])
    component = 0
    if len(arguments) == 2:
        literal = arguments[1]
        if not isinstance(literal, LiteralNode) or not isinstance(literal.value, int):
            return None
        component = literal.value
    elif len(arguments) > 2:
        return None
    if component not in range(4):
        return None
    return f"{image}.gather({sampler}, {coordinate}, {component})"


def _sampled_call(generator, name, arguments):
    image, shape, _reference = _image_operand(generator, arguments[0])
    if image is None:
        return None
    if not shape.needs_sampler:
        return _refuse(
            generator, name, "samples an image that is not a sampled texture"
        )
    if len(arguments) < 3 or not _is_sampler(generator, arguments[1]):
        return _refuse(generator, name, "reads a texture without its sampler")
    sampler = _sampler_operand(generator, arguments[1])
    if sampler is None:
        return _refuse(generator, name, "takes its sampler from an expression")
    rest = list(arguments[2:])
    if name in _COMPARE_METHODS:
        call = _compare_call(generator, name, image, shape, sampler, rest)
    elif name in {"texture", "textureLod", "textureGrad"}:
        call = _sample_call(generator, name, image, shape, sampler, rest)
    elif name.startswith("textureProj"):
        call = _projective_call(generator, name, image, shape, sampler, rest)
    elif name == "textureGather":
        call = _gather_call(generator, image, shape, sampler, rest)
    elif name == "textureQueryLod" and len(rest) == 1 and not shape.depth:
        call = f"{image}.query_lod({sampler}, {generator.generate_expression(rest[0])})"
    else:
        call = None
    if call is None:
        return _refuse(
            generator,
            name,
            f"has no spirv-std method for these operands on a {shape.spelling()} image",
        )
    return call


def _size_call(image, shape, method, argument):
    count = _SIZE_COMPONENTS.get(shape.dim, 2) + int(shape.arrayed)
    query = f"{image}.{method}::<{'u32' if count == 1 else f'UVec{count}'}>({argument})"
    return f"({query} as i32)" if count == 1 else f"{query}.as_ivec{count}()"


def _has_levels(shape):
    return shape.sampled and shape.dim != "buffer" and not shape.multisampled


def _unsigned_size_call(image, shape, level):
    count = _SIZE_COMPONENTS.get(shape.dim, 2) + int(shape.arrayed)
    size = "u32" if count == 1 else f"UVec{count}"
    if level is None:
        return f"{image}.query_size::<{size}>()"
    return f"{image}.query_size_lod::<{size}>({level})"


def _widened_read(read, shape):
    """A read of ``shape``'s components, widened to the four CrossGL's ``imageLoad`` returns."""
    count = shape.components
    if count == 4:
        return read
    scalar = shape.scalar
    vector = _GLAM_VECTORS[scalar]
    value_type = scalar if count == 1 else f"{vector}{count}"
    lanes = ["__t"] if count == 1 else [f"__t.{lane}" for lane in _LANES[:count]]
    lanes += [_ZERO[scalar]] * (3 - count) + [_ONE[scalar]]
    return f"{{ let __t: {value_type} = {read}; {vector}4::new({', '.join(lanes)}) }}"


def _narrowed_texel(value, shape):
    count = shape.components
    if count == 4:
        return value
    return f"({value}).x" if count == 1 else f"({value}).{_LANES[:count]}()"


def _image_only_call(generator, name, arguments):
    image, shape, _reference = _image_operand(generator, arguments[0])
    if image is None:
        return None
    rest = list(arguments[1:])
    generated = [generator.generate_expression(arg) for arg in rest]
    call = None
    if name == "texelFetch" and shape.sampled and not shape.depth:
        if shape.dim == "buffer" and len(rest) == 1:
            call = f"{image}.fetch({generated[0]})"
        elif shape.multisampled and len(rest) == 2:
            call = (
                f"{image}.fetch_with({generated[0]}, "
                f"spirv_std::image::sample_with::sample_index({generated[1]}))"
            )
        elif len(rest) == 2 and shape.dim != "buffer":
            call = f"{image}.fetch_with_lod({generated[0]}, {_integer(generator, rest[1])})"
    elif name in {"textureSize", "imageSize"}:
        if name == "imageSize" and shape.sampled:
            call = None
        elif (
            rest and shape.sampled and shape.dim != "buffer" and not shape.multisampled
        ):
            call = _size_call(
                image, shape, "query_size_lod", _integer(generator, rest[0])
            )
        elif not rest and (
            shape.multisampled or shape.dim == "buffer" or not shape.sampled
        ):
            call = _size_call(image, shape, "query_size", "")
    elif (
        name == "textureDimensions"
        and not shape.arrayed
        and len(rest) <= int(_has_levels(shape))
    ):
        # WGSL's textureDimensions: the unsigned size, of the given level (0 by default) when the image has levels.
        # WGSL leaves an array's layer count out of it, and CrossGL's textureSize includes it, so an arrayed image
        # is refused rather than given either meaning.
        level = _integer(generator, rest[0]) if rest else "0"
        call = _unsigned_size_call(image, shape, level if _has_levels(shape) else None)
    elif name == "textureQueryLevels" and not rest and shape.sampled:
        if shape.dim != "buffer" and not shape.multisampled:
            call = f"({image}.query_levels() as i32)"
    elif name in {"textureSamples", "imageSamples"} and not rest:
        if shape.multisampled and shape.sampled == (name == "textureSamples"):
            call = f"({image}.query_samples() as i32)"
    elif name in {"imageLoad", "imageStore"}:
        call = _storage_call(generator, name, image, shape, generated)
    elif name == "subpassLoad" and shape.dim == "subpass" and not rest:
        if not shape.multisampled:
            call = f"{image}.read_subpass(IVec2::ZERO)"
    if call is None:
        return _refuse(
            generator,
            name,
            f"has no spirv-std method for these operands on a {shape.spelling()} image",
        )
    return call


def _storage_call(generator, name, image, shape, generated):
    if shape.sampled or shape.multisampled or shape.dim == "subpass":
        return None
    if name == "imageLoad" and len(generated) == 1:
        return _widened_read(f"{image}.read({generated[0]})", shape)
    if name == "imageStore" and len(generated) == 2:
        texel = _narrowed_texel(generated[1], shape)
        return f"unsafe {{ {image}.write({generated[0]}, {texel}) }}"
    return None


# The SPIR-V instruction and image operands of each read spirv-std's methods do not take for 1D-array images (its
# ImageCoordinate has no 1D-array form) and texel buffers (its fetch adds a Lod operand, which a buffer has none of).
# Each entry: the instruction, whether it needs a sampler, the depth reference's position (or None), and the image
# operand of the operands after the coordinate.
_INSTRUCTIONS = {
    "texture": (
        ("OpImageSampleImplicitLod", True, None, ""),
        ("OpImageSampleImplicitLod", True, None, "Bias"),
    ),
    "textureLod": (None, ("OpImageSampleExplicitLod", True, None, "Lod")),
    "textureGrad": (None, None, ("OpImageSampleExplicitLod", True, None, "Grad")),
    "textureCompare": (None, ("OpImageSampleDrefImplicitLod", True, 0, "")),
    "textureCompareLod": (None, None, ("OpImageSampleDrefExplicitLod", True, 0, "Lod")),
    "textureCompareGrad": (
        None,
        None,
        None,
        ("OpImageSampleDrefExplicitLod", True, 0, "Grad"),
    ),
    "texelFetch": (
        ("OpImageFetch", False, None, ""),
        ("OpImageFetch", False, None, "Lod"),
    ),
    "imageLoad": (("OpImageRead", False, None, ""),),
}


def _instruction_operands(generator, name, shape, rest):
    """The instruction entry and the operand values after the coordinate, or ``None``."""
    forms = _INSTRUCTIONS.get(name, ())
    count = len(rest) - 1
    if count < 0 or count >= len(forms) or forms[count] is None:
        return None
    instruction, needs_sampler, dref, operand = forms[count]
    if name == "texelFetch" and (shape.dim == "buffer") != (operand == ""):
        return None
    if name == "imageLoad" and (shape.sampled or shape.multisampled):
        return None
    if name != "imageLoad" and (not shape.sampled or shape.depth != (dref is not None)):
        return None
    values = [generator.generate_expression(arg) for arg in rest]
    if name == "texelFetch" and operand == "Lod":
        values[-1] = _integer(generator, rest[-1], "i32")
    return instruction, needs_sampler, dref, operand, values


def _instruction_call(generator, name, arguments):
    """A read of a 1D-array image or a texel buffer, as the SPIR-V instruction itself (rust-gpu inline assembly)."""
    _image, shape, reference = _image_operand(generator, arguments[0])
    rest = list(arguments[1:])
    sampler = None
    if name in _SAMPLER_CALLS:
        if not rest or not _is_sampler(generator, rest[0]):
            return _refuse(generator, name, "reads a texture without its sampler")
        sampler = _sampler_operand(generator, rest[0])
        rest = rest[1:]
    if name == "imageStore" and len(rest) == 2 and not shape.sampled:
        return _inline_write(generator, reference, rest)
    parts = _instruction_operands(generator, name, shape, rest)
    if parts is None or (parts[1] and sampler is None):
        return _refuse(
            generator,
            name,
            f"has no rust-gpu lowering for these operands on a {shape.spelling()} image",
        )
    instruction, _needs_sampler, dref, operand, values = parts
    zero = "0.0f32" if dref is not None else f"{_GLAM_VECTORS[shape.scalar]}4::ZERO"
    names = ["coordinate"] + (["dref"] if dref is not None else [])
    names += {"": [], "Bias": ["bias"], "Lod": ["lod"], "Grad": ["dx", "dy"]}[operand]
    lets = [f"let __{n} = {v};" for n, v in zip(names, values)]
    loads = [f'"%{n} = OpLoad _ {{{n}}}",' for n in names]
    registers = [f"{n} = in(reg) &__{n}," for n in names]
    source = "%image"
    if sampler is not None:
        lets.append(f"let __sampler = {sampler};")
        loads += [
            '"%sampler = OpLoad _ {sampler}",',
            '"%sampled = OpSampledImage _ %image %sampler",',
        ]
        registers.append("sampler = in(reg) &__sampler,")
        source = "%sampled"
    tail = " ".join(f"%{n}" for n in names[1 : 2 if dref is not None else 1])
    image_operands = f"{operand} " + " ".join(
        f"%{n}" for n in names[1 + (dref is not None) :]
    )
    instruction_text = f"%result = {instruction} typeof*{{result}} {source} %coordinate"
    if tail:
        instruction_text += f" {tail}"
    if operand:
        instruction_text += f" {image_operands}"
    lines = [
        '"%image = OpLoad _ {image}",',
        *loads,
        f'"{instruction_text}",',
        '"OpStore {result} %result",',
    ]
    registers = [
        f"image = in(reg) {reference},",
        *registers,
        "result = in(reg) &mut __result,",
    ]
    return (
        f"{{ {' '.join(lets)} let mut __result = {zero}; unsafe {{ asm!({' '.join(lines)} "
        f"{' '.join(registers)}) }}; __result }}"
    )


def _inline_write(generator, reference, rest):
    coordinate, texel = (generator.generate_expression(arg) for arg in rest)
    return (
        f"{{ let __coordinate = {coordinate}; let __texel = {texel}; unsafe {{ asm!("
        '"%image = OpLoad _ {image}", "%coordinate = OpLoad _ {coordinate}", "%texel = OpLoad _ {texel}", '
        '"OpImageWrite %image %coordinate %texel", '
        f"image = in(reg) {reference}, coordinate = in(reg) &__coordinate, texel = in(reg) &__texel) }} }}"
    )


def generate_rust_gpu_image_call(generator, name, arguments):
    """The spirv-std method call of a CrossGL image built-in, or ``None`` when ``name`` is not one on an image."""
    if name not in RUST_GPU_IMAGE_CALLS or generator.is_user_defined_function(name):
        return None
    arguments = list(arguments or [])
    if not arguments:
        return None
    shape = _operand_shape(generator, arguments[0])
    if shape is None:
        return None
    if name in _OFFSET_CALLS:
        return _refuse(
            generator,
            name,
            "reads with a texel offset, and spirv-std's image operands have no Offset",
        )
    if name in _UNLOWERED_CALLS:
        return _refuse(generator, name, "has no spirv-std method (OpImageDrefGather)")
    if (shape.dim == "1D" and shape.arrayed) or (
        shape.dim == "buffer" and name == "texelFetch"
    ):
        return _instruction_call(generator, name, arguments)
    if name in _SAMPLER_CALLS:
        return _sampled_call(generator, name, arguments)
    return _image_only_call(generator, name, arguments)


def rust_gpu_image_result_type(generator, name, arguments):
    """The CrossGL type of a CrossGL image built-in call on an image, else ``None``."""
    if name not in RUST_GPU_IMAGE_CALLS or generator.is_user_defined_function(name):
        return None
    arguments = list(arguments or [])
    if not arguments:
        return None
    type_name = generator.expression_result_type(arguments[0])
    shape = image_shape(_array_parts(generator, type_name)[0])
    if shape is None:
        return None
    if name == "imageStore":
        return "void"
    if name in {"textureQueryLevels", "textureSamples", "imageSamples"}:
        return "int"
    if name == "textureQueryLod":
        return "vec2"
    if name in {"textureSize", "imageSize", "textureDimensions"}:
        count = _SIZE_COMPONENTS.get(shape.dim, 2) + int(shape.arrayed)
        prefix = "u" if name == "textureDimensions" else "i"
        return (
            ("uint" if prefix == "u" else "int")
            if count == 1
            else f"{prefix}vec{count}"
        )
    if name in _COMPARE_METHODS or (
        shape.depth and name in {"texture", "textureLod", "textureGrad"}
    ):
        return "float"
    return f"{_CROSSGL_VECTORS[shape.scalar]}4"


_IMPORTS = (
    (re.compile(r"\bImage!\("), "use spirv_std::Image;\n"),
    (re.compile(r"&(?:\[)?Sampler\b"), "use spirv_std::Sampler;\n"),
    (re.compile(r"\bRuntimeArray<"), "use spirv_std::RuntimeArray;\n"),
    (
        re.compile(
            r"\.(?:sample_depth_reference_with|sample_with_project_coordinate_with|fetch_with)\("
        ),
        "use spirv_std::image::ImageWithMethods;\n",
    ),
    (re.compile(r"\basm!\("), "use core::arch::asm;\n"),
)


def add_rust_gpu_image_imports(code):
    """``code`` with the spirv-std image imports its image types and methods need, after ``use spirv_std::spirv;``."""
    imports = "".join(line for pattern, line in _IMPORTS if pattern.search(code))
    anchor = "use spirv_std::spirv;\n"
    if not imports or anchor not in code:
        return code
    if "use core::arch::asm;" in imports:
        # rust-gpu's SPIR-V inline assembly is an experimental asm! architecture.
        code = code.replace(
            "#![no_std]\n", "#![no_std]\n#![feature(asm_experimental_arch)]\n", 1
        )
    return code.replace(anchor, anchor + imports, 1)
