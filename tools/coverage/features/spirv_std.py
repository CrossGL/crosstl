"""spirv_std's image types, ``Image`` and ``SampledImage``, through whose methods a rust-gpu shader samples, fetches,
reads, writes and queries textures, its ``ByteAddressableBuffer``, through which it loads and stores words of an
untyped storage buffer, its ``RuntimeArray``, an array of unknown length, and its ``debug_printf!`` macros.

spirv_std's ``CooperativeMatrix`` is not read here. No feature list measures cooperative matrices yet, in any language,
and they are to be added to all of them at once, so that the CrossGL IR row compares like with like. rust-gpu 0.11.0
also does not compile ``CooperativeMatrix::length`` ("cannot use a dynamic value as an instruction type").

``image.rs`` gives ``Image`` its methods in ``impl`` blocks that fix some of the type's constant parameters, such as
a sampled or a storage image, and leave the others generic. A ``where Self: HasGather`` bound limits a method further,
to the images its trait is implemented for. A method is probed on the 2D image that its block and its bounds allow
(:func:`_image`): a sampled ``f32`` image unless they fix otherwise, multisampled where the first 2D image a bound
allows is, and a depth image for a method that takes a depth reference, as GLSL's shadow samplers are. ``read`` and
``write`` are defined for storage images and again for images of unknown sampling, which Vulkan does not have (it
requires a Sampled operand of 1 or 2), so each method counts once by name, from the first block that has it.

The probes declare the image and its sampler as resources of the entry point. A method that samples with an implicit
level of detail, or queries one, needs derivatives, which only a fragment shader has in Vulkan, so it is probed in
one. CrossGL samples GLSL's combined samplers, so the CrossGL probe of each method calls the GLSL function
:data:`COUNTERPARTS` gives, which takes the same inputs in the same order.
"""

from __future__ import annotations

import re

from ..model import STRING, VOID, Feature, Param, Signature, Type
from ..spellings import SUBSCRIPT
from .common import prototype
from .rust import enclosed, header, separated

# The GLSL function each method computes. A projective depth comparison holds its depth reference in the third
# component of GLSL's coordinate and the divisor in the fourth (GLSL 4.60, section 8.9.2), where spirv_std takes the
# depth reference after a coordinate whose last component is the divisor; :data:`PROJECTIVE_DEPTH` reads both from
# one vector in GLSL's order. ``const`` marks the level of detail 0 that ``fetch`` reads.
COUNTERPARTS = {
    "Image::fetch": "vec4 texelFetch(sampler2D, ivec2, const int)",
    "Image::fetch_with_lod": "vec4 texelFetch(sampler2D, ivec2, int)",
    "Image::gather": "vec4 textureGather(sampler2D, vec2)",
    "Image::sample": "vec4 texture(sampler2D, vec2)",
    "Image::sample_bias": "vec4 texture(sampler2D, vec2, float)",
    "Image::sample_by_lod": "vec4 textureLod(sampler2D, vec2, float)",
    "Image::sample_by_gradient": "vec4 textureGrad(sampler2D, vec2, vec2, vec2)",
    "Image::sample_depth_reference": "float texture(sampler2DShadow, vec3)",
    "Image::sample_depth_reference_by_lod": (
        "float textureLod(sampler2DShadow, vec3, float)"
    ),
    "Image::sample_depth_reference_by_gradient": (
        "float textureGrad(sampler2DShadow, vec3, vec2, vec2)"
    ),
    "Image::sample_with_project_coordinate": "vec4 textureProj(sampler2D, vec3)",
    "Image::sample_with_project_coordinate_by_lod": (
        "vec4 textureProjLod(sampler2D, vec3, float)"
    ),
    "Image::sample_with_project_coordinate_by_gradient": (
        "vec4 textureProjGrad(sampler2D, vec3, vec2, vec2)"
    ),
    "Image::sample_depth_reference_with_project_coordinate": (
        "float textureProj(sampler2DShadow, vec4)"
    ),
    "Image::sample_depth_reference_with_project_coordinate_by_lod": (
        "float textureProjLod(sampler2DShadow, vec4, float)"
    ),
    "Image::sample_depth_reference_with_project_coordinate_by_gradient": (
        "float textureProjGrad(sampler2DShadow, vec4, vec2, vec2)"
    ),
    "Image::read": "vec4 imageLoad(image2D, ivec2)",
    "Image::write": "void imageStore(image2D, ivec2, vec4)",
    "Image::read_subpass": "vec4 subpassLoad(subpassInput)",
    "Image::query_levels": "int textureQueryLevels(sampler2D)",
    "Image::query_lod": "vec2 textureQueryLod(sampler2D, vec2)",
    "Image::query_size": "ivec2 textureSize(sampler2DMS)",
    "Image::query_size_lod": "ivec2 textureSize(sampler2D, int)",
    "Image::query_samples": "int textureSamples(sampler2DMS)",
    "SampledImage::sample": "vec4 texture(sampler2D, vec2)",
    "SampledImage::sample_by_lod": "vec4 textureLod(sampler2D, vec2, float)",
    "SampledImage::query_size_lod": "ivec2 textureSize(sampler2D, int)",
    "SampledImage::query_size": "ivec2 textureSize(sampler2DMS)",
}
# The opaque argument :class:`..kernels.Rust` makes for a projective coordinate and the depth reference it is compared
# with, from one vector in GLSL's order.
PROJECTIVE_DEPTH = "ProjectiveDepth"
# The probe type of each type a method of a 2D ``f32`` image with four components takes or returns.
_TYPES = {
    "SampledType::SampleResult": Type("float", (4,)),
    "SampledType::Vec4": Type("float", (4,)),
    "SampledType::Vec2": Type("float", (2,)),
    "SampledType": Type("float"),
    "Sampler": Type.other("Sampler"),
    "f32": Type("float"),
    "u32": Type("uint"),
}
# The image a size query returns the size of has two dimensions.
_SIZE = Type("uint", (2,))
_COORDINATE = re.compile(r"impl\s+ImageCoordinate(SubpassData)?<\s*(\w+)\s*,(.*)>")
# A constant parameter an impl block or a trait implementation fixes, as ``{ Sampled::Yes as u32 }``.
_FIXED = re.compile(r"\{\s*(\w+)::(\w+)\s+as\s+u32\s*\}")
_IMPL = re.compile(r"^impl\s*<", re.M)
_METHOD = re.compile(r"^    pub (unsafe )?fn (\w+)", re.M)
# A string literal; one continues on the next line after a backslash, as ray_tracing.rs's OpTraceRayKHR does.
_STRING = re.compile(r'"(?:\\.|[^"\\])*"', re.S)
# The instructions that take an implicit level of detail, or compute one.
_DERIVATIVES = re.compile(r"\bOpImage\w*ImplicitLod\b|\bOpImageQueryLod\b")
_DIMENSIONS = {"TwoD": "2D", "SubpassData": "subpass"}


def clean(text: str) -> str:
    """``text`` without comments, and with the punctuation of its string literals blanked: the ``asm!`` strings hold
    braces that need not balance, and name the instructions a method executes."""
    text = re.sub(r"//[^\n]*", "", text)
    return _STRING.sub(lambda m: re.sub(r"[^\w\s]", " ", m.group(0)), text)


def _opening(text: str, position: int) -> int:
    """The index of the first brace from ``position`` that is outside angle brackets: the one that opens a block. A
    brace inside them holds a constant, as in ``Image<SampledType, { Sampled::Yes as u32 }>``.
    """
    depth = 0
    while depth or text[position] != "{":
        if text[position] == "{":
            _, position = enclosed(text, position)
            continue
        if text[position] == "<":
            depth += 1
        elif text[position] == ">" and text[position - 1] != "-":
            depth -= 1
        position += 1
    return position


def _blocks(text: str) -> list[tuple[str, str]]:
    """``(header, body)`` of each ``impl`` block: what follows its generic parameters, up to the block, and the
    block."""
    blocks = []
    for match in _IMPL.finditer(text):
        _, start = enclosed(text, match.end() - 1)
        opening = _opening(text, start)
        body, _ = enclosed(text, opening)
        blocks.append((text[start:opening].strip(), body))
    return blocks


def _image(fixed: dict[str, str], depth: bool) -> str:
    """The ``Image!`` spelling of the 2D image whose constant parameters ``fixed`` gives, or empty when no probe image
    has them."""
    dimension = _DIMENSIONS.get(fixed.get("Dimensionality", "TwoD"))
    sampled = fixed.get("Sampled", "Yes")
    if dimension is None or sampled == "Unknown" or fixed.get("Arrayed") == "True":
        return ""
    if sampled == "No":
        # A storage image declares its format; a subpass input's is its attachment's.
        parts = [
            "type=f32" if dimension == "subpass" else "format=rgba32f",
            "sampled=false",
        ]
    else:
        parts = ["type=f32", "sampled"]
    if depth:
        parts.append("depth")
    if fixed.get("Multisampled") == "True":
        parts.append("multisampled")
    return f"Image!({', '.join([dimension, *parts])})"


def _bounds(blocks: list[tuple[str, str]]) -> dict[str, list[dict[str, str]]]:
    """The constant parameters each image trait's implementations fix, in source order, for the 2D images."""
    bounds: dict[str, list[dict[str, str]]] = {}
    for head, _ in blocks:
        implemented = re.match(r"(\w+)\s+for\s+Image\b", head)
        if implemented:
            fixed = dict(_FIXED.findall(head))
            if fixed.get("Dimensionality") == "TwoD":
                bounds.setdefault(implemented.group(1), []).append(fixed)
    return bounds


def _allowed(
    fixed: dict[str, str], traits: list[str], bounds: dict[str, list[dict[str, str]]]
) -> dict[str, str] | None:
    """``fixed`` with what the first implementation of each trait in ``traits`` that agrees with it fixes, or None
    when one has no such implementation."""
    for trait in traits:
        agreeing = [
            implementation
            for implementation in bounds.get(trait, ())
            if all(
                fixed.get(key, value) == value for key, value in implementation.items()
            )
        ]
        if not agreeing:
            return None
        fixed = {**fixed, **agreeing[0]}
    return fixed


def _type(spelling: str, generics: str) -> Type:
    spelling = spelling.strip()
    coordinate = _COORDINATE.fullmatch(spelling)
    if coordinate:
        subpass, scalar, rest = coordinate.groups()
        floating = scalar == "f32" or re.search(rf"\b{scalar}\s*:\s*Float\b", generics)
        # An arrayed coordinate is a projective one here: its last component is the divisor.
        size = 3 if "Arrayed::True" in rest else 2
        # Integer coordinates are signed, as GLSL's are.
        return Type("float" if floating and not subpass else "int", (size,))
    if re.fullmatch(r"\w+", spelling) and re.search(
        rf"\b{spelling}\s*:\s*ImageSizeQuery\b", generics
    ):
        return _SIZE
    return _TYPES.get(spelling, Type.other(spelling))


def _signature(parsed: tuple[str, str, str], receiver: str) -> Signature:
    """The probe signature of a method on the image ``receiver``."""
    generics, parameters, result = parsed
    params: list[Param] = []
    for part in separated(parameters):
        if re.fullmatch(r"&\s*self", part):
            params.append(Param(Type.other(receiver)))
            continue
        label, _, spelling = part.partition(":")
        label = label.strip()
        if label == "depth_reference" and params[-1].type.shape == (3,):
            params[-1] = Param(Type.other(PROJECTIVE_DEPTH))
        elif label == "component":
            # A probe gathers the first component, as GLSL's textureGather does without one; Vulkan requires it to be
            # a constant (VUID-StandaloneSpirv-OpImageGather-04664). The literal is the u32 the method takes.
            params.append(Param(Type("int"), "const"))
        elif "SubpassData" in spelling:
            # A subpass input is read at the fragment's own position, as GLSL's subpassLoad reads it.
            params.append(Param(_type(spelling, generics), "const"))
        else:
            params.append(Param(_type(spelling, generics)))
    returned = _type(result, generics) if result else VOID
    return Signature(returned, tuple(params))


def images(text: str) -> list[Feature]:
    """The methods of ``Image`` and ``SampledImage`` in ``image.rs``, each named ``Type::method``."""
    blocks = _blocks(clean(text))
    bounds = _bounds(blocks)
    features: list[Feature] = []
    for head, body in blocks:
        owner = re.match(r"(Image|SampledImage)\s*<", head)
        if owner is None:
            continue
        fixed = dict(_FIXED.findall(head))
        for match in _METHOD.finditer(body):
            name = f"{owner.group(1)}::{match.group(2)}"
            if any(feature.name == name for feature in features):
                continue
            _, end = enclosed(body, body.index("(", match.end()))
            opening = _opening(body, end)
            method, _ = enclosed(body, opening)
            allowed = _allowed(
                fixed, re.findall(r"\b(Has\w+)\b", body[end:opening]), bounds
            )
            features.append(
                _feature(
                    name,
                    bool(match.group(1)),
                    header(body, match.end()),
                    allowed,
                    owner.group(1),
                    method,
                )
            )
    for head, body in blocks:
        if re.match(r"ImageWithMethods\s*<", head):
            features += _with_features(body)
    return sorted(features, key=lambda f: f.name)


# ImageWithMethods gives an image the ``*_with`` forms of its methods, which take the image operands that the functions
# of ``image/sample_with.rs`` make: a bias, a level of detail, gradients or a sample index. A form is one feature per
# operand that SPIR-V lets its instruction take (SPIR-V 1.6, section 3.14, Image Operands): Bias with an implicit
# level of detail, Lod and Grad with an explicit one, Lod also with OpImageFetch, and Sample with OpImageFetch from a
# multisampled image. ``gather_with`` has none: OpImageGather takes none of them without
# SPV_AMD_texture_gather_bias_lod, and without operands it is ``gather``.
_WITH = re.compile(r"^    fn (\w+_with)\b", re.M)
_OPERANDS = {
    "OpImageFetch": ("lod", "sample_index"),
    "OpImageSample": ("bias", "lod", "grad"),
}
# The opaque argument :class:`..kernels.Rust` makes for each operand: a fetch takes an integer level of detail.
SAMPLE_PARAMS = "SampleParams({})"
_WITH_COUNTERPARTS = {
    ("fetch_with", "lod"): "vec4 texelFetch(sampler2D, ivec2, int)",
    ("fetch_with", "sample_index"): "vec4 texelFetch(sampler2DMS, ivec2, int)",
    ("sample_with", "bias"): "vec4 texture(sampler2D, vec2, float)",
    ("sample_with", "lod"): "vec4 textureLod(sampler2D, vec2, float)",
    ("sample_with", "grad"): "vec4 textureGrad(sampler2D, vec2, vec2, vec2)",
    (
        "sample_depth_reference_with",
        "bias",
    ): "float texture(sampler2DShadow, vec3, float)",
    (
        "sample_depth_reference_with",
        "lod",
    ): "float textureLod(sampler2DShadow, vec3, float)",
    (
        "sample_depth_reference_with",
        "grad",
    ): "float textureGrad(sampler2DShadow, vec3, vec2, vec2)",
    (
        "sample_with_project_coordinate_with",
        "bias",
    ): "vec4 textureProj(sampler2D, vec3, float)",
    (
        "sample_with_project_coordinate_with",
        "lod",
    ): "vec4 textureProjLod(sampler2D, vec3, float)",
    (
        "sample_with_project_coordinate_with",
        "grad",
    ): "vec4 textureProjGrad(sampler2D, vec3, vec2, vec2)",
    (
        "sample_depth_reference_with_project_coordinate_with",
        "bias",
    ): "float textureProj(sampler2DShadow, vec4, float)",
    (
        "sample_depth_reference_with_project_coordinate_with",
        "lod",
    ): "float textureProjLod(sampler2DShadow, vec4, float)",
    (
        "sample_depth_reference_with_project_coordinate_with",
        "grad",
    ): "float textureProjGrad(sampler2DShadow, vec4, vec2, vec2)",
}


def _with_features(body: str) -> list[Feature]:
    """The ``*_with`` methods of the ImageWithMethods implementation ``body``, one feature per image operand."""
    features = []
    for match in _WITH.finditer(body):
        _, end = enclosed(body, body.index("(", match.end()))
        method, _ = enclosed(body, _opening(body, end))
        instruction = re.search(r"\b(OpImageFetch|OpImageSample)", method)
        parsed = header(body, match.end())
        depth = "depth_reference" in parsed[1]
        for operand in _OPERANDS[instruction.group(1)] if instruction else ():
            fixed = {"Multisampled": "True"} if operand == "sample_index" else {}
            found = _signature(parsed, _image(fixed, depth))
            kind = (
                "integer lod"
                if (operand, instruction.group(1)) == ("lod", "OpImageFetch")
                else operand
            )
            params = tuple(
                (
                    Param(Type.other(SAMPLE_PARAMS.format(kind)))
                    if p.type.opaque == "SampleParams"
                    else p
                )
                for p in found.params
            )
            spelled, counterpart = prototype(
                _WITH_COUNTERPARTS[(match.group(1), operand)]
            )
            features.append(
                Feature(
                    "rust",
                    "function",
                    f"Image::{match.group(1)}({operand})",
                    Signature(found.result, params),
                    group="image",
                    crossgl=spelled,
                    # A bias scales an implicit level of detail, which needs derivatives.
                    stage="fragment" if operand == "bias" else "compute",
                    spelling=f".{match.group(1)}",
                    counterpart=counterpart,
                )
            )
    return features


def _feature(
    name: str,
    unsafe: bool,
    parsed: tuple[str, str, str],
    fixed: dict[str, str] | None,
    owner: str,
    method: str,
) -> Feature:
    bare = name.rpartition("::")[2]
    depth = "depth_reference" in parsed[1]
    image = _image(fixed, depth) if fixed is not None else ""
    found = None
    if image:
        receiver = f"SampledImage<{image}>" if owner == "SampledImage" else image
        found = _signature(parsed, receiver)
    spelled, counterpart = "", None
    if name in COUNTERPARTS:
        spelled, counterpart = prototype(COUNTERPARTS[name])
    # Only a fragment shader has derivatives, and only it reads subpass inputs.
    fragment = _DERIVATIVES.search(method) or image.startswith("Image!(subpass")
    return Feature(
        "rust",
        "function",
        name,
        found,
        group="image",
        crossgl=spelled,
        stage="fragment" if fragment else "compute",
        note="" if image else "no probe image has the type the method is defined for",
        spelling=f"{'unsafe ' if unsafe else ''}.{bare}",
        counterpart=counterpart,
    )


# ByteAddressableBuffer views a storage buffer of u32 words, as HLSL's byte-address buffers are: a probe makes one from
# the storage buffer an entry point takes (:data:`..objects.UNIFORMS`), and loads or stores a u32 at a byte index
# that is four times a word index read from the input, since the index must be a multiple of 4. CrossGL declares
# the byte-address buffer itself and calls HLSL's Load and Store on it, which take the same byte index.
BYTE_BUFFER = "ByteAddressableBuffer<&[u32]>"
MUTABLE_BYTE_BUFFER = "ByteAddressableBuffer<&mut [u32]>"
# The opaque argument :class:`..kernels.Rust` makes for a byte index from a word index.
BYTE_INDEX = "ByteIndex"
_BYTE_BUFFERS = {"&'a [u32]": BYTE_BUFFER, "&'a mut [u32]": MUTABLE_BYTE_BUFFER}
_BYTE_METHODS = {
    "load": (
        "{}.Load({} * 4u)",
        "uint Load(ByteAddressBuffer, uint)",
    ),
    "store": (
        "{}.Store({} * 4u, {})",
        "void Store(RWByteAddressBuffer, uint, uint)",
    ),
}
# load_unchecked and store_unchecked skip the bounds check, which HLSL's Load and Store never make.
_BYTE_METHODS["load_unchecked"] = _BYTE_METHODS["load"]
_BYTE_METHODS["store_unchecked"] = _BYTE_METHODS["store"]
_BYTE_UNSPELLED = "CrossGL declares a byte-address buffer as a resource, and makes none from a storage buffer or from another"


def byte_buffers(text: str) -> list[Feature]:
    """The methods of ``ByteAddressableBuffer`` in ``byte_addressable_buffer.rs``, counted once by name."""
    text = clean(text)
    features: list[Feature] = []
    for match in re.finditer(r"^impl<'a> ByteAddressableBuffer<(.+)> \{", text, re.M):
        receiver = _BYTE_BUFFERS[match.group(1)]
        body, _ = enclosed(text, match.end() - 1)
        for method in _METHOD.finditer(body):
            bare = method.group(2)
            name = f"ByteAddressableBuffer::{bare}"
            if any(feature.name == name for feature in features):
                continue
            features.append(_byte_feature(name, bool(method.group(1)), receiver))
    return features


def _byte_feature(name: str, unsafe: bool, receiver: str) -> Feature:
    bare = name.rpartition("::")[2]
    word = Type("uint")
    params = [Param(Type.other(receiver)), Param(Type.other(BYTE_INDEX))]
    spelling = f"{'unsafe ' if unsafe else ''}.{bare}"
    if bare.startswith("load"):
        found = Signature(word, tuple(params))
    elif bare.startswith("store"):
        found = Signature(VOID, (*params, Param(word)))
    elif bare == "as_ref":
        found = Signature(Type.other(BYTE_BUFFER), (Param(Type.other(receiver)),))
    else:
        # from_slice and from_mut_slice take the storage buffer of words itself.
        words = "&[u32]" if receiver == BYTE_BUFFER else "&mut [u32]"
        found = Signature(Type.other(receiver), (Param(Type.other(words)),))
        spelling = f"spirv_std::{name}"
    spelled, counterpart = "", None
    if bare in _BYTE_METHODS:
        spelled, text = _BYTE_METHODS[bare]
        counterpart = prototype(text)[1]
    return Feature(
        "rust",
        "function",
        name,
        found,
        group="byte_addressable_buffer",
        crossgl=spelled,
        note="" if spelled else _BYTE_UNSPELLED,
        spelling=spelling,
        counterpart=counterpart,
    )


# RuntimeArray indexes an array of unknown length without a bounds check, and returns a reference to the element. A
# probe indexes a storage buffer of u32 words, which CrossGL declares as an unsized buffer and subscripts, and reads
# the word the reference refers to (:class:`..kernels.Rust` reads a ``&u32`` through ``*``).
RUNTIME_ARRAY = "RuntimeArray<u32>"
MUTABLE_RUNTIME_ARRAY = "&mut RuntimeArray<u32>"


def runtime_arrays(text: str) -> list[Feature]:
    """The methods of ``RuntimeArray`` in ``runtime_array.rs``."""
    text = clean(text)
    features: list[Feature] = []
    for match in re.finditer(r"^impl<T> RuntimeArray<T> \{", text, re.M):
        body, _ = enclosed(text, match.end() - 1)
        for method in _METHOD.finditer(body):
            bare = method.group(2)
            mutable = bare.endswith("_mut")
            receiver = Param(
                Type.other(MUTABLE_RUNTIME_ARRAY if mutable else RUNTIME_ARRAY)
            )
            reference = Type.other("&mut u32" if mutable else "&u32")
            features.append(
                Feature(
                    "rust",
                    "function",
                    f"RuntimeArray::{bare}",
                    Signature(reference, (receiver, Param(Type("usize")))),
                    group="runtime_array",
                    crossgl=SUBSCRIPT,
                    spelling=f"{'unsafe ' if method.group(1) else ''}.{bare}",
                    counterpart=Signature(
                        Type("uint"), (receiver, Param(Type("uint")))
                    ),
                )
            )
    return features


# The debug_printf! macros write a Vulkan debug printf (NonSemantic.DebugPrintf), as CrossGL's printf does. A probe
# prints the empty format string; debug_printfln! appends a newline to it.
_PRINTF = re.compile(r"^#\[proc_macro\]\npub fn (debug_printf\w*)\(", re.M)
_PRINTF_CROSSGL = {"debug_printf": "printf", "debug_printfln": 'printf("\\n")'}


def debug_printf(text: str) -> list[Feature]:
    """The ``debug_printf`` macros of spirv-std's macro crate, ``macros/src/lib.rs``."""
    return [
        Feature(
            "rust",
            "function",
            f"{match.group(1)}!",
            Signature(VOID, (Param(STRING, "const"),)),
            group="debug_printf",
            crossgl=_PRINTF_CROSSGL.get(match.group(1), ""),
            note="" if match.group(1) in _PRINTF_CROSSGL else "no CrossGL spelling",
            spelling=f"unsafe spirv_std::{match.group(1)}!",
        )
        for match in _PRINTF.finditer(text)
    ]
