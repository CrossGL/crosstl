"""Built-in type names the GLSL, HLSL and Metal generators write unchanged, and the refusal for every other one.

The importers carry the built-in types of the languages they read into the IR as plain names
(:data:`~crosstl.translator.semantics.builtin_type_names.BUILTIN_TYPE_NAMES`), and each generator maps the ones it
knows to its own spelling. A built-in name a generator had no mapping for used to be written unchanged: GLSL, HLSL
and Metal output declared ``Float16 v;``, ``__half v;`` or ``atomic_uint counter;``, which no compiler for the
target accepts.

A generator also maps type names it never writes: it asks for the spelling of a declaration that a later lowering
replaces (a ``texture2DShadow`` paired with a comparison sampler, a byte-address buffer bound as words) or of an
intermediate it discards. So the refusal is decided on the output. :func:`written_type_name` returns the name
unchanged and records it when it is a built-in type that neither the program nor the target declares, and
:func:`refuse_written_foreign_types` refuses each recorded name that the rendered program still uses as a type
identifier and does not declare itself.

Each set lists the plain type names one target defines, built from families as
:mod:`~crosstl.translator.semantics.builtin_type_names` is, and restricted to the spellings its compiler accepts:

- GLSL 4.60 §4.1 with ``GL_EXT_shader_explicit_arithmetic_types``, ``GL_EXT_bfloat16``, ``GL_KHR_vulkan_glsl``
  (``texture*``, ``sampler``, ``samplerShadow``, subpass inputs), ``GL_EXT_ray_query``, ``GL_EXT_ray_tracing``,
  ``GL_NV_ray_tracing``, ``GL_NV_shader_invocation_reorder`` and ``GL_OES_EGL_image_external``, checked with
  glslangValidator. ``hvec2`` is a reserved word, and a ``texture*Shadow`` type does not exist: depth comparison
  belongs to the sampler (``samplerShadow``).
- HLSL as DXC 1.9 compiles it for shader model 6.x: scalars, vectors and matrices of one to four components or rows
  ("Scalar Types", "Vector Type", "Matrix Type"), including the sized ``float16_t2`` and ``float32_t4x4``
  spellings, and the resource and system objects. There is no ``RWTextureCube`` and no rasterizer-ordered cube or
  multisampled texture; ``HitObject`` lives in the ``dx`` namespace.
- MSL 4.0 §2, checked with ``xcrun metal -std=metal3.2``: there is no ``double``, no ``bfloat`` matrix and no packed
  matrix, and ``auto`` is C++ type deduction.
"""

from __future__ import annotations

import re

from ..semantics.builtin_type_names import BUILTIN_TYPE_NAMES
from .lowering_refusals import refuse_unsupported

__all__ = [
    "CUDA_TYPE_NAMES",
    "GLSL_TYPE_NAMES",
    "HIP_TYPE_NAMES",
    "HLSL_TYPE_NAMES",
    "METAL_TYPE_NAMES",
    "sized_float_spelling",
    "sized_vector_parts",
    "refuse_written_foreign_types",
    "written_foreign_type_names",
    "written_type_name",
]

_COUNTS = (2, 3, 4)
_SHAPES = tuple(f"{c}x{r}" for c in _COUNTS for r in _COUNTS)


def _glsl():
    names = {"void", "bool", "int", "uint", "float", "double", "bfloat16_t"}
    for width in ("8", "16", "32", "64"):
        names.update((f"int{width}_t", f"uint{width}_t"))
    names.update(("float16_t", "float32_t", "float64_t"))
    vectors = ("vec", "ivec", "uvec", "bvec", "dvec", "f16vec", "f32vec", "f64vec")
    vectors += tuple(f"{sign}{width}vec" for sign in "iu" for width in (8, 16, 32, 64))
    for prefix in vectors:
        names.update(f"{prefix}{count}" for count in _COUNTS)
    for prefix in ("mat", "dmat", "f16mat", "f32mat", "f64mat"):
        names.update(f"{prefix}{count}" for count in _COUNTS)
        names.update(f"{prefix}{shape}" for shape in _SHAPES)
    dimensions = (
        "1D",
        "2D",
        "3D",
        "Cube",
        "2DRect",
        "Buffer",
        "1DArray",
        "2DArray",
        "CubeArray",
        "2DMS",
        "2DMSArray",
    )
    for kind in ("sampler", "texture", "image"):
        for prefix in ("", "i", "u"):
            names.update(f"{prefix}{kind}{dimension}" for dimension in dimensions)
    names.update(
        f"sampler{dimension}Shadow"
        for dimension in (
            "1D",
            "2D",
            "Cube",
            "2DRect",
            "1DArray",
            "2DArray",
            "CubeArray",
        )
    )
    for prefix in ("", "i", "u"):
        names.update((f"{prefix}subpassInput", f"{prefix}subpassInputMS"))
    names.update(
        (
            "sampler",
            "samplerShadow",
            "atomic_uint",
            "accelerationStructureEXT",
            "accelerationStructureNV",
            "rayQueryEXT",
            "hitObjectNV",
            "samplerExternalOES",
        )
    )
    return frozenset(names)


_HLSL_ELEMENTS = (
    "bool",
    "int",
    "uint",
    "dword",
    "half",
    "float",
    "double",
    "min16float",
    "min10float",
    "min16int",
    "min12int",
    "min16uint",
    "int16_t",
    "uint16_t",
    "int32_t",
    "uint32_t",
    "int64_t",
    "uint64_t",
    "float16_t",
    "float32_t",
    "float64_t",
)


def _hlsl():
    names = {"void", *_HLSL_ELEMENTS}
    counts = (1, *_COUNTS)
    for element in _HLSL_ELEMENTS:
        names.update(f"{element}{count}" for count in counts)
        names.update(
            f"{element}{rows}x{columns}" for rows in counts for columns in counts
        )
    names.update(
        (
            "SamplerState",
            "SamplerComparisonState",
            "Buffer",
            "RWBuffer",
            "ByteAddressBuffer",
            "RWByteAddressBuffer",
            "StructuredBuffer",
            "RWStructuredBuffer",
            "AppendStructuredBuffer",
            "ConsumeStructuredBuffer",
            "ConstantBuffer",
            "TextureBuffer",
            "RasterizerOrderedBuffer",
            "RasterizerOrderedByteAddressBuffer",
            "RasterizerOrderedStructuredBuffer",
            "RaytracingAccelerationStructure",
            "RayDesc",
            "RayQuery",
            "BuiltInTriangleIntersectionAttributes",
            "InputPatch",
            "OutputPatch",
            "PointStream",
            "LineStream",
            "TriangleStream",
            "FeedbackTexture2D",
            "FeedbackTexture2DArray",
        )
    )
    names.update(
        f"Texture{dimension}"
        for dimension in (
            "1D",
            "1DArray",
            "2D",
            "2DArray",
            "2DMS",
            "2DMSArray",
            "3D",
            "Cube",
            "CubeArray",
        )
    )
    writable = ("1D", "1DArray", "2D", "2DArray", "3D")
    names.update(
        f"RWTexture{dimension}" for dimension in (*writable, "2DMS", "2DMSArray")
    )
    names.update(f"RasterizerOrderedTexture{dimension}" for dimension in writable)
    return frozenset(names)


_METAL_ELEMENTS = (
    "bool",
    "char",
    "uchar",
    "short",
    "ushort",
    "int",
    "uint",
    "long",
    "ulong",
    "half",
    "float",
    "bfloat",
)


def _metal():
    names = {"void", "auto", "size_t", "ptrdiff_t", *_METAL_ELEMENTS}
    for width in ("8", "16", "32", "64"):
        names.update((f"int{width}_t", f"uint{width}_t"))
    for element in _METAL_ELEMENTS:
        names.update(f"{element}{count}" for count in _COUNTS)
        names.update(f"packed_{element}{count}" for count in _COUNTS)
    for element in ("half", "float"):
        names.update(f"{element}{shape}" for shape in _SHAPES)
    names.update(
        (
            "texture1d",
            "texture1d_array",
            "texture2d",
            "texture2d_array",
            "texture2d_ms",
            "texture2d_ms_array",
            "texture3d",
            "texturecube",
            "texturecube_array",
            "texture_buffer",
            "depth2d",
            "depth2d_array",
            "depth2d_ms",
            "depth2d_ms_array",
            "depthcube",
            "depthcube_array",
            "sampler",
            "atomic_int",
            "atomic_uint",
            "atomic_bool",
            "atomic_float",
            "atomic_ulong",
            "command_buffer",
            "compute_command",
            "render_command",
            "rasterization_rate_map_data",
            "rasterization_rate_map_decoder",
            # ``metal::raytracing`` (MSL §2.17), which the generator opens where it writes them.
            "acceleration_structure",
            "instance_acceleration_structure",
            "primitive_acceleration_structure",
            "intersection_function_table",
            "visible_function_table",
            "intersector",
            "intersection_query",
        )
    )
    return frozenset(names)


_C_VECTOR_ELEMENTS = (
    "char",
    "uchar",
    "short",
    "ushort",
    "int",
    "uint",
    "long",
    "ulong",
    "longlong",
    "ulonglong",
    "float",
    "double",
)


def _gpu_cxx(extra):
    """The plain type names a CUDA or HIP device compilation defines, plus the toolkit's own ``extra`` names.

    C++ fundamental and ``<cstdint>`` types, the built-in vector types of one to four components
    (CUDA C++ Programming Guide "Built-in Vector Types", HIP ``hip_vector_types.h``) and ``dim3``. ``uint`` and
    ``ushort`` are the ``<sys/types.h>`` aliases both toolkits' host headers bring in on Linux and macOS; the
    generators still write them for ``uint`` values.
    """
    names = {
        "void",
        "auto",
        "bool",
        "char",
        "short",
        "int",
        "long",
        "unsigned",
        "signed",
        "float",
        "double",
        "size_t",
        "ptrdiff_t",
        "uint",
        "ushort",
        "dim3",
        "half",
        "half2",
        "__half",
        "__half2",
        *extra,
    }
    for width in ("8", "16", "32", "64"):
        names.update((f"int{width}_t", f"uint{width}_t"))
    for element in _C_VECTOR_ELEMENTS:
        names.update(f"{element}{count}" for count in (1, *_COUNTS))
    return frozenset(names)


GLSL_TYPE_NAMES = _glsl()
HLSL_TYPE_NAMES = _hlsl()
METAL_TYPE_NAMES = _metal()
# ``cuda_fp16.h``, ``cuda_bf16.h`` and the runtime's texture, surface and driver pointer handles.
CUDA_TYPE_NAMES = _gpu_cxx(
    (
        "nv_bfloat16",
        "nv_bfloat162",
        "__nv_bfloat16",
        "__nv_bfloat162",
        "cudaTextureObject_t",
        "cudaSurfaceObject_t",
        "CUdeviceptr",
    )
)
# ``hip_fp16.h``, ``hip_bf16.h`` and the runtime's texture, surface and device pointer handles.
HIP_TYPE_NAMES = _gpu_cxx(
    (
        "hip_bfloat16",
        "__hip_bfloat16",
        "__hip_bfloat162",
        "hipTextureObject_t",
        "hipSurfaceObject_t",
        "hipDeviceptr_t",
    )
)


# A vector named by a sized scalar and a component count, as HLSL and CUDA-style sources write ``int16_t2`` or
# ``float32_t4``.
_SIZED_VECTOR = re.compile(
    r"(?P<component>(?:u?int(?:8|16|32|64)|float(?:16|32|64)|bfloat16)_t)(?P<lanes>[234])"
)


_SIZED_FLOAT = re.compile(r"float(?P<bits>16|32|64)_t(?P<shape>(?:[234](?:x[234])?)?)")
_FLOAT_BY_BITS = {"16": "half", "32": "float", "64": "double"}


def sized_float_spelling(name):
    """``name`` with a sized floating-point element (``float32_t3x3``) spelled by width (``float3x3``).

    HLSL writes ``float16_t``, ``float32_t`` and ``float64_t`` scalars, vectors and matrices; GLSL and MSL name the same
    types ``half``, ``float`` and ``double``. Any other name is returned unchanged.
    """
    match = _SIZED_FLOAT.fullmatch(name)
    if match is None:
        return name
    return f"{_FLOAT_BY_BITS[match.group('bits')]}{match.group('shape')}"


def sized_vector_parts(name):
    """``(component, lanes)`` of a sized-scalar vector name such as ``int16_t2``, or ``None`` for any other name."""
    match = _SIZED_VECTOR.fullmatch(name)
    if match is None:
        return None
    return match.group("component"), int(match.group("lanes"))


def written_type_name(name, target_type_names, declared_type_names, foreign_names=None):
    """``name`` as the target writes a type it has no mapping for, recorded in ``foreign_names`` when it is foreign.

    A name the program declares (a structure, enumeration, alias, constant block or generic parameter), a name the
    target defines and a name that is no built-in type of any language, which the importers and lowerings declare
    themselves, are not recorded. ``foreign_names`` is the set of the rendering in progress, or ``None`` outside one.
    """
    if (
        foreign_names is not None
        and name in BUILTIN_TYPE_NAMES
        and name not in target_type_names
        and name not in declared_type_names
    ):
        foreign_names.add(name)
    return name


# Comments and string literals, blanked before the output is searched.
_NON_CODE = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"', re.S)
_PREPROCESSOR_LINE = re.compile(r"^[ \t]*#[^\n]*", re.M)
_DEFINE = re.compile(r"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)", re.M)
_DECLARATIONS = (
    re.compile(
        r"\b(?:struct|class|union|enum|typename)\s+(?:class\s+|struct\s+)?([A-Za-z_]\w*)"
    ),
    re.compile(r"\busing\s+([A-Za-z_]\w*)\s*="),
    re.compile(r"\btypedef\b[^;{}]*?\b([A-Za-z_]\w*)\s*(?:\[[^\]]*\]\s*)*;"),
)
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
_QUALIFIER_BEFORE = re.compile(r"(?:\.|->|::)\s*$")


def written_foreign_type_names(code, foreign_names):
    """The names of ``foreign_names`` that ``code`` uses as unqualified identifiers and does not declare, sorted.

    A name in a comment, a string literal or a preprocessor line, a member or qualified name (``x.half``,
    ``dx::HitObject``) and a name the output declares as a structure, class, union, enumeration, template
    parameter, ``using`` or ``typedef`` alias or macro is not a use of the foreign built-in type.
    """
    if not foreign_names or not isinstance(code, str):
        return []
    text = _NON_CODE.sub(lambda match: re.sub(r"[^\n]", " ", match.group(0)), code)
    declared = set(_DEFINE.findall(text))
    text = _PREPROCESSOR_LINE.sub("", text)
    for pattern in _DECLARATIONS:
        declared.update(pattern.findall(text))
    candidates = set(foreign_names) - declared
    used = set()
    for match in _IDENTIFIER.finditer(text):
        name = match.group(0)
        if name in candidates and name not in used:
            if not _QUALIFIER_BEFORE.search(
                text[max(0, match.start() - 8) : match.start()]
            ):
                used.add(name)
    return sorted(used)


def refuse_written_foreign_types(code, foreign_names, target_label):
    """Refuse, in the rendering in progress, each foreign built-in type the rendered ``code`` still names."""
    for name in written_foreign_type_names(code, foreign_names):
        refuse_unsupported(
            target_label,
            "type",
            f"the built-in type '{name}' has no {target_label} spelling",
            name,
        )
