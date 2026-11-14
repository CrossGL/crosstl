"""Exact lookup tables of the SPIR-V generator.

Each table lists every spelling it accepts, the canonical one included, so that a lookup never passes an unknown
key through as its own value. Where a lookup only classifies a name, the key it returns is compared with exact
sets and never written to the module; the docstrings say which refusal catches a name no table knows.
"""

from __future__ import annotations

# Every spelling of each derivative built-in (GLSL, HLSL) and its SPIR-V instruction (SPIR-V, Derivative Instructions).
SPIRV_DERIVATIVE_OPCODES = {
    "dFdx": "OpDPdx",
    "ddx": "OpDPdx",
    "dFdxFine": "OpDPdxFine",
    "ddx_fine": "OpDPdxFine",
    "dFdxCoarse": "OpDPdxCoarse",
    "ddx_coarse": "OpDPdxCoarse",
    "dFdy": "OpDPdy",
    "ddy": "OpDPdy",
    "dFdyFine": "OpDPdyFine",
    "ddy_fine": "OpDPdyFine",
    "dFdyCoarse": "OpDPdyCoarse",
    "ddy_coarse": "OpDPdyCoarse",
    "fwidth": "OpFwidth",
    "fwidthFine": "OpFwidthFine",
    "fwidth_fine": "OpFwidthFine",
    "fwidthCoarse": "OpFwidthCoarse",
    "fwidth_coarse": "OpFwidthCoarse",
}

# Every spelling of each integer bit built-in and its GLSL name: HLSL countbits, reversebits, firstbitlow and
# firstbithigh are GLSL bitCount, bitfieldReverse, findLSB and findMSB (GLSL.std.450 FindILsb, FindSMsb/FindUMsb).
SPIRV_INTEGER_BIT_BUILTINS = {
    "bitCount": "bitCount",
    "countbits": "bitCount",
    "bitfieldReverse": "bitfieldReverse",
    "reversebits": "bitfieldReverse",
    "findLSB": "findLSB",
    "firstbitlow": "findLSB",
    "findMSB": "findMSB",
    "firstbithigh": "findMSB",
}

# HLSL spellings of GLSL built-ins the call dispatch knows by their GLSL name.
SPIRV_BUILTIN_CALL_ALIASES = {
    "frac": "fract",
    "inverseSqrt": "inversesqrt",
    "lerp": "mix",
    "rsqrt": "inversesqrt",
}

# Every memory or access qualifier a resource may carry, and the qualifier the generator decorates it by.
SPIRV_MEMORY_QUALIFIERS = {
    "coherent": "coherent",
    "globallycoherent": "globallycoherent",
    "volatile": "volatile",
    "restrict": "restrict",
    "readonly": "readonly",
    "read": "readonly",
    "access::read": "readonly",
    "writeonly": "writeonly",
    "write": "writeonly",
    "access::write": "writeonly",
    "readwrite": "readwrite",
    "read_write": "readwrite",
    "access::read_write": "readwrite",
}

# The SPIR-V BuiltIn of each system-value semantic a mesh output member may carry, per output array
# (GL_EXT_mesh_shader section 7.1.6, gl_MeshVerticesEXT and gl_MeshPrimitivesEXT; HLSL mesh shader outputs). HLSL
# semantics are case-insensitive, so the SV_ keys are lower case. D3D12_SHADING_RATE and the Vulkan primitive
# shading rate share their encoding (bits 0-1 vertical, bits 2-3 horizontal), so SV_ShadingRate needs no
# conversion.
SPIRV_MESH_OUTPUT_BUILTINS = {
    "vertices": {
        "gl_Position": "Position",
        "sv_position": "Position",
        "gl_PointSize": "PointSize",
    },
    "primitives": {
        "gl_PrimitiveID": "PrimitiveId",
        "sv_primitiveid": "PrimitiveId",
        "gl_Layer": "Layer",
        "sv_rendertargetarrayindex": "Layer",
        "gl_ViewportIndex": "ViewportIndex",
        "sv_viewportarrayindex": "ViewportIndex",
        "gl_CullPrimitiveEXT": "CullPrimitiveEXT",
        "sv_cullprimitive": "CullPrimitiveEXT",
        "gl_PrimitiveShadingRateEXT": "PrimitiveShadingRateKHR",
        "sv_shadingrate": "PrimitiveShadingRateKHR",
    },
}

# The SPIR-V spelling of each scalar type alias, keyed by every alias the generator meets in imported programs.
SPIRV_SCALAR_ALIASES = {
    "f32": "float",
    "f64": "double",
    "f16": "float",
    "float16": "float",
    "float16_t": "float",
    "half": "float",
    "min16float": "float",
    "i8": "int",
    "int8": "int",
    "int8_t": "int",
    "char": "int",
    "i16": "int",
    "int16": "int",
    "int16_t": "int",
    "short": "int",
    "i32": "int",
    "u8": "uint",
    "uint8": "uint",
    "uint8_t": "uint",
    "uchar": "uint",
    "u16": "uint",
    "uint16": "uint",
    "uint16_t": "uint",
    "ushort": "uint",
    "u32": "uint",
    "int64": "i64",
    "int64_t": "i64",
    "long": "i64",
    "ptrdiff_t": "i64",
    "uint64": "u64",
    "uint64_t": "u64",
    "ulong": "u64",
    "size_t": "u64",
}

_BFLOAT16_ALIASES = frozenset({"bfloat", "bfloat16_t"})


def spirv_scalar_key(type_name, bfloat16_type_name):
    """The scalar spelling ``type_name`` is classified by: its SPIR-V scalar if it is an alias of one, else itself.

    The key is compared with scalar type names and never written. A name that is no scalar alias (a scalar's
    own spelling, a vector, matrix, image or structure name) classifies as itself; an undeclared type name is
    refused before generation (``semantics.declared_types``).
    """
    text = str(type_name)
    if text in _BFLOAT16_ALIASES:
        return bfloat16_type_name
    scalar = SPIRV_SCALAR_ALIASES.get(text)
    return text if scalar is None else scalar


def spirv_builtin_call_key(function_name):
    """The name the built-in call dispatch selects a lowering by: the GLSL name of an HLSL alias, else itself.

    The key selects a lowering and is never written: a name no lowering knows reaches the end of
    ``call_builtin_function``, which raises ``SPIRVBuiltinLoweringError`` (reason ``unknown-function``).
    """
    alias = SPIRV_BUILTIN_CALL_ALIASES.get(function_name)
    return function_name if alias is None else alias


def _mesh_semantic_key(semantic):
    text = str(semantic)
    return text.lower() if text[:3].lower() == "sv_" else text


def spirv_mesh_output_builtin(role, semantic):
    """The BuiltIn a ``role`` (``vertices``/``primitives``) mesh output member with ``semantic`` decorates, or None.

    None means a user semantic, which takes a Location; :func:`spirv_mesh_output_refusal` says when a semantic
    that names a system value has no BuiltIn here.
    """
    if semantic is None:
        return None
    return SPIRV_MESH_OUTPUT_BUILTINS.get(role, {}).get(_mesh_semantic_key(semantic))


def spirv_mesh_output_refusal(role, semantic):
    """Why ``semantic`` cannot mark a ``role`` mesh output member, or None if it is a BuiltIn or a user semantic.

    A system value (``SV_*`` or ``gl_*``) with no BuiltIn in ``role``, such as ``SV_ClipDistance`` or a
    per-primitive semantic on a vertex, would otherwise become a user output at a Location, which the next
    stage never reads as that system value.
    """
    if semantic is None or spirv_mesh_output_builtin(role, semantic) is not None:
        return None
    text = str(semantic)
    if text[:3].lower() != "sv_" and not text.startswith("gl_"):
        return None
    other = {
        "vertices": ("primitives", "primitive"),
        "primitives": ("vertices", "vertex"),
    }.get(role)
    if other is not None and spirv_mesh_output_builtin(other[0], semantic) is not None:
        return f"mesh {role} output semantic '{text}' is a per-{other[1]} system value"
    return (
        f"mesh {role} output semantic '{text}' names a system value with no SPIR-V mesh output "
        "BuiltIn the generator lowers"
    )
