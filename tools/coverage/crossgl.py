"""How the coverage probes write CrossGL: the template layer.

The CrossGL IR has no compatibility guarantee. Every CrossGL probe is generated from this module and the expression
syntax :class:`~tools.coverage.dialect.Dialect` shares with GLSL, so a pull request that changes how CrossGL writes a
compute kernel, a storage buffer, a type or a built-in function updates this module too, and no probe text is pinned
anywhere else.
"""

from __future__ import annotations

from .dialect import Buffer, Dialect

# Built-in functions CrossGL spells as GLSL does rather than as the backend does. A rename applies only when the GLSL
# feature list has the new name; any other backend function is probed under its own spelling.
RENAMES = {
    "hlsl": {
        "lerp": "mix",
        "frac": "fract",
        "rsqrt": "inversesqrt",
        "mad": "fma",
        "countbits": "bitCount",
        "firstbithigh": "findMSB",
        "firstbitlow": "findLSB",
        "reversebits": "bitfieldReverse",
        "asint": "floatBitsToInt",
        "asuint": "floatBitsToUint",
        "atan2": "atan",
        "ddx": "dFdx",
        "ddy": "dFdy",
        "ddx_coarse": "dFdxCoarse",
        "ddx_fine": "dFdxFine",
        "ddy_coarse": "dFdyCoarse",
        "ddy_fine": "dFdyFine",
        "GroupMemoryBarrierWithGroupSync": "barrier",
        "GroupMemoryBarrier": "memoryBarrierShared",
        "AllMemoryBarrier": "memoryBarrier",
        "DeviceMemoryBarrier": "memoryBarrierBuffer",
        "InterlockedAdd": "atomicAdd",
        "InterlockedMin": "atomicMin",
        "InterlockedMax": "atomicMax",
        "InterlockedAnd": "atomicAnd",
        "InterlockedOr": "atomicOr",
        "InterlockedXor": "atomicXor",
        "WaveActiveAllTrue": "allInvocations",
        "WaveActiveAnyTrue": "anyInvocation",
        "WaveActiveAllEqual": "allInvocationsEqual",
    },
    "wgsl": {
        "atan2": "atan",
        "inverseSqrt": "inversesqrt",
        "faceForward": "faceforward",
        "countOneBits": "bitCount",
        "reverseBits": "bitfieldReverse",
        "firstLeadingBit": "findMSB",
        "firstTrailingBit": "findLSB",
        "extractBits": "bitfieldExtract",
        "insertBits": "bitfieldInsert",
        "bitcast": "floatBitsToUint",
        "select": "mix",
        "dpdx": "dFdx",
        "dpdy": "dFdy",
        "dpdxCoarse": "dFdxCoarse",
        "dpdxFine": "dFdxFine",
        "dpdyCoarse": "dFdyCoarse",
        "dpdyFine": "dFdyFine",
        "pack4x8snorm": "packSnorm4x8",
        "pack4x8unorm": "packUnorm4x8",
        "pack2x16snorm": "packSnorm2x16",
        "pack2x16unorm": "packUnorm2x16",
        "pack2x16float": "packHalf2x16",
        "unpack4x8snorm": "unpackSnorm4x8",
        "unpack4x8unorm": "unpackUnorm4x8",
        "unpack2x16snorm": "unpackSnorm2x16",
        "unpack2x16unorm": "unpackUnorm2x16",
        "unpack2x16float": "unpackHalf2x16",
        "workgroupBarrier": "barrier",
        "storageBarrier": "memoryBarrierBuffer",
        "textureBarrier": "memoryBarrierImage",
        "subgroupAll": "allInvocations",
        "subgroupAny": "anyInvocation",
    },
    "metal": {
        "atan2": "atan",
        "fabs": "abs",
        "fmax": "max",
        "fmin": "min",
        "rsqrt": "inversesqrt",
        "rint": "roundEven",
        "select": "mix",
        "popcount": "bitCount",
        "reverse_bits": "bitfieldReverse",
        "extract_bits": "bitfieldExtract",
        "insert_bits": "bitfieldInsert",
        "dfdx": "dFdx",
        "dfdy": "dFdy",
        "discard_fragment": "discard",
        "pack_float_to_snorm4x8": "packSnorm4x8",
        "pack_float_to_unorm4x8": "packUnorm4x8",
        "pack_float_to_snorm2x16": "packSnorm2x16",
        "pack_float_to_unorm2x16": "packUnorm2x16",
        "unpack_snorm4x8_to_float": "unpackSnorm4x8",
        "unpack_unorm4x8_to_float": "unpackUnorm4x8",
        "unpack_snorm2x16_to_float": "unpackSnorm2x16",
        "unpack_unorm2x16_to_float": "unpackUnorm2x16",
        "simd_all": "allInvocations",
        "simd_any": "anyInvocation",
    },
}


def spelling(language: str, name: str, glsl_names: set[str] | frozenset[str]) -> str:
    """The CrossGL name of ``language``'s built-in ``name``."""
    renamed = RENAMES.get(language, {}).get(name, name)
    return renamed if renamed in glsl_names else name


class CrossGL(Dialect):
    language = "crossgl"
    suffix = ".cgl"

    def kernel(self, buffers: list[Buffer], body: list[str]) -> str:
        lines = [
            "shader Probe {",
            "    compute {",
            "        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;",
        ]
        lines += [f"        buffer {kind}* {name};" for kind, name in buffers]
        lines += [
            "        void main() {",
            "            uint i = gl_GlobalInvocationID.x;",
        ]
        lines += [f"            {line}" for line in body]
        lines += ["        }", "    }", "}"]
        return "\n".join(lines) + "\n"


DIALECT = CrossGL()
