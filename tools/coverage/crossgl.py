"""How the coverage probes write CrossGL: the template layer.

The CrossGL IR has no compatibility guarantee. Every CrossGL probe is generated from this module and the expression
syntax :class:`~tools.coverage.dialect.Dialect` shares with GLSL, so a pull request that changes how CrossGL writes a
compute kernel, a storage buffer, a type or a built-in function updates this module too, and no probe text is pinned
anywhere else.
"""

from __future__ import annotations

from .dialect import Buffer, Dialect
from .templates import C_FAMILY, Template

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
    # CUDA after :mod:`.features.cuda` drops the single-precision suffix: ``fabsf`` arrives here as ``fabs``.
    "cuda": {
        "fabs": "abs",
        "fmin": "min",
        "fmax": "max",
        "rsqrt": "inversesqrt",
        "atan2": "atan",
        "rint": "roundEven",
        "__popc": "bitCount",
        "__brev": "bitfieldReverse",
        "__float_as_int": "floatBitsToInt",
        "__float_as_uint": "floatBitsToUint",
        "__int_as_float": "intBitsToFloat",
        "__uint_as_float": "uintBitsToFloat",
        "__isnanf": "isnan",
        "__isinff": "isinf",
        "__syncthreads": "barrier",
        "__threadfence": "memoryBarrier",
        "__threadfence_block": "memoryBarrierShared",
        "atomicExch": "atomicExchange",
        "atomicCAS": "atomicCompSwap",
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
    "mojo": {
        "atan2": "atan",
        "rsqrt": "inversesqrt",
        "pop_count": "bitCount",
        "bit_reverse": "bitfieldReverse",
    },
    # Rust names a function by its trait or type, as ``Float::ln`` or ``Vec3::lerp``; only the ones that compute what
    # the GLSL function does are renamed.
    "rust": {
        "Float::ln": "log",
        "Float::powf": "pow",
        "Float::mul_add": "fma",
        "Float::to_degrees": "degrees",
        "Float::to_radians": "radians",
        "Float::is_nan": "isnan",
        "Float::is_infinite": "isinf",
        "Float::atan2": "atan",
        "Vec3::ln": "log",
        "Vec3::powf": "pow",
        "Vec3::mul_add": "fma",
        "Vec3::lerp": "mix",
        "Vec3::fract_gl": "fract",
        "Vec3::is_nan_mask": "isnan",
        "Vec3::cmpeq": "equal",
        "Vec3::cmpne": "notEqual",
        "Vec3::cmplt": "lessThan",
        "Vec3::cmple": "lessThanEqual",
        "Vec3::cmpgt": "greaterThan",
        "Vec3::cmpge": "greaterThanEqual",
        "Derivative::dfdx": "dFdx",
        "Derivative::dfdy": "dFdy",
        "Derivative::fwidth": "fwidth",
        "Derivative::dfdx_fine": "dFdxFine",
        "Derivative::dfdy_fine": "dFdyFine",
        "Derivative::fwidth_fine": "fwidthFine",
        "Derivative::dfdx_coarse": "dFdxCoarse",
        "Derivative::dfdy_coarse": "dFdyCoarse",
        "Derivative::fwidth_coarse": "fwidthCoarse",
        "workgroup_memory_barrier": "memoryBarrierShared",
        "workgroup_memory_barrier_with_group_sync": "barrier",
        "device_memory_barrier": "memoryBarrierBuffer",
        "all_memory_barrier": "memoryBarrier",
        "atomic_i_add": "atomicAdd",
        "atomic_f_add": "atomicAdd",
        "atomic_s_min": "atomicMin",
        "atomic_u_min": "atomicMin",
        "atomic_f_min": "atomicMin",
        "atomic_s_max": "atomicMax",
        "atomic_u_max": "atomicMax",
        "atomic_f_max": "atomicMax",
        "atomic_and": "atomicAnd",
        "atomic_or": "atomicOr",
        "atomic_xor": "atomicXor",
        "atomic_exchange": "atomicExchange",
        "subgroup_all": "allInvocations",
        "subgroup_any": "anyInvocation",
        "subgroup_all_equal": "allInvocationsEqual",
    },
}
# HIP spells its device functions the way CUDA does, and Slang its built-in functions the way HLSL does.
RENAMES["hip"] = RENAMES["cuda"]
RENAMES["slang"] = RENAMES["hlsl"]


def spelling(language: str, name: str, glsl_names: set[str] | frozenset[str]) -> str:
    """The CrossGL name of ``language``'s built-in ``name``."""
    renamed = RENAMES.get(language, {}).get(name, name)
    return renamed if renamed in glsl_names else name


class CrossGL(Dialect):
    language = "crossgl"
    suffix = ".cgl"

    # CrossGL declares HLSL's resource object types and calls their methods under HLSL's names; the backend assigns
    # the registers.
    def resource(self, spelling: str, name: str, register: str) -> str:
        return f"{spelling} {name};"

    def kernel(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        lines = ["shader Probe {"]
        lines += [f"    {line}" for line in parts.get("prelude", ())]
        lines += [
            "    compute {",
            "        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;",
        ]
        lines += [f"        buffer {kind}* {name};" for kind, name in buffers]
        lines += [f"        {line}" for line in parts.get("declarations", ())]
        lines += [
            "        void main() {",
            "            uint i = gl_GlobalInvocationID.x;",
        ]
        lines += [f"            {line}" for line in body]
        lines += ["        }", "    }", "}"]
        return "\n".join(lines) + "\n"

    def fragment(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        lines = ["shader Probe {"]
        lines += [f"    {line}" for line in parts.get("prelude", ())]
        lines += [
            "    struct FragmentInput {",
            "        vec2 uv;",
            "    }",
            "    struct FragmentOutput {",
            "        vec4 color;",
            "    }",
            "    fragment {",
        ]
        lines += [f"        buffer {kind}* {name};" for kind, name in buffers]
        lines += [f"        {line}" for line in parts.get("declarations", ())]
        lines += [
            "        FragmentOutput main(FragmentInput input) {",
            "            FragmentOutput output;",
            "            uint i = 0u;",
        ]
        lines += [f"            {line}" for line in body]
        lines += [
            "            output.color = vec4(input.uv, 0.0, 1.0);",
            "            return output;",
            "        }",
            "    }",
            "}",
        ]
        return "\n".join(lines) + "\n"


DIALECT = CrossGL()


def _scalar(spelling: str, source: str = "iin") -> Template:
    return Template(f"{spelling} v = {spelling}({source}[i]);\noutp[i] = float(v * v);")


_PAIR = "struct Pair {\n    float a;\n    float b;\n};"
_FRAGMENT = (
    "shader Probe {\n    struct FragmentInput {\n        vec2 uv;\n    }\n    struct FragmentOutput {\n"
    "        vec4 color;\n    }\n    fragment {\n        FragmentOutput main(FragmentInput input) {\n"
    "            FragmentOutput output;\n{body}            output.color = vec4(input.uv, 0.0, 1.0);\n"
    "            return output;\n        }\n    }\n}\n"
)

# Language-level probes (:mod:`.constructs`). Kernel templates use the CrossGL kernel above: ``prelude`` lines are
# shader-level declarations, ``declarations`` go inside the compute block.
CONSTRUCTS: dict[str, Template] = {
    **C_FAMILY,
    "struct": Template(C_FAMILY["struct"].body, prelude=_PAIR),
    "array-of-structs": Template(C_FAMILY["array-of-structs"].body, prelude=_PAIR),
    "global-constant": Template(
        "outp[i] = fin[i] * SCALE;", prelude="const float SCALE = 2.0;"
    ),
    "runtime-array-length": Template("outp[i] = float(outp.length());"),
    "half": _scalar("half", "fin"),
    "double": _scalar("double", "fin"),
    "int8": _scalar("int8_t"),
    "uint8": _scalar("uint8_t"),
    "int16": _scalar("int16_t"),
    "uint16": _scalar("uint16_t"),
    "int64": _scalar("int64_t"),
    "uint64": _scalar("uint64_t"),
    "shared-memory": Template(
        "tile[0] = fin[i];\nbarrier();\noutp[i] = tile[0];", "shared float tile[1];"
    ),
    "local-invocation-id": Template("outp[i] = float(gl_LocalInvocationID.x);"),
    "workgroup-id": Template("outp[i] = float(gl_WorkGroupID.x);"),
    "local-invocation-index": Template("outp[i] = float(gl_LocalInvocationIndex);"),
    "num-workgroups": Template("outp[i] = float(gl_NumWorkGroups.x);"),
    "uniform-buffer": Template(
        "outp[i] = fin[i] * scale;", prelude="cbuffer Params {\n    float scale;\n};"
    ),
    "read-only-buffer": Template("outp[i] = rin[i];", "readonly buffer float* rin;"),
    "sampled-texture": Template(
        "outp[i] = textureLod(tex, vec2(fin[i], 0.5), 0.0).x;", "uniform sampler2D tex;"
    ),
    "storage-image": Template(
        "imageStore(img, ivec2(0, 0), vec4(fin[i]));\noutp[i] = 0.0;",
        "layout(rgba32f) uniform image2D img;",
    ),
    "vertex": Template(
        program="shader Probe {\n    struct VertexInput {\n        vec4 position;\n    }\n"
        "    struct VertexOutput {\n        vec4 position @ gl_Position;\n    }\n    vertex {\n"
        "        VertexOutput main(VertexInput input) {\n            VertexOutput output;\n"
        "            output.position = input.position * 2.0;\n            return output;\n        }\n"
        "    }\n}\n"
    ),
    "fragment": Template(program=_FRAGMENT.replace("{body}", "")),
    "discard": Template(
        program=_FRAGMENT.replace(
            "{body}",
            "            if (input.uv.x < 0.5) {\n                discard;\n            }\n",
        )
    ),
}
