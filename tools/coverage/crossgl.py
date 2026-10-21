"""How the coverage probes write CrossGL: the template layer.

The CrossGL IR has no compatibility guarantee. Every CrossGL probe is generated from this module and the expression
syntax :class:`~tools.coverage.dialect.Dialect` shares with GLSL, so a pull request that changes how CrossGL writes a
compute kernel, a storage buffer, a type or a built-in function updates this module too, and no probe text is pinned
anywhere else.
"""

from __future__ import annotations

from .dialect import Buffer, Dialect, ProbeError
from .model import WIDE_INTEGERS
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


# Functions CrossGL writes with other arguments than the backend, as GLSL prototypes. WGSL's texture functions take a
# texture and a sampler, where CrossGL, like GLSL, takes one combined sampler; ``const`` marks an argument the probe
# passes as the constant 0, for a WGSL function that always reads level 0.
COUNTERPARTS = {
    "wgsl": {
        "textureDimensions": "ivec2 textureSize(sampler2D, const int)",
        "textureGather": "vec4 textureGather(sampler2D, vec2)",
        "textureGatherCompare": "vec4 textureGather(sampler2DShadow, vec2, float)",
        "textureLoad": "vec4 texelFetch(sampler2D, ivec2, int)",
        "textureNumLayers": "ivec3 textureSize(sampler2DArray, const int)",
        "textureNumLevels": "int textureQueryLevels(sampler2D)",
        "textureNumSamples": "int textureSamples(sampler2DMS)",
        "textureSample": "vec4 texture(sampler2D, vec2)",
        "textureSampleBias": "vec4 texture(sampler2D, vec2, float)",
        "textureSampleCompare": "float texture(sampler2DShadow, vec3)",
        "textureSampleCompareLevel": (
            "float textureLod(sampler2DShadow, vec3, const float)"
        ),
        "textureSampleGrad": "vec4 textureGrad(sampler2D, vec2, vec2, vec2)",
        "textureSampleLevel": "vec4 textureLod(sampler2D, vec2, float)",
        "textureStore": "void imageStore(image2D, ivec2, vec4)",
        # WGSL returns a result struct where CrossGL, like GLSL, returns one part and writes the other to an output
        # parameter. Its only compare-exchange is the weak one, and CrossGL's is atomicCompSwap.
        "frexp": "vec3 frexp(vec3, out ivec3)",
        "modf": "vec3 modf(vec3, out vec3)",
        "atomicCompareExchangeWeak": "int atomicCompSwap(inout int, int, int)",
    },
    # Metal's barrier takes memory flags, which the probe passes as mem_none: an execution barrier only.
    "metal": {"threadgroup_barrier": "void barrier()"},
}
# Metal functions whose overloads take a pointer or return an opaque type (atomic compare-exchange takes the expected
# value through a pointer, a ballot returns a simd_vote) are measured in the output direction only, through the
# CrossGL operation that translates to them.
COUNTERPARTS["metal"].update(
    {
        "atomic_compare_exchange_weak": "int atomicCompSwap(inout int, int, int)",
        "atomic_compare_exchange_weak_explicit": (
            "int atomicCompSwap(inout int, int, int)"
        ),
        "simd_ballot": "uvec4 subgroupBallot(bool)",
    }
)
# Metal's atomic functions, in forms with and without a memory order. CrossGL's, like GLSL's, are relaxed and take
# none, so the probe passes memory_order_relaxed to the Metal function.
_METAL_ATOMICS = {
    "exchange": "atomicExchange",
    "fetch_add": "atomicAdd",
    "fetch_and": "atomicAnd",
    "fetch_max": "atomicMax",
    "fetch_min": "atomicMin",
    "fetch_or": "atomicOr",
    "fetch_xor": "atomicXor",
}
RENAMES["metal"].update({f"atomic_{op}": name for op, name in _METAL_ATOMICS.items()})
COUNTERPARTS["metal"].update(
    {
        f"atomic_{op}_explicit": f"int {name}(inout int, int)"
        for op, name in _METAL_ATOMICS.items()
    }
)


def spelling(language: str, name: str, glsl_names: set[str] | frozenset[str]) -> str:
    """The CrossGL name of ``language``'s built-in ``name``."""
    renamed = RENAMES.get(language, {}).get(name, name)
    return renamed if renamed in glsl_names else name


# The stage block of each ray-tracing stage a probe runs in.
_RAYTRACING_BLOCKS = {
    "ray-generation": "ray_generation",
    "intersection": "ray_intersection",
    "any-hit": "ray_any_hit",
    "closest-hit": "ray_closest_hit",
}


class CrossGL(Dialect):
    language = "crossgl"
    suffix = ".cgl"
    wide_integers = True
    narrow_scalars = True
    packed_bytes = True
    # The spellings CrossGL's own importers write for these scalars; the HLSL importer reads a packed 8-bit word as
    # ``uint``, which HLSL converts to and from implicitly.
    _SCALARS = {
        "int64": "int64_t",
        "uint64": "uint64_t",
        "half": "float16",
        "int16": "int16",
        "uint16": "uint16",
        "int8x4_packed": "uint",
        "uint8x4_packed": "uint",
    }
    _PREFIX = {**Dialect._PREFIX, "half": "f16", "int16": "i16", "uint16": "u16"}

    def scalar(self, scalar: str) -> str:
        return self._SCALARS.get(scalar, scalar)

    def vector(self, scalar: str, size: int) -> str:
        if scalar in WIDE_INTEGERS:
            raise ProbeError("no 64-bit integer vector probe for CrossGL yet")
        return super().vector(scalar, size)

    def matrix(self, scalar: str, columns: int, rows: int) -> str:
        if scalar == "half":
            shape = f"{columns}" if columns == rows else f"{columns}x{rows}"
            return f"f16mat{shape}"
        return super().matrix(scalar, columns, rows)

    # CrossGL declares HLSL's resource object types and calls their methods under HLSL's names; the backend assigns
    # the registers.
    def resource(self, spelling: str, name: str, register: str) -> str:
        return f"{spelling} {name};"

    # GLSL's samplers and images keep their GLSL spelling; the backend assigns the bindings.
    def uniform(self, spelling: str, name: str, binding: int, image_format: str) -> str:
        layout = f"layout({image_format}) " if image_format else ""
        return f"{layout}uniform {spelling} {name};"

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

    def stage_input(self, name: str) -> str:
        return f"input.{name}"

    def fragment(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        lines = ["shader Probe {"]
        lines += [f"    {line}" for line in parts.get("prelude", ())]
        lines += [
            "    struct FragmentInput {",
            "        vec2 uv;",
            *(f"        {kind} {name};" for kind, name in parts.get("inputs", ())),
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

    # A hit shader's parameters are the ray payload and the hit attributes, which the backends declare as their
    # target requires.
    _HIT_PARAMETERS = (
        "Payload payload @ payload, "
        "BuiltInTriangleIntersectionAttributes attributes @ hit_attribute"
    )

    def raytracing(
        self, stage: str, buffers: list[Buffer], body: list[str], **parts
    ) -> str:
        hit = stage in ("any-hit", "closest-hit")
        lines = ["shader Probe {"]
        lines += [f"    {line}" for line in parts.get("prelude", ())]
        if hit:
            lines += ["    struct Payload {", "        vec4 color;", "    }"]
        lines += [f"    {_RAYTRACING_BLOCKS[stage]} {{"]
        lines += [f"        buffer {kind}* {name};" for kind, name in buffers]
        lines += [f"        {line}" for line in parts.get("declarations", ())]
        lines += [
            f"        void main({self._HIT_PARAMETERS if hit else ''}) {{",
            "            uint i = 0u;",
        ]
        lines += [f"            {line}" for line in body]
        lines += ["        }", "    }", "}"]
        return "\n".join(lines) + "\n"

    def geometry(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        lines = ["shader Probe {"]
        lines += [f"    {line}" for line in parts.get("prelude", ())]
        lines += ["    geometry {", *(f"        {line}" for line in _POINTS_IN_AND_OUT)]
        lines += [f"        buffer {kind}* {name};" for kind, name in buffers]
        lines += [f"        {line}" for line in parts.get("declarations", ())]
        lines += ["        void main() {", "            uint i = 0u;"]
        lines += [f"            {line}" for line in body]
        lines += ["        }", "    }", "}"]
        return "\n".join(lines) + "\n"

    def _program(
        self,
        block: str,
        shared: tuple[str, ...],
        buffers: list[Buffer],
        functions: list[str],
        parts: dict,
    ) -> str:
        """A probe whose ``block`` stage declares the buffers and then ``functions``, after the shader-level
        ``shared`` declarations."""
        lines = ["shader Probe {"]
        lines += [f"    {line}" for line in (*parts.get("prelude", ()), *shared)]
        lines += [f"    {block} {{"]
        lines += [f"        buffer {kind}* {name};" for kind, name in buffers]
        lines += [f"        {line}" for line in parts.get("declarations", ())]
        lines += [f"        {line}" for line in functions]
        lines += ["    }", "}"]
        return "\n".join(lines) + "\n"

    def tessellation_control(
        self, buffers: list[Buffer], body: list[str], **parts
    ) -> str:
        functions = [
            "PatchConstants constants(InputPatch<ControlPoint, 3> patch) {",
            "    PatchConstants output;",
            "    output.edges[0] = 1.0;",
            "    output.edges[1] = 1.0;",
            "    output.edges[2] = 1.0;",
            "    output.inside = 1.0;",
            "    uint i = 0u;",
            *(f"    {line}" for line in body),
            "    return output;",
            "}",
            "ControlPoint main(InputPatch<ControlPoint, 3> patch, uint id @ SV_OutputControlPointID)",
            "    @domain(tri)",
            "    @partitioning(integer)",
            "    @outputtopology(triangle_cw)",
            "    @outputcontrolpoints(3)",
            "    @patchconstantfunc(constants) {",
            "    ControlPoint output;",
            "    output.position = patch[id].position * 2.0;",
            "    return output;",
            "}",
        ]
        shared = (
            "struct ControlPoint {",
            "    vec4 position @ POSITION;",
            "}",
            "struct PatchConstants {",
            "    float edges[3] @ SV_TessFactor;",
            "    float inside @ SV_InsideTessFactor;",
            "}",
        )
        return self._program("tessellation_control", shared, buffers, functions, parts)

    def mesh(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        functions = [
            "void main(@ vertices out Vertex verts[3], @ indices out uvec3 tris[1])",
            "    @numthreads(1, 1, 1)",
            "    @outputtopology(triangle) {",
            "    uint i = 0u;",
            *(f"    {line}" for line in body),
            "    verts[0].position = vec4(0.0, 0.0, 0.0, 1.0);",
            "    verts[1].position = vec4(1.0, 0.0, 0.0, 1.0);",
            "    verts[2].position = vec4(0.0, 1.0, 0.0, 1.0);",
            "    tris[0] = uvec3(0, 1, 2);",
            "}",
        ]
        shared = ("struct Vertex {", "    vec4 position @ gl_Position;", "}")
        return self._program("mesh", shared, buffers, functions, parts)

    def task(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        functions = [
            "void main() @numthreads(1, 1, 1) {",
            "    uint i = 0u;",
            *(f"    {line}" for line in body),
            "}",
        ]
        shared = (
            "struct Payload {",
            "    uint value;",
            "}",
            f"groupshared Payload {self.payload} @ taskPayloadSharedEXT;",
        )
        return self._program("task", shared, buffers, functions, parts)


# A geometry stage's input and output primitives, as the CrossGL importers write them.
_POINTS_IN_AND_OUT = ("layout(points) in;", "layout(points, max_vertices = 1) out;")

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


def _raytracing(
    block: str, parameters: str = "", body: str = "outp[0] = 1.0;"
) -> Template:
    """A ray-tracing stage program; a stage that takes ``parameters`` declares their struct, else writes a buffer."""
    kind = parameters.split(" ", 1)[0]
    lines = ["shader Probe {"]
    if kind:
        lines += [f"    struct {kind} {{", "        vec4 color;", "    }"]
    lines += [f"    {block} {{"]
    if not kind:
        lines += ["        buffer float* outp;"]
    lines += [
        f"        void main({parameters}) {{",
        f"            {body}",
        "        }",
        "    }",
        "}",
    ]
    return Template(program="\n".join(lines) + "\n")


_HIT = (
    "Payload payload @ payload, BuiltInTriangleIntersectionAttributes attributes @ hit_attribute",
    "payload.color = vec4(attributes.barycentrics, 0.0, 1.0);",
)
_RAYTRACING_STAGES = {
    "ray-generation": _raytracing("ray_generation"),
    "intersection": _raytracing("ray_intersection"),
    "any-hit": _raytracing("ray_any_hit", *_HIT),
    "closest-hit": _raytracing("ray_closest_hit", *_HIT),
    "miss": _raytracing(
        "ray_miss",
        "Payload payload @ payload",
        "payload.color = vec4(0.0, 0.0, 0.0, 1.0);",
    ),
    "callable": _raytracing(
        "ray_callable",
        "CallData data @ callable_data",
        "data.color = vec4(1.0, 0.0, 0.0, 1.0);",
    ),
}

# A triangle patch of three control points, tessellated once. CrossGL states the domain, partitioning, topology and
# patch-constant function on the stage, which every target needs, and names the tessellation values by their
# semantics, as the HLSL importer does.
_PATCH = (
    "shader Probe {\n    struct ControlPoint {\n        vec4 position @ POSITION;\n    }\n"
    "    struct PatchConstants {\n        float edges[3] @ SV_TessFactor;\n"
    "        float inside @ SV_InsideTessFactor;\n    }\n"
)
_TESSELLATION_STAGES = {
    "tessellation-control": Template(
        program=_PATCH + "    tessellation_control {\n"
        "        PatchConstants constants(InputPatch<ControlPoint, 3> patch) {\n"
        "            PatchConstants output;\n            output.edges[0] = 1.0;\n            output.edges[1] = 1.0;\n"
        "            output.edges[2] = 1.0;\n            output.inside = 1.0;\n            return output;\n"
        "        }\n"
        "        ControlPoint main(InputPatch<ControlPoint, 3> patch, uint id @ SV_OutputControlPointID)\n"
        "            @domain(tri)\n            @partitioning(integer)\n            @outputtopology(triangle_cw)\n"
        "            @outputcontrolpoints(3)\n            @patchconstantfunc(constants) {\n"
        "            ControlPoint output;\n            output.position = patch[id].position * 2.0;\n"
        "            return output;\n        }\n    }\n}\n"
    ),
    "tessellation-evaluation": Template(
        program=_PATCH
        + "    struct DomainOutput {\n        vec4 position @ SV_Position;\n    }\n"
        "    tessellation_evaluation {\n"
        "        DomainOutput main(PatchConstants constants, vec3 coord @ SV_DomainLocation,\n"
        "                          OutputPatch<ControlPoint, 3> patch) @domain(tri) {\n"
        "            DomainOutput output;\n"
        "            output.position = patch[0].position * coord.x + patch[1].position * coord.y"
        " + patch[2].position * coord.z;\n            return output;\n        }\n    }\n}\n"
    ),
}

# A mesh shader that writes one triangle, and a task shader that hands a payload to one mesh workgroup. The task
# payload lives in task-payload storage, which CrossGL names by the GLSL qualifier.
_MESH_STAGES = {
    "mesh": Template(
        program="shader Probe {\n    struct Vertex {\n        vec4 position @ gl_Position;\n    }\n    mesh {\n"
        "        void main(@ vertices out Vertex verts[3], @ indices out uvec3 tris[1])\n"
        "            @numthreads(1, 1, 1)\n            @outputtopology(triangle) {\n"
        "            SetMeshOutputCounts(3, 1);\n"
        "            verts[0].position = vec4(0.0, 0.0, 0.0, 1.0);\n"
        "            verts[1].position = vec4(1.0, 0.0, 0.0, 1.0);\n"
        "            verts[2].position = vec4(0.0, 1.0, 0.0, 1.0);\n"
        "            tris[0] = uvec3(0, 1, 2);\n        }\n    }\n}\n"
    ),
    "task": Template(
        program="shader Probe {\n    struct Payload {\n        uint value;\n    }\n"
        "    groupshared Payload payload @ taskPayloadSharedEXT;\n    task {\n"
        "        void main() @numthreads(1, 1, 1) {\n            payload.value = 1u;\n"
        "            DispatchMesh(1u, 1u, 1u, payload);\n        }\n    }\n}\n"
    ),
}

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
    "geometry": Template(
        program="shader Probe {\n    in vec4 inPosition[1] @location(0);\n    geometry {\n"
        + "".join(f"        {line}\n" for line in _POINTS_IN_AND_OUT)
        + "        void main() {\n            gl_Position = inPosition[0] * 2.0;\n"
        "            EmitVertex();\n            EndPrimitive();\n        }\n    }\n}\n"
    ),
    "fragment": Template(program=_FRAGMENT.replace("{body}", "")),
    **_TESSELLATION_STAGES,
    **_MESH_STAGES,
    **_RAYTRACING_STAGES,
    "discard": Template(
        program=_FRAGMENT.replace(
            "{body}",
            "            if (input.uv.x < 0.5) {\n                discard;\n            }\n",
        )
    ),
}
