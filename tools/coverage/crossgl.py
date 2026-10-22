"""How the coverage probes write CrossGL: the template layer.

The CrossGL IR has no compatibility guarantee. Every CrossGL probe is generated from this module and the expression
syntax :class:`~tools.coverage.dialect.Dialect` shares with GLSL, so a pull request that changes how CrossGL writes a
compute kernel, a storage buffer, a type or a built-in function updates this module too, and no probe text is pinned
anywhere else.
"""

from __future__ import annotations

from .dialect import (
    UNIFORM_ARRAY_LENGTH,
    Buffer,
    Dialect,
    NoSpelling,
    ProbeError,
    uniform_read,
)
from .model import (
    COOP_COLUMNS,
    COOP_ENUMERATORS,
    WIDE_INTEGERS,
    Type,
    coop_element,
    coop_literal,
    tuple_element,
)
from .objects import HIT_OBJECT, OBJECTS, VERTEX
from .spellings import ELEMENT, SUBSCRIPT
from .templates import C_FAMILY, Template

# The stage block of each ray-tracing stage a probe runs in.
_RAYTRACING_BLOCKS = {
    "ray-generation": "ray_generation",
    "intersection": "ray_intersection",
    "any-hit": "ray_any_hit",
    "closest-hit": "ray_closest_hit",
    "callable": "ray_callable",
}


class CrossGL(Dialect):
    language = "crossgl"
    suffix = ".cgl"
    wide_integers = True
    narrow_scalars = True
    packed_bytes = True
    pointer_sized = True
    string = "string"
    # The spellings CrossGL's own importers write for these scalars; the HLSL importer reads a packed 8-bit word as
    # ``uint``, which HLSL converts to and from implicitly, and the Rust importer reads ``usize`` as ``uint``.
    _SCALARS = {
        "int64": "int64_t",
        "uint64": "uint64_t",
        "half": "float16",
        "int16": "int16",
        "uint16": "uint16",
        "int8x4_packed": "uint",
        "uint8x4_packed": "uint",
        "usize": "uint",
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
    def uniform(self, spelling: str, name: str, binding: int, qualifier: str) -> str:
        layout = f"layout({qualifier}) " if qualifier else ""
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
    _CALLABLE_PARAMETERS = "Payload payload @ callable_data"

    def raytracing(
        self, stage: str, buffers: list[Buffer], body: list[str], **parts
    ) -> str:
        hit = stage in ("any-hit", "closest-hit")
        callable_ = stage == "callable"
        # A ray-generation probe that traces declares its payload as a local.
        traced = parts.get("traced", False)
        lines = ["shader Probe {"]
        lines += [f"    {line}" for line in parts.get("prelude", ())]
        if hit or callable_ or traced:
            lines += ["    struct Payload {", "        vec4 color;", "    }"]
        lines += [f"    {_RAYTRACING_BLOCKS[stage]} {{"]
        lines += [f"        buffer {kind}* {name};" for kind, name in buffers]
        lines += [f"        {line}" for line in parts.get("declarations", ())]
        lines += [
            f"        void main({self._HIT_PARAMETERS if hit else self._CALLABLE_PARAMETERS if callable_ else ''}) {{",
            "            uint i = 0u;",
            *([f"            Payload {self.payload};"] if traced else ()),
        ]
        lines += [f"            {line}" for line in body]
        lines += ["        }", "    }", "}"]
        return "\n".join(lines) + "\n"

    def geometry(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        stream = parts.get("parameter")
        if stream:
            return self._streaming_geometry(stream, buffers, body, parts)
        lines = ["shader Probe {"]
        lines += [f"    {line}" for line in parts.get("prelude", ())]
        lines += ["    geometry {", *(f"        {line}" for line in _POINTS_IN_AND_OUT)]
        lines += [f"        buffer {kind}* {name};" for kind, name in buffers]
        lines += [f"        {line}" for line in parts.get("declarations", ())]
        lines += ["        void main() {", "            uint i = 0u;"]
        lines += [f"            {line}" for line in body]
        lines += ["        }", "    }", "}"]
        return "\n".join(lines) + "\n"

    def _streaming_geometry(
        self, stream: str, buffers: list[Buffer], body: list[str], parts: dict
    ) -> str:
        """A geometry stage written as Direct3D writes one, whose entry point takes its input point and the output
        stream ``obj``."""
        functions = [
            f"void main(point vec4 position[1] @ POSITION, inout {stream} obj)",
            "    @maxvertexcount(1) {",
            "    uint i = 0u;",
            f"    {VERTEX} {self.vertex};",
            f"    {self.vertex}.value = position[0];",
            *(f"    {line}" for line in body),
            "}",
        ]
        shared = (f"struct {VERTEX} {{", "    vec4 value @ TEXCOORD0;", "}")
        return self._program("geometry", shared, buffers, functions, parts)

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

    def absent(self, stage: str) -> str:
        return "CrossGL has no work-graph node stage" if stage == "node" else ""

    def type(self, kind: Type) -> str:
        # A Metal ballot returns a simd_vote or quad_vote mask (features.metal.VOTES), which no CrossGL type holds.
        if kind.opaque in ("simd_vote", "quad_vote"):
            raise NoSpelling(f"CrossGL has no type for Metal's {kind.opaque} mask")
        # CrossGL spells Slang's cooperative vectors, their enumerations and tuples as Slang does.
        if kind == COOP_COLUMNS:
            return super().type(kind.with_scalar("int"))
        if coop_element(kind.opaque):
            return f"CoopVec<{kind.scalar}, {kind.shape[0]}>"
        if kind.opaque in COOP_ENUMERATORS:
            return kind.opaque
        held = tuple_element(kind.opaque)
        if held:
            return f"Tuple<{', '.join([kind.scalar] * held[1])}>"
        return super().type(kind)

    def literal(self, kind: Type) -> str:
        return coop_literal(kind) or super().literal(kind)

    def address(self, name: str) -> str:
        return f"&{name}"

    def pointer(self, kind: Type, name: str) -> str:
        # CrossGL declares a pointer to device memory as a uniform of a pointer type, as Slang does.
        return f"uniform {self.type(kind)}* {name};"

    def assignment(
        self, name: str, kind: Type, arguments: list[str]
    ) -> tuple[list[str], str] | None:
        if name != ELEMENT:
            return None
        vector, index, element = arguments
        return [self.declare(kind, "w", vector), f"w[{index}] = {element};"], "w"

    def call(self, name: str, arguments: list[str]) -> str:
        if name == SUBSCRIPT:
            vector, index = arguments
            return f"{vector}[{index}]"
        return super().call(name, arguments)

    def reading(self, opaque: str) -> tuple[Type, str] | None:
        element = coop_element(opaque)
        if element:
            return element, "{}[0]"
        held = tuple_element(opaque)
        return (held[0], "{}._0") if held else None

    # CrossGL spells Direct3D's hit object as HLSL does.
    def holding(self, opaque: str) -> str:
        return OBJECTS[HIT_OBJECT].spelling if opaque == HIT_OBJECT else ""


# A geometry stage's input and output primitives, as the CrossGL importers write them.
_POINTS_IN_AND_OUT = ("layout(points) in;", "layout(points, max_vertices = 1) out;")

DIALECT = CrossGL()


class CrossGLES(CrossGL):
    """The CrossGL probes of the WebGL output direction. GLSL ES 3.00 has no compute stage and no storage buffers, so
    a kernel probe is a fragment stage that reads its inputs from uniform arrays and writes a local array, as the
    GLSL ES probe does (:data:`.dialect.UNIFORM_ARRAY_LENGTH`). Its entry point returns nothing, so a probe body can
    return early."""

    def read(self, buffer: str, offset: int) -> str:
        return uniform_read(buffer, offset)

    def kernel(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        *inputs, (output, _) = buffers
        lines = ["shader Probe {"]
        lines += [f"    {line}" for line in parts.get("prelude", ())]
        lines += ["    fragment {"]
        lines += [
            f"        uniform {kind} {name}[{UNIFORM_ARRAY_LENGTH}];"
            for kind, name in inputs
        ]
        lines += [f"        {line}" for line in parts.get("declarations", ())]
        lines += [
            "        void main() {",
            f"            {output} outp[1];",
            "            uint i = 0u;",
        ]
        lines += [f"            {line}" for line in body]
        lines += ["        }", "    }", "}"]
        return "\n".join(lines) + "\n"

    def fragment(self, buffers: list[Buffer], body: list[str], **parts) -> str:
        if parts.get("inputs"):
            raise ProbeError("no shader-input probe for GLSL ES yet")
        return self.kernel(buffers, body, **parts)


ES_DIALECT = CrossGLES()


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
