"""Vector-buffer, struct-buffer, struct-buffer-copy, uniform-members, uniform-vector and uniform-vector-packing
probes: buffers whose elements are not scalars and constant blocks with more than one word, in every language.

The first three write a buffer of four-component vectors or of ``Pair`` structs, so a translation has to keep the
element's stride and member offsets for the host to read what the native probe writes. struct-buffer-copy stores a
whole struct into a buffer element and loads it back, which SPIR-V and WGSL do between a struct with explicit offsets
and one without. WebGL 2 has no storage buffers, so it has none of these.

The uniform probes read every word of the 16-byte constants block the harness binds: four floats, one four-component
vector, or a three-component vector followed by a float. In std140, a cbuffer and WGSL's uniform layout that float
packs into the vector's last four bytes, at offset 12. Metal aligns ``float3`` to 16 bytes and OpenCL C sizes it as
16, so Metal's probe declares a ``packed_float3`` and OpenCL C's struct puts the float at offset 16, a layout of its
own. Metal, OpenCL C, Mojo and Rust receive
the block as an entry-point parameter; CUDA and HIP declare it in constant memory.
"""

from __future__ import annotations

from dataclasses import replace

from .base import ABSENT, PAIR, Template
from .statements import c_text

NAMES = (
    "vector-buffer",
    "struct-buffer",
    "struct-buffer-copy",
    "uniform-members",
    "uniform-vector",
    "uniform-vector-packing",
)

_VECTOR = (
    "outp[i] = $vec4(fin[i], fin[i] * 2.0, 1.0, -0.5);\noutp[i].y += 0.25;\n"
    "outp[i].w = outp[i].x * outp[i].z;"
)
_MEMBERS = "outp[i].a = fin[i];\noutp[i].b = outp[i].a * 2.0 + 1.0;"
_COPY = (
    "Pair p;\np.a = fin[i] * 0.5;\np.b = p.a + 1.0;\noutp[i] = p;\nPair q = outp[i];\n"
    "outp[i].b = q.a * q.b + 0.25;"
)

_WEIGHTS = "outp[i] = fin[i] * {0}w0 + {0}w1 * {0}w2 - {0}w3;"
_SPREAD = "$vec4 t = {0}v * fin[i];\noutp[i] = t.x + t.y * 2.0 + t.z * 4.0 + t.w * 8.0;"
_PACKED = "outp[i] = (fin[i] + {0}a.x) * {0}a.y - {0}a.z * {0}b;"

# The members of each constants block, as C declarations.
_BLOCKS = {
    "uniform-members": "    float w0;\n    float w1;\n    float w2;\n    float w3;",
    "uniform-vector": "    $vec4 v;",
    "uniform-vector-packing": "    $vec3 a;\n    float b;",
}
_USES = {
    "uniform-members": _WEIGHTS,
    "uniform-vector": _SPREAD,
    "uniform-vector-packing": _PACKED,
}


def _storage(struct: str) -> dict[str, Template]:
    """The three buffer probes of a language that writes C-like vector constructors, with ``struct`` the declaration
    of ``Pair``, which has to precede a buffer declared at program scope."""
    return {
        "vector-buffer": Template(_VECTOR, output="$vec4"),
        "struct-buffer": Template(_MEMBERS, prelude=struct, output="Pair"),
        "struct-buffer-copy": Template(_COPY, prelude=struct, output="Pair"),
    }


def _blocks(head: str, tail: str, prefix: str = "") -> dict[str, Template]:
    """The uniform probes of a language that declares the block ``head`` members ``tail`` at program scope."""
    return {
        name: Template(_USES[name].format(prefix), f"{head}\n{members}\n{tail}")
        for name, members in _BLOCKS.items()
    }


_GLSL_HEAD = "layout(std140, binding = 3) uniform Params {"

GLSL = {
    **_storage(PAIR),
    **_blocks(_GLSL_HEAD, "};"),
}

FAMILY = {
    **_storage(PAIR),
    **_blocks("cbuffer Params : register(b3) {", "};"),
}

# CrossGL declares structs and the constants block at shader scope.
CROSSGL = {
    **_storage(PAIR),
    **{
        name: Template(
            _USES[name].format(""),
            prelude=f"cbuffer Params @binding(3) {{\n{members}\n}};",
        )
        for name, members in _BLOCKS.items()
    },
}

WEBGL = {
    **dict.fromkeys(NAMES[:3], ABSENT),
    **_blocks("layout(std140) uniform Params {", "};"),
}

_METAL_BLOCKS = {
    **_BLOCKS,
    "uniform-vector-packing": "    packed_float3 a;\n    float b;",
}

METAL = {
    **_storage(PAIR),
    **{
        name: Template(
            _USES[name].format("params."),
            f"struct Params {{\n{members}\n}};",
            params="constant Params& params [[buffer(3)]]",
        )
        for name, members in _METAL_BLOCKS.items()
    },
}

_WGSL_PAIR = "struct Pair {\n    a: f32,\n    b: f32,\n}"
_WGSL_BLOCKS = {
    "uniform-members": "    w0: f32,\n    w1: f32,\n    w2: f32,\n    w3: f32,",
    "uniform-vector": "    v: vec4<f32>,",
    "uniform-vector-packing": "    a: vec3<f32>,\n    b: f32,",
}
_WGSL_USES = {
    **_USES,
    "uniform-vector": (
        "let t = params.v * fin[i];\noutp[i] = t.x + t.y * 2.0 + t.z * 4.0 + t.w * 8.0;"
    ),
}

WGSL = {
    "vector-buffer": Template(_VECTOR, output="$vec4"),
    "struct-buffer": Template(_MEMBERS, _WGSL_PAIR, output="Pair"),
    "struct-buffer-copy": Template(
        "var p: Pair;\np.a = fin[i] * 0.5;\np.b = p.a + 1.0;\noutp[i] = p;\nlet q = outp[i];\n"
        "outp[i].b = q.a * q.b + 0.25;",
        _WGSL_PAIR,
        output="Pair",
    ),
    **{
        name: Template(
            _WGSL_USES[name].format("params."),
            f"struct Params {{\n{members}\n}}\n@group(0) @binding(3) var<uniform> params: Params;",
        )
        for name, members in _WGSL_BLOCKS.items()
    },
}


def _c(construct: str, pair: str, block: str) -> dict[str, Template]:
    """The probes CUDA, HIP and OpenCL C share, in C with casts: ``construct`` opens a four-component vector,
    ``pair`` declares ``Pair`` and ``block`` formats the declaration of the struct ``Params`` from its members.
    """

    def text(source: str) -> str:
        return c_text(source.replace("$vec4(", construct), "uint")

    return {
        "vector-buffer": Template(text(_VECTOR), output="$vec4"),
        "struct-buffer": Template(text(_MEMBERS), pair, output="Pair"),
        "struct-buffer-copy": Template(text(_COPY), pair, output="Pair"),
        "uniform-members": Template(
            text(_WEIGHTS.format("params.")), block.format(_BLOCKS["uniform-members"])
        ),
        "uniform-vector-packing": Template(
            text(_PACKED.format("params.")),
            block.format(_BLOCKS["uniform-vector-packing"]),
        ),
    }


_CUDA_BLOCK = "struct Params {{\n{}\n}};\n__constant__ Params params;"

# CUDA's vector types have no arithmetic operators, so its uniform-vector probe scales each component.
CUDA = {
    **_c("make_float4(", PAIR, _CUDA_BLOCK),
    "uniform-vector": Template(
        c_text(
            "float4 t = make_float4(v.x * fin[i], v.y * fin[i], v.z * fin[i], v.w * fin[i]);\n"
            "outp[i] = t.x + t.y * 2.0 + t.z * 4.0 + t.w * 8.0;",
            "uint",
        ),
        "__constant__ float4 v;",
    ),
}

_OPENCL_PAIR = "typedef struct {\n    float a;\n    float b;\n} Pair;"
_OPENCL_BLOCK = "typedef struct {{\n{}\n}} Params;"

OPENCL = {
    **{
        name: (
            replace(template, params="const Params params")
            if name.startswith("uniform")
            else template
        )
        for name, template in _c("(float4)(", _OPENCL_PAIR, _OPENCL_BLOCK).items()
    },
    "uniform-vector": Template(
        c_text(_SPREAD.format(""), "uint"), params="const $vec4 v"
    ),
}

_RUST_DERIVE = "#[derive(Clone, Copy, Default)]\n#[repr(C)]\n"
_RUST_PAIR = f"{_RUST_DERIVE}pub struct Pair {{\n    a: f32,\n    b: f32,\n}}"
_RUST_UNIFORM = "#[spirv(uniform, descriptor_set = 0, binding = 3)] "

RUST = {
    "vector-buffer": Template(_VECTOR.replace("$vec4(", "$vec4::new("), output="$vec4"),
    "struct-buffer": Template(_MEMBERS, _RUST_PAIR, output="Pair"),
    "struct-buffer-copy": Template(
        "let mut p = Pair::default();\np.a = fin[i] * 0.5;\np.b = p.a + 1.0;\noutp[i] = p;\nlet q = outp[i];\n"
        "outp[i].b = q.a * q.b + 0.25;",
        _RUST_PAIR,
        output="Pair",
    ),
    "uniform-members": Template(
        _WEIGHTS.format("params."),
        f"{_RUST_DERIVE}pub struct Params {{\n    w0: f32,\n    w1: f32,\n    w2: f32,\n    w3: f32,\n}}",
        params=f"{_RUST_UNIFORM}params: &Params",
    ),
    "uniform-vector": Template(
        "let t = *v * fin[i];\noutp[i] = t.x + t.y * 2.0 + t.z * 4.0 + t.w * 8.0;",
        params=f"{_RUST_UNIFORM}v: &Vec4",
    ),
    "uniform-vector-packing": Template(
        _PACKED.format("params."),
        f"{_RUST_DERIVE}pub struct Params {{\n    a: Vec3,\n    b: f32,\n}}",
        params=f"{_RUST_UNIFORM}params: &Params",
    ),
}

_MOJO_VEC4 = "SIMD[DType.float32, 4]"
_MOJO_PAIR = (
    "@fieldwise_init\nstruct Pair(TrivialRegisterPassable):\n"
    "    var a: Float32\n    var b: Float32"
)

MOJO = {
    "vector-buffer": Template(
        f"outp[i] = {_MOJO_VEC4}(fin[i], fin[i] * 2.0, 1.0, -0.5)\noutp[i][1] += 0.25\n"
        "outp[i][3] = outp[i][0] * outp[i][2]",
        output="$vec4",
    ),
    "struct-buffer": Template(
        "outp[i].a = fin[i]\noutp[i].b = outp[i].a * 2.0 + 1.0",
        _MOJO_PAIR,
        output="Pair",
    ),
    "struct-buffer-copy": Template(
        "var p = Pair(0.0, 0.0)\np.a = fin[i] * 0.5\np.b = p.a + 1.0\noutp[i] = p\nvar q = outp[i]\n"
        "outp[i].b = q.a * q.b + 0.25",
        _MOJO_PAIR,
        output="Pair",
    ),
    "uniform-members": Template(
        _WEIGHTS.format("params.").rstrip(";"),
        "@fieldwise_init\nstruct Params(TrivialRegisterPassable):\n"
        "    var w0: Float32\n    var w1: Float32\n    var w2: Float32\n    var w3: Float32",
        params="params: Params",
    ),
    "uniform-vector": Template(
        "var t = v * fin[i]\noutp[i] = t[0] + t[1] * 2.0 + t[2] * 4.0 + t[3] * 8.0",
        params=f"v: {_MOJO_VEC4}",
    ),
    "uniform-vector-packing": Template(
        _PACKED.format("params.")
        .rstrip(";")
        .replace(".x", "[0]")
        .replace(".y", "[1]")
        .replace(".z", "[2]"),
        "@fieldwise_init\nstruct Params(TrivialRegisterPassable):\n"
        "    var a: SIMD[DType.float32, 3]\n    var b: Float32",
        params="params: Params",
    ),
}
