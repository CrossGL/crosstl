"""Function and value-semantics probes in the languages that do not write them as the C family does.

GLSL, HLSL, WGSL, Rust and Mojo pass and assign arrays by value. C and C++ pass an array as a pointer and cannot
assign one, so the CUDA, HIP and OpenCL C kernels never write the parameter, and Metal uses ``metal::array``. WGSL,
Rust and OpenCL C have no overloading, and WGSL, Rust and Mojo no switch fallthrough. Mojo copies an array
only when asked to, with ``copy()``.
"""

from __future__ import annotations

from .base import ABSENT, C_FAMILY, Template

_ARRAY_PARAMETER = C_FAMILY["array-parameter"].body
_STRUCT_RETURN = C_FAMILY["struct-return"].body
# The C kernels' total() reads the same elements and leaves the caller's array alone.
_C_TOTAL = "float total(const float v[4]) {\n    return 0.0f + v[1] + v[2] + v[3];\n}"
_C_MAKE = "Pair make(float x) {\n    Pair p;\n    p.a = x;\n    p.b = x * 2.0f;\n    return p;\n}"
_C_PAIR = "struct Pair {\n    float a;\n    float b;\n};\n"

CUDA = {
    "array-parameter": Template(_ARRAY_PARAMETER, f"__device__ {_C_TOTAL}"),
    "array-copy": ABSENT,
    "struct-return": Template(_STRUCT_RETURN, f"{_C_PAIR}\n__device__ {_C_MAKE}"),
    "function-overloading": Template(
        C_FAMILY["function-overloading"].body,
        "__device__ float scaled(float x) {\n    return x * 2.0f;\n}\n\n"
        "__device__ float scaled(int x) {\n    return (float)x * 4.0f;\n}",
    ),
}

OPENCL = {
    "array-parameter": Template(
        _ARRAY_PARAMETER.replace("float(iin[i] + k)", "(float)(iin[i] + k)"), _C_TOTAL
    ),
    "array-copy": ABSENT,
    "struct-return": Template(
        _STRUCT_RETURN,
        "typedef struct {\n    float a;\n    float b;\n} Pair;\n\n" + _C_MAKE,
    ),
    "function-overloading": ABSENT,
}

METAL = {
    "array-parameter": Template(
        _ARRAY_PARAMETER.replace("float values[4];", "array<float, 4> values;"),
        "float total(array<float, 4> v) {\n    v[0] = 0.0;\n    return v[0] + v[1] + v[2] + v[3];\n}",
    ),
    "array-copy": Template(
        C_FAMILY["array-copy"]
        .body.replace("float a[4];", "array<float, 4> a;")
        .replace("float b[4] = a;", "array<float, 4> b = a;")
    ),
}

_WGSL_FILL = "for (var k = 0; k < 4; k++) {\n    NAME[k] = f32(iin[i] + k);\n}\n"
_WGSL_PAIR = "struct Pair {\n    a: f32,\n    b: f32,\n}"

WGSL = {
    "array-parameter": Template(
        "var values: array<f32, 4>;\n"
        + _WGSL_FILL.replace("NAME", "values")
        + "outp[i] = total(values) * 10.0 + values[0];",
        "fn total(v_in: array<f32, 4>) -> f32 {\n    var v = v_in;\n    v[0] = 0.0;\n"
        "    return v[0] + v[1] + v[2] + v[3];\n}",
    ),
    "array-copy": Template(
        "var a: array<f32, 4>;\n"
        + _WGSL_FILL.replace("NAME", "a")
        + "var b = a;\nb[0] = 9.0;\noutp[i] = a[0] * 10.0 + b[0] + b[3];"
    ),
    "struct-return": Template(
        "let p = make(fin[i]);\noutp[i] = p.a + p.b * 4.0;",
        _WGSL_PAIR
        + "\n\nfn make(x: f32) -> Pair {\n    var p: Pair;\n    p.a = x;\n    p.b = x * 2.0;\n"
        "    return p;\n}",
    ),
    "function-overloading": ABSENT,
    "switch-fallthrough": ABSENT,
}

_RUST_FILL = "for k in 0..4 {\n    NAME[k] = (iin[i] + k as i32) as f32;\n}\n"

RUST = {
    "array-parameter": Template(
        "let mut values = [0.0; 4];\n"
        + _RUST_FILL.replace("NAME", "values")
        + "outp[i] = total(values) * 10.0 + values[0];",
        "fn total(mut v: [f32; 4]) -> f32 {\n    v[0] = 0.0;\n    v[0] + v[1] + v[2] + v[3]\n}",
    ),
    "array-copy": Template(
        "let mut a = [0.0; 4];\n"
        + _RUST_FILL.replace("NAME", "a")
        + "let mut b = a;\nb[0] = 9.0;\noutp[i] = a[0] * 10.0 + b[0] + b[3];"
    ),
    "struct-return": Template(
        "let p = make(fin[i]);\noutp[i] = p.a + p.b * 4.0;",
        "#[derive(Clone, Copy, Default)]\nstruct Pair {\n    a: f32,\n    b: f32,\n}\n\n"
        "fn make(x: f32) -> Pair {\n    Pair { a: x, b: x * 2.0 }\n}",
    ),
    "function-overloading": ABSENT,
    "switch-fallthrough": ABSENT,
}

_MOJO_FILL = "for k in range(4):\n    NAME[k] = Float32(Int(iin[i]) + k)\n"

MOJO = {
    "array-parameter": Template(
        "var values = Array[Float32, 4](fill=0.0)\n"
        + _MOJO_FILL.replace("NAME", "values")
        + "outp[i] = total(values.copy()) * 10.0 + values[0]",
        "def total(var v: Array[Float32, 4]) -> Float32:\n    v[0] = 0.0\n    return v[0] + v[1] + v[2] + v[3]",
    ),
    "array-copy": Template(
        "var a = Array[Float32, 4](fill=0.0)\n"
        + _MOJO_FILL.replace("NAME", "a")
        + "var b = a.copy()\nb[0] = 9.0\noutp[i] = a[0] * 10.0 + b[0] + b[3]"
    ),
    "struct-return": Template(
        "var p = make(fin[i])\noutp[i] = p.a + p.b * 4.0",
        "@fieldwise_init\nstruct Pair(Copyable, Movable):\n    var a: Float32\n    var b: Float32\n\n\n"
        "def make(x: Float32) -> Pair:\n    return Pair(x, x * 2.0)",
    ),
    "function-overloading": Template(
        "outp[i] = scaled(fin[i]) + scaled(iin[i])",
        "def scaled(x: Float32) -> Float32:\n    return x * 2.0\n\n\n"
        "def scaled(x: Int32) -> Float32:\n    return Float32(x) * 4.0",
    ),
    "switch-fallthrough": ABSENT,
}
