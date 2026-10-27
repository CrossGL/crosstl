"""Constructor, copy and vector-comparison probes in the languages that do not write them as the C family does.

HLSL builds a matrix from vectors by rows, splats a scalar only through a cast and has no struct constructors, only
initializer lists. CUDA and HIP vector types are plain structs built from scalars, with no operators, and CUDA, HIP,
OpenCL C and Mojo have no matrices. OpenCL C builds vectors with vector literals, and its vector comparisons give
``int`` vectors of -1 and 0. Rust builds vectors with glam's methods, and Mojo joins SIMD values.
"""

from __future__ import annotations

from .base import ABSENT, C_FAMILY, Template

_C_PAIR = "struct Pair {\n    float a;\n    float b;\n};"
_C_STRUCT_CONSTRUCTOR = "Pair p = {fin[i], 2.0f};\nPair q = {p.b, p.a * 3.0f};\noutp[i] = p.a + q.a * 10.0f + q.b * 100.0f;"
_C_STRUCT_COPY = (
    "Pair a;\na.a = fin[i];\na.b = 2.0f;\nPair b = a;\nb.a = 9.0f;\n"
    "outp[i] = a.a * 10.0f + b.a + b.b;"
)

# GLSL's relational operators take scalars only; its vector comparisons are functions.
GLSL = {
    "vector-comparison": Template(
        C_FAMILY["vector-comparison"]
        .body.replace("a < $vec2(0.0, 8.0)", "lessThan(a, $vec2(0.0, 8.0))")
        .replace("a == $vec2(fin[i], 4.0)", "equal(a, $vec2(fin[i], 4.0))")
    ),
}

# A matrix built from the CrossGL columns (f, 1) and (2, 3) has those vectors as its HLSL rows, so it is written with
# the vectors (f, 2) and (1, 3), and CrossGL's m[c][r] is HLSL's m[r][c].
HLSL = {
    "vector-constructor": Template(
        C_FAMILY["vector-constructor"].body.replace(
            "$vec3(f + 1.0)", "($vec3)(f + 1.0)"
        )
    ),
    "matrix-from-vectors": Template(
        "float f = float(iin[i]);\nfloat2x2 m = float2x2(float2(f, 2.0), float2(1.0, 3.0));\n"
        "outp[i] = m[1][0] * 100.0 + m[0][1] * 10.0 + m[0][0];"
    ),
    "struct-constructor": Template(
        "Pair p = {fin[i], 2.0};\nPair q = {p.b, p.a * 3.0};\noutp[i] = p.a + q.a * 10.0 + q.b * 100.0;",
        _C_PAIR,
    ),
}

METAL = {
    "struct-constructor": Template(
        "Pair p = Pair{fin[i], 2.0};\nPair q = Pair{p.b, p.a * 3.0};\n"
        "outp[i] = p.a + q.a * 10.0 + q.b * 100.0;",
        _C_PAIR,
    ),
}

CUDA = {
    "vector-constructor": ABSENT,
    "matrix-from-vectors": ABSENT,
    "vector-comparison": ABSENT,
    "struct-constructor": Template(_C_STRUCT_CONSTRUCTOR, _C_PAIR),
    "struct-copy": Template(_C_STRUCT_COPY, _C_PAIR),
}

_OPENCL_PAIR = "typedef struct {\n    float a;\n    float b;\n} Pair;"

OPENCL = {
    "vector-constructor": Template(
        "float f = fin[i];\nfloat2 h = (float2)(f, 1.0f);\nfloat4 v = (float4)(h, 2.0f, f);\n"
        "float3 s = (float3)(f + 1.0f);\n"
        "outp[i] = v.x * 1000.0f + v.y * 100.0f + v.z * 10.0f + v.w + s.z * 0.5f;"
    ),
    "matrix-from-vectors": ABSENT,
    "vector-comparison": Template(
        "float2 a = (float2)(fin[i], (float)iin[i]);\nint2 lt = a < (float2)(0.0f, 8.0f);\n"
        "int2 eq = a == (float2)(fin[i], 4.0f);\nfloat r = 0.0f;\nif (lt.x) {\n    r += 1.0f;\n}\n"
        "if (any(lt)) {\n    r += 2.0f;\n}\nif (all(lt)) {\n    r += 4.0f;\n}\nif (all(eq)) {\n    r += 8.0f;\n}\n"
        "outp[i] = r;"
    ),
    "struct-constructor": Template(_C_STRUCT_CONSTRUCTOR, _OPENCL_PAIR),
    "struct-copy": Template(_C_STRUCT_COPY, _OPENCL_PAIR),
}

_WGSL_PAIR = "struct Pair {\n    a: f32,\n    b: f32,\n}"

WGSL = {
    "vector-constructor": Template(
        "let f = fin[i];\nlet h = vec2<f32>(f, 1.0);\nlet v = vec4<f32>(h, 2.0, f);\nlet s = vec3<f32>(f + 1.0);\n"
        "outp[i] = v.x * 1000.0 + v.y * 100.0 + v.z * 10.0 + v.w + s.z * 0.5;"
    ),
    "matrix-from-vectors": Template(
        "let f = f32(iin[i]);\nlet m = mat2x2<f32>(vec2<f32>(f, 1.0), vec2<f32>(2.0, 3.0));\n"
        "outp[i] = m[0][1] * 100.0 + m[1][0] * 10.0 + m[0][0];"
    ),
    "vector-comparison": Template(
        "let a = vec2<f32>(fin[i], f32(iin[i]));\nlet lt = a < vec2<f32>(0.0, 8.0);\n"
        "let eq = a == vec2<f32>(fin[i], 4.0);\nvar r = 0.0;\nif (lt.x) {\n    r += 1.0;\n}\n"
        "if (any(lt)) {\n    r += 2.0;\n}\nif (all(lt)) {\n    r += 4.0;\n}\nif (all(eq)) {\n    r += 8.0;\n}\n"
        "outp[i] = r;"
    ),
    "struct-constructor": Template(
        "let p = Pair(fin[i], 2.0);\nlet q = Pair(p.b, p.a * 3.0);\noutp[i] = p.a + q.a * 10.0 + q.b * 100.0;",
        _WGSL_PAIR,
    ),
    "struct-copy": Template(
        "var a: Pair;\na.a = fin[i];\na.b = 2.0;\nvar b = a;\nb.a = 9.0;\noutp[i] = a.a * 10.0 + b.a + b.b;",
        _WGSL_PAIR,
    ),
}

_RUST_PAIR = (
    "#[derive(Clone, Copy, Default)]\nstruct Pair {\n    a: f32,\n    b: f32,\n}"
)

RUST = {
    "vector-constructor": Template(
        "let f = fin[i];\nlet h = $vec2::new(f, 1.0);\nlet v = h.extend(2.0).extend(f);\n"
        "let s = $vec3::splat(f + 1.0);\n"
        "outp[i] = v.x * 1000.0 + v.y * 100.0 + v.z * 10.0 + v.w + s.z * 0.5;"
    ),
    "matrix-from-vectors": Template(
        "let f = iin[i] as f32;\nlet m = $mat2::from_cols($vec2::new(f, 1.0), $vec2::new(2.0, 3.0));\n"
        "outp[i] = m.col(0).y * 100.0 + m.col(1).x * 10.0 + m.col(0).x;"
    ),
    "vector-comparison": Template(
        "let a = $vec2::new(fin[i], iin[i] as f32);\nlet lt = a.cmplt($vec2::new(0.0, 8.0));\n"
        "let eq = a.cmpeq($vec2::new(fin[i], 4.0));\nlet mut r = 0.0;\nif lt.x {\n    r += 1.0;\n}\n"
        "if lt.any() {\n    r += 2.0;\n}\nif lt.all() {\n    r += 4.0;\n}\nif eq.all() {\n    r += 8.0;\n}\n"
        "outp[i] = r;"
    ),
    "struct-constructor": Template(
        "let p = Pair { a: fin[i], b: 2.0 };\nlet q = Pair { a: p.b, b: p.a * 3.0 };\n"
        "outp[i] = p.a + q.a * 10.0 + q.b * 100.0;",
        _RUST_PAIR,
    ),
    "struct-copy": Template(
        "let mut a = Pair::default();\na.a = fin[i];\na.b = 2.0;\nlet mut b = a;\nb.a = 9.0;\n"
        "outp[i] = a.a * 10.0 + b.a + b.b;",
        _RUST_PAIR,
    ),
}

_MOJO_PAIR = "@fieldwise_init\nstruct Pair(Copyable, Movable):\n    var a: Float32\n    var b: Float32"

MOJO = {
    # A SIMD value is built from scalars, from one scalar it splats, or by joining two halves.
    "vector-constructor": Template(
        "var f = fin[i]\nvar h = $vec2(f, 1.0)\nvar v = h.join($vec2(2.0, f))\nvar s = $vec4(f + 1.0)\n"
        "outp[i] = v[0] * 1000.0 + v[1] * 100.0 + v[2] * 10.0 + v[3] + s[2] * 0.5"
    ),
    "matrix-from-vectors": ABSENT,
    "vector-comparison": Template(
        "var a = $vec2(fin[i], Float32(iin[i]))\nvar lt = a.lt($vec2(0.0, 8.0))\n"
        "var eq = a.eq($vec2(fin[i], 4.0))\nvar r: Float32 = 0.0\nif lt[0]:\n    r += 1.0\n"
        "if lt.reduce_or():\n    r += 2.0\nif lt.reduce_and():\n    r += 4.0\nif eq.reduce_and():\n    r += 8.0\n"
        "outp[i] = r"
    ),
    "struct-constructor": Template(
        "var p = Pair(fin[i], 2.0)\nvar q = Pair(p.b, p.a * 3.0)\noutp[i] = p.a + q.a * 10.0 + q.b * 100.0",
        _MOJO_PAIR,
    ),
    "struct-copy": Template(
        "var a = Pair(0.0, 0.0)\na.a = fin[i]\na.b = 2.0\nvar b = a.copy()\nb.a = 9.0\n"
        "outp[i] = a.a * 10.0 + b.a + b.b",
        _MOJO_PAIR,
    ),
}
