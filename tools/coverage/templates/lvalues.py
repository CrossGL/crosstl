"""Inout-argument-lvalues, switch-case-scope, deferred-initialization, integer-literal-limits,
unsigned-vector-arithmetic and lvalue-increment probes in the languages that do not write them as the C family does.

Metal, OpenCL C, WGSL and Mojo cannot pass a vector component by reference, so their probes copy it into a local and
back. WGSL passes only whole variables by pointer, so it does the same for the array element and the member. CUDA
and HIP vector types are plain structs without operators, OpenCL C passes pointers and converts with casts, and
Rust and Mojo, which have neither ``switch`` nor ``++``, match and add one instead. Rust and Mojo reject a read of a
local that only a loop assigns, so they give it a value first.
"""

from __future__ import annotations

from .base import (
    ABSENT,
    C_FAMILY,
    COUNT_SUM,
    GROW,
    LVALUE_SUM,
    PAIR,
    Template,
)
from .statements import c_body, c_vector_body

_COMPONENT = "grow(v.y);"
_COPIED = "float t = v.y;\ngrow(t);\nv.y = t;"
_VALUE_SUM = "vals[0] * 4.0 + vals[1] * 16.0 + vals[2] * 64.0"
_LIMIT_SUM = (
    "{f}((hi - k) >> 24{u}) + {f}(lo >> 28{u}) * 256.0 + {f}((top - {uint}(k)) >> 28{u}) * 4096.0"
    " + {f}(mask >> 31{u}) * 65536.0 + {f}({int}({fff}) + k) * 131072.0"
)
_UVEC_SUM = (
    "{f}(c{x}) + {f}(c{y}) * 64.0 + {f}(d{x}) * 1024.0 + {f}(d{y} >> 24{u}) * 65536.0"
)


def _cuda_grow() -> str:
    return (
        PAIR + "\n__device__ void grow(float& a) {\n    a = a * 2.0f + 1.0f;\n}\n\n"
        "__device__ void copy2(float x, float& a, float& b) {\n    a = x;\n    b = x * 2.0f;\n}"
    )


METAL = {
    "inout-argument-lvalues": Template(
        C_FAMILY["inout-argument-lvalues"].body.replace(_COMPONENT, _COPIED),
        PAIR
        + "\n"
        + GROW.replace("inout float", "thread float&").replace(
            "out float", "thread float&"
        ),
    ),
}

CUDA = {
    "inout-argument-lvalues": Template(
        c_vector_body("inout-argument-lvalues", "unsigned int", "make_float{}("),
        _cuda_grow(),
    ),
    "switch-case-scope": Template(c_body("switch-case-scope", "unsigned int")),
    "deferred-initialization": Template(
        c_vector_body("deferred-initialization", "unsigned int", "make_float{}(")
    ),
    "integer-literal-limits": Template(
        c_body("integer-literal-limits", "unsigned int")
    ),
    "unsigned-vector-arithmetic": ABSENT,
    "lvalue-increment": Template(
        c_body("lvalue-increment", "unsigned int").replace("$ivec2(", "make_int2(")
    ),
}

OPENCL = {
    "inout-argument-lvalues": Template(
        c_vector_body("inout-argument-lvalues", "uint", "(float{})(")
        .replace(_COMPONENT, _COPIED)
        .replace("grow(", "grow(&")
        .replace("p.a, values", "&p.a, &values"),
        "typedef struct {\n    float a;\n    float b;\n} Pair;\n\n"
        "void grow(float* a) {\n    *a = *a * 2.0f + 1.0f;\n}\n\n"
        "void copy2(float x, float* a, float* b) {\n    *a = x;\n    *b = x * 2.0f;\n}",
    ),
    "switch-case-scope": Template(c_body("switch-case-scope", "uint")),
    "deferred-initialization": Template(
        c_vector_body("deferred-initialization", "uint", "(float{})(")
    ),
    "integer-literal-limits": Template(
        c_body("integer-literal-limits", "uint").replace(
            "int(0xFFFFFFF0u)", "(int)0xFFFFFFF0u"
        )
    ),
    "unsigned-vector-arithmetic": Template(
        "uint2 a = (uint2)((uint)iin[i], 4294967295u);\nuint2 b = a * 3u + (uint2)(1u, 2u);\n"
        "uint2 c = (b >> (uint2)(1u, 28u)) | (a & (uint2)(8u, 8u));\n"
        "uint2 d = b / (uint2)(5u, 7u) + b % (uint2)(4u, 3u);\n"
        "outp[i] = (float)c.x + (float)c.y * 64.0f + (float)d.x * 1024.0f + (float)(d.y >> 24u) * 65536.0f;"
    ),
    "lvalue-increment": Template(
        c_body("lvalue-increment", "uint").replace("$ivec2(", "(int2)(")
    ),
}

WGSL = {
    "inout-argument-lvalues": Template(
        "var values: array<f32, 3>;\nvalues[0] = 1.0;\nvalues[1] = 2.0;\nvalues[2] = 3.0;\nlet j = iin[i] % 3;\n"
        "var p: Pair;\np.a = 0.0;\np.b = 0.5;\nvar v = vec2<f32>(4.0, f32(iin[i]));\nvar t = values[j];\n"
        "grow(&t);\nvalues[j] = t;\nt = p.b;\ngrow(&t);\np.b = t;\nt = v.y;\ngrow(&t);\nv.y = t;\nvar s: f32;\n"
        f"var u: f32;\ncopy2(fin[i], &s, &u);\np.a = s;\nvalues[2 - j] = u;\noutp[i] = {LVALUE_SUM};",
        "struct Pair {\n    a: f32,\n    b: f32,\n}\n\nfn grow(a: ptr<function, f32>) {\n    *a = *a * 2.0 + 1.0;\n}\n\n"
        "fn copy2(x: f32, a: ptr<function, f32>, b: ptr<function, f32>) {\n    *a = x;\n    *b = x * 2.0;\n}",
    ),
    "switch-case-scope": Template(
        "var acc: f32 = 0.0;\nswitch (iin[i] & 3) {\n    case 0: {\n        let t = fin[i];\n        acc = t * 2.0;\n"
        "    }\n    case 1: {\n        let t = 3.0;\n        let n = iin[i] * 5;\n        acc = t + f32(n);\n    }\n"
        "    default: {\n        let t = iin[i] * 4;\n        acc = f32(t) + 0.5;\n    }\n}\noutp[i] = acc;"
    ),
    "deferred-initialization": Template(
        "var x: f32;\nif (iin[i] > 7) {\n    x = fin[i];\n} else {\n    x = 2.0;\n}\nvar last: i32;\n"
        "for (var k = 0; k < 3; k++) {\n    last = iin[i] + k;\n}\nvar v: vec2<f32>;\nv.x = x;\nv.y = f32(last);\n"
        "outp[i] = v.x + v.y * 16.0;"
    ),
    "integer-literal-limits": Template(
        "let hi = 2147483647;\nlet lo = -2147483647 - 1;\nlet top = 4294967295u;\nlet mask = 0x80000000u;\n"
        "let k = iin[i];\noutp[i] = "
        + _LIMIT_SUM.format(f="f32", u="u", uint="u32", int="i32", fff="0xFFFFFFF0u")
        + ";"
    ),
    "unsigned-vector-arithmetic": Template(
        "let a = vec2<u32>(u32(iin[i]), 4294967295u);\nlet b = a * 3u + vec2<u32>(1u, 2u);\n"
        "let c = (b >> vec2<u32>(1u, 28u)) | (a & vec2<u32>(8u, 8u));\n"
        "let d = b / vec2<u32>(5u, 7u) + b % vec2<u32>(4u, 3u);\n"
        "outp[i] = " + _UVEC_SUM.format(f="f32", x=".x", y=".y", u="u") + ";"
    ),
    "lvalue-increment": Template(
        "var counts = array<i32, 3>(0, 10, 20);\nlet j = iin[i] % 3;\ncounts[j]++;\ncounts[2]++;\nvar k = 0;\n"
        "var vals: array<f32, 3>;\nvals[k] = fin[i];\nk++;\nvals[k] = 2.0;\nk++;\nvals[k] = f32(k);\n"
        f"var c = vec2<i32>(iin[i], 5);\nc.y++;\nc.x--;\noutp[i] = f32({COUNT_SUM}) + {_VALUE_SUM};"
    ),
}

RUST = {
    "inout-argument-lvalues": Template(
        "let mut values = [1.0, 2.0, 3.0];\nlet j = (iin[i] % 3) as usize;\nlet mut p = Pair { a: 0.0, b: 0.5 };\n"
        "let mut v = $vec2::new(4.0, iin[i] as f32);\ngrow(&mut values[j]);\ngrow(&mut p.b);\ngrow(&mut v.y);\n"
        f"copy2(fin[i], &mut p.a, &mut values[2 - j]);\noutp[i] = {LVALUE_SUM};",
        "#[derive(Clone, Copy, Default)]\nstruct Pair {\n    a: f32,\n    b: f32,\n}\n\n"
        "fn grow(a: &mut f32) {\n    *a = *a * 2.0 + 1.0;\n}\n\n"
        "fn copy2(x: f32, a: &mut f32, b: &mut f32) {\n    *a = x;\n    *b = x * 2.0;\n}",
    ),
    "switch-case-scope": Template(
        "let acc = match iin[i] & 3 {\n    0 => {\n        let t = fin[i];\n        t * 2.0\n    }\n    1 => {\n"
        "        let t = 3.0;\n        let n = iin[i] * 5;\n        t + n as f32\n    }\n    _ => {\n"
        "        let t = iin[i] * 4;\n        t as f32 + 0.5\n    }\n};\noutp[i] = acc;"
    ),
    "deferred-initialization": Template(
        "let x;\nif iin[i] > 7 {\n    x = fin[i];\n} else {\n    x = 2.0;\n}\nlet mut last = 0;\nlet mut k = 0;\n"
        "while k < 3 {\n    last = iin[i] + k;\n    k += 1;\n}\nlet mut v = $vec2::ZERO;\nv.x = x;\nv.y = last as f32;\n"
        "outp[i] = v.x + v.y * 16.0;"
    ),
    "integer-literal-limits": Template(
        "let hi: i32 = 2147483647;\nlet lo: i32 = -2147483647 - 1;\nlet top: u32 = 4294967295;\n"
        "let mask: u32 = 0x80000000;\nlet k = iin[i];\n"
        "outp[i] = ((hi - k) >> 24) as f32 + (lo >> 28) as f32 * 256.0 + ((top - k as u32) >> 28) as f32 * 4096.0"
        " + (mask >> 31) as f32 * 65536.0 + (0xFFFFFFF0u32 as i32 + k) as f32 * 131072.0;"
    ),
    "unsigned-vector-arithmetic": Template(
        "let a = $uvec2::new(iin[i] as u32, 4294967295);\n"
        "let b = a.wrapping_mul($uvec2::splat(3)).wrapping_add($uvec2::new(1, 2));\n"
        "let c = (b >> $uvec2::new(1, 28)) | (a & $uvec2::new(8, 8));\n"
        "let d = b / $uvec2::new(5, 7) + b % $uvec2::new(4, 3);\n"
        "outp[i] = c.x as f32 + c.y as f32 * 64.0 + d.x as f32 * 1024.0 + (d.y >> 24) as f32 * 65536.0;"
    ),
    "lvalue-increment": Template(
        "let mut counts = [0, 10, 20];\nlet j = (iin[i] % 3) as usize;\ncounts[j] += 1;\ncounts[2] += 1;\n"
        "let mut k = 0;\nlet mut vals = [0.0; 3];\nvals[k] = fin[i];\nk += 1;\nvals[k] = 2.0;\nk += 1;\n"
        "vals[k] = k as f32;\nlet mut c = $ivec2::new(iin[i], 5);\nc.y += 1;\nc.x -= 1;\n"
        "outp[i] = (counts[0] + counts[1] * 8 + counts[2] * 256 + k as i32 * 8192 + c.x * 16384 + c.y * 524288)"
        f" as f32 + {_VALUE_SUM};"
    ),
}

MOJO = {
    "inout-argument-lvalues": Template(
        "var values = Array[Float32, 3](fill=0.0)\nvalues[0] = 1.0\nvalues[1] = 2.0\nvalues[2] = 3.0\n"
        "var j = Int(iin[i] % 3)\nvar p = Pair(0.0, 0.5)\nvar v = $vec2(4.0, Float32(iin[i]))\ngrow(values[j])\n"
        "grow(p.b)\nvar t = v[1]\ngrow(t)\nv[1] = t\ncopy2(fin[i], p.a, values[2 - j])\n"
        "outp[i] = " + LVALUE_SUM.replace("v.y", "v[1]"),
        "@fieldwise_init\nstruct Pair(Copyable, Movable):\n    var a: Float32\n    var b: Float32\n\n\n"
        "def grow(mut a: Float32):\n    a = a * 2.0 + 1.0\n\n\n"
        "def copy2(x: Float32, mut a: Float32, mut b: Float32):\n    a = x\n    b = x * 2.0",
    ),
    "switch-case-scope": ABSENT,
    "deferred-initialization": Template(
        "var x: Float32\nif iin[i] > 7:\n    x = fin[i]\nelse:\n    x = 2.0\nvar last: Int32 = 0\n"
        "for k in range(3):\n    last = iin[i] + Int32(k)\nvar v = $vec2(0.0, 0.0)\nv[0] = x\nv[1] = Float32(last)\n"
        "outp[i] = v[0] + v[1] * 16.0"
    ),
    "integer-literal-limits": Template(
        "var hi: Int32 = 2147483647\nvar lo: Int32 = -2147483647 - 1\nvar top: UInt32 = 4294967295\n"
        "var mask: UInt32 = 0x80000000\nvar k = iin[i]\noutp[i] = "
        + _LIMIT_SUM.format(
            f="Float32", u="", uint="UInt32", int="Int32", fff="UInt32(0xFFFFFFF0)"
        )
    ),
    "unsigned-vector-arithmetic": Template(
        "var a = $uvec2(UInt32(iin[i]), 4294967295)\nvar b = a * 3 + $uvec2(1, 2)\n"
        "var c = (b >> $uvec2(1, 28)) | (a & $uvec2(8, 8))\nvar d = b // $uvec2(5, 7) + b % $uvec2(4, 3)\n"
        "outp[i] = " + _UVEC_SUM.format(f="Float32", x="[0]", y="[1]", u="")
    ),
    "lvalue-increment": Template(
        "var counts = Array[Int32, 3](fill=0)\ncounts[1] = 10\ncounts[2] = 20\nvar j = Int(iin[i] % 3)\n"
        "counts[j] += 1\ncounts[2] += 1\nvar k = 0\nvar vals = Array[Float32, 3](fill=0.0)\nvals[k] = fin[i]\n"
        "k += 1\nvals[k] = 2.0\nk += 1\nvals[k] = Float32(k)\nvar c = $ivec2(iin[i], 5)\nc[1] += 1\nc[0] -= 1\n"
        "outp[i] = Float32(counts[0] + counts[1] * 8 + counts[2] * 256 + Int32(k) * 8192 + c[0] * 16384"
        f" + c[1] * 524288) + {_VALUE_SUM}"
    ),
}
