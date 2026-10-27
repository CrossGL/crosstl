"""Infinite-loop, loop-condition-call, indexed-compound-assignment, vector-compound-assignment,
nested-member-assignment and bool-array probes in the languages that do not write them as the C family does.

Metal passes references as ``thread`` references. CUDA and HIP pass references, mark device functions ``__device__``
and have no vector operators; OpenCL C passes pointers, converts with casts, builds vectors with vector literals and
declares structs with ``typedef``. WGSL passes pointers, loops without a condition with ``loop``, shifts by unsigned
amounts and picks between values with ``select``. Rust loops without a condition with ``loop`` and indexes with
``usize``, and Rust and Mojo step a loop that calls a function in its condition with ``while``. Mojo has no ``for``
loop with a condition and writes vector members by index.
"""

from __future__ import annotations

import re

from .base import ABSENT, ADVANCE, BAG_SUM, C_FAMILY, SLOT, Template

_LITERAL = re.compile(r"\b(\d+\.\d+)\b")


def _f(source: str) -> str:
    """``source`` with single-precision literals."""
    return _LITERAL.sub(r"\1f", source)


def _c(name: str) -> str:
    """The C family's probe with casts in place of conversions and single-precision literals."""
    return _f(C_FAMILY[name].body.replace("float(", "(float)("))


def _pointers(name: str) -> str:
    """The C family's probe with casts, single-precision literals and the address of ``k`` for each reference."""
    return re.sub(r"\b(advance|slot)\(k\)", r"\1(&k)", _c(name))


_C_ADVANCE = "int advance(int* k) {\n    *k = *k + 5;\n    return *k;\n}"
_C_SLOT = "int slot(int* k) {\n    *k = *k + 1;\n    return *k;\n}"
_C_BAG = (
    "typedef struct {\n    float2 pos;\n    float w[2];\n} Item;\n"
    "typedef struct {\n    Item items[2];\n    int n;\n} Bag;"
)
_INDEXED_SUM = "a[0] + a[1] * 16.0 + a[2] * 256.0 + a[3] * 4096.0"

METAL = {
    "loop-condition-call": Template(
        C_FAMILY["loop-condition-call"].body,
        ADVANCE.replace("inout int k", "thread int& k"),
    ),
    "indexed-compound-assignment": Template(
        C_FAMILY["indexed-compound-assignment"].body,
        SLOT.replace("inout int k", "thread int& k"),
    ),
}

CUDA = {
    "infinite-loop": Template(_f(C_FAMILY["infinite-loop"].body)),
    "loop-condition-call": Template(
        _f(C_FAMILY["loop-condition-call"].body),
        "__device__ " + ADVANCE.replace("inout int k", "int& k"),
    ),
    "indexed-compound-assignment": Template(
        _f(C_FAMILY["indexed-compound-assignment"].body),
        "__device__ " + SLOT.replace("inout int k", "int& k"),
    ),
    "vector-compound-assignment": ABSENT,
    "nested-member-assignment": Template(
        _f(C_FAMILY["nested-member-assignment"].body),
        C_FAMILY["nested-member-assignment"].declarations,
    ),
    "bool-array": Template(_f(C_FAMILY["bool-array"].body)),
}

OPENCL = {
    "infinite-loop": Template(_c("infinite-loop")),
    "loop-condition-call": Template(_pointers("loop-condition-call"), _C_ADVANCE),
    "indexed-compound-assignment": Template(
        _pointers("indexed-compound-assignment"), _C_SLOT
    ),
    "vector-compound-assignment": Template(
        "float4 v = (float4)((float)iin[i], 1.0f, 2.0f, 3.0f);\nfloat4 w = (float4)(0.5f, 0.25f, 4.0f, 1.0f);\n"
        "v += w;\nv *= 2.0f;\nv -= w * 2.0f;\nv /= (float4)(2.0f, 4.0f, 8.0f, 2.0f);\nv *= w;\nv.y += 1.0f;\n"
        "outp[i] = v.x + v.y * 64.0f + v.z * 4096.0f + v.w * 65536.0f;"
    ),
    "nested-member-assignment": Template(_c("nested-member-assignment"), _C_BAG),
    "bool-array": Template(_c("bool-array")),
}

WGSL = {
    "infinite-loop": Template(
        "var k = iin[i];\nvar acc: f32 = 0.0;\nloop {\n    acc += 2.0;\n    k = k + 3;\n    if (k > 20) {\n"
        "        break;\n    }\n}\nwhile (true) {\n    acc += 64.0;\n    if (acc > 200.0) {\n        break;\n"
        "    }\n}\noutp[i] = acc + f32(k) * 1024.0;"
    ),
    "loop-condition-call": Template(
        "var k = iin[i];\nvar acc: f32 = 0.0;\nwhile (advance(&k) < 40) {\n    acc += 1.0;\n}\n"
        "for (var j = advance(&k); j < 64 && advance(&k) < 60; j += 4) {\n    acc += 32.0;\n}\n"
        "outp[i] = acc + f32(k) * 4096.0;",
        "fn advance(k: ptr<function, i32>) -> i32 {\n    *k = *k + 5;\n    return *k;\n}",
    ),
    "indexed-compound-assignment": Template(
        "var a: array<f32, 4>;\nfor (var j = 0; j < 4; j++) {\n    a[j] = f32(j);\n}\nvar k = iin[i] & 1;\n"
        "a[slot(&k)] += 8.0;\na[slot(&k)] *= 4.0;\na[k - 1] -= 0.5;\n"
        f"outp[i] = {_INDEXED_SUM} + f32(k) * 65536.0;",
        "fn slot(k: ptr<function, i32>) -> i32 {\n    *k = *k + 1;\n    return *k;\n}",
    ),
    "vector-compound-assignment": Template(
        "var v = vec4<f32>(f32(iin[i]), 1.0, 2.0, 3.0);\nlet w = vec4<f32>(0.5, 0.25, 4.0, 1.0);\n"
        "v += w;\nv *= 2.0;\nv -= w * 2.0;\nv /= vec4<f32>(2.0, 4.0, 8.0, 2.0);\nv *= w;\nv.y += 1.0;\n"
        "outp[i] = v.x + v.y * 64.0 + v.z * 4096.0 + v.w * 65536.0;"
    ),
    "nested-member-assignment": Template(
        "var b: Bag;\nfor (var j = 0; j < 2; j++) {\n    b.items[j].pos.x = f32(j);\n"
        "    b.items[j].pos.y = f32(j) + 2.0;\n    b.items[j].w[0] = 1.0;\n    b.items[j].w[1] = 3.0;\n}\n"
        "let k = iin[i] & 1;\nb.n = iin[i];\nb.items[k].pos.y = f32(b.n) * 2.0;\nb.items[1 - k].w[k] += 8.0;\n"
        f"b.items[k].w[1 - k] = b.items[1 - k].pos.x + 4.0;\noutp[i] = {BAG_SUM};",
        "struct Item {\n    pos: vec2<f32>,\n    w: array<f32, 2>,\n}\n\n"
        "struct Bag {\n    items: array<Item, 2>,\n    n: i32,\n}",
    ),
    "bool-array": Template(
        "var flags: array<bool, 4>;\nfor (var j = 0u; j < 4u; j++) {\n    flags[j] = ((iin[i] >> j) & 1) == 1;\n}\n"
        "let either = flags[0] || flags[3];\nflags[1] = !flags[1];\n"
        "outp[i] = select(0.0, 1.0, flags[1]) + select(0.0, 2.0, flags[2]) + select(0.0, 4.0, either)"
        " + f32(iin[i]) * 8.0;"
    ),
}

RUST = {
    "infinite-loop": Template(
        "let mut k = iin[i];\nlet mut acc = 0.0;\nloop {\n    acc += 2.0;\n    k += 3;\n    if k > 20 {\n"
        "        break;\n    }\n}\nloop {\n    acc += 64.0;\n    if acc > 200.0 {\n        break;\n    }\n}\n"
        "outp[i] = acc + k as f32 * 1024.0;"
    ),
    "loop-condition-call": Template(
        "let mut k = iin[i];\nlet mut acc = 0.0;\nwhile advance(&mut k) < 40 {\n    acc += 1.0;\n}\n"
        "let mut j = advance(&mut k);\nwhile j < 64 && advance(&mut k) < 60 {\n    acc += 32.0;\n    j += 4;\n}\n"
        "outp[i] = acc + k as f32 * 4096.0;",
        "fn advance(k: &mut i32) -> i32 {\n    *k += 5;\n    *k\n}",
    ),
    "indexed-compound-assignment": Template(
        "let mut a = [0.0f32; 4];\nfor j in 0..4 {\n    a[j] = j as f32;\n}\nlet mut k = iin[i] & 1;\n"
        "a[slot(&mut k)] += 8.0;\na[slot(&mut k)] *= 4.0;\na[(k - 1) as usize] -= 0.5;\n"
        f"outp[i] = {_INDEXED_SUM} + k as f32 * 65536.0;",
        "fn slot(k: &mut i32) -> usize {\n    *k += 1;\n    *k as usize\n}",
    ),
    "vector-compound-assignment": Template(
        "let mut v = $vec4::new(iin[i] as f32, 1.0, 2.0, 3.0);\nlet w = $vec4::new(0.5, 0.25, 4.0, 1.0);\n"
        "v += w;\nv *= 2.0;\nv -= w * 2.0;\nv /= $vec4::new(2.0, 4.0, 8.0, 2.0);\nv *= w;\nv.y += 1.0;\n"
        "outp[i] = v.x + v.y * 64.0 + v.z * 4096.0 + v.w * 65536.0;"
    ),
    "nested-member-assignment": Template(
        "let item = Item {\n    pos: $vec2::ZERO,\n    w: [0.0; 2],\n};\nlet mut b = Bag {\n    items: [item; 2],\n"
        "    n: 0,\n};\nfor j in 0..2 {\n    b.items[j].pos.x = j as f32;\n    b.items[j].pos.y = j as f32 + 2.0;\n"
        "    b.items[j].w[0] = 1.0;\n    b.items[j].w[1] = 3.0;\n}\nlet k = (iin[i] & 1) as usize;\nb.n = iin[i];\n"
        "b.items[k].pos.y = b.n as f32 * 2.0;\nb.items[1 - k].w[k] += 8.0;\n"
        f"b.items[k].w[1 - k] = b.items[1 - k].pos.x + 4.0;\noutp[i] = {BAG_SUM};",
        "#[derive(Clone, Copy)]\nstruct Item {\n    pos: $vec2,\n    w: [f32; 2],\n}\n\n"
        "#[derive(Clone, Copy)]\nstruct Bag {\n    items: [Item; 2],\n    n: i32,\n}",
    ),
    # rust-gpu cannot fill a bool array from a repeat expression such as [false; 4].
    "bool-array": Template(
        "let mut flags = [false, false, false, false];\nfor j in 0..4 {\n    flags[j] = ((iin[i] >> j as i32) & 1) == 1;\n}\n"
        "let either = flags[0] || flags[3];\nflags[1] = !flags[1];\n"
        "let r = if flags[1] { 1.0 } else { 0.0 };\nlet s = if flags[2] { 2.0 } else { 0.0 };\n"
        "let t = if either { 4.0 } else { 0.0 };\noutp[i] = r + s + t + iin[i] as f32 * 8.0;"
    ),
}

MOJO = {
    "infinite-loop": Template(
        "var k = iin[i]\nvar acc: Float32 = 0.0\nwhile True:\n    acc += 2.0\n    k = k + 3\n    if k > 20:\n"
        "        break\nwhile True:\n    acc += 64.0\n    if acc > 200.0:\n        break\n"
        "outp[i] = acc + Float32(k) * 1024.0"
    ),
    "loop-condition-call": Template(
        "var k: Int32 = iin[i]\nvar acc: Float32 = 0.0\nwhile advance(k) < 40:\n    acc += 1.0\n"
        "var j = advance(k)\nwhile j < 64 and advance(k) < 60:\n    acc += 32.0\n    j += 4\n"
        "outp[i] = acc + Float32(k) * 4096.0",
        "def advance(mut k: Int32) -> Int32:\n    k = k + 5\n    return k",
    ),
    "indexed-compound-assignment": Template(
        "var a = Array[Float32, 4](fill=0.0)\nfor j in range(4):\n    a[j] = Float32(j)\n"
        "var k: Int32 = iin[i] & 1\na[slot(k)] += 8.0\na[slot(k)] *= 4.0\na[Int(k - 1)] -= 0.5\n"
        f"outp[i] = {_INDEXED_SUM} + Float32(k) * 65536.0",
        "def slot(mut k: Int32) -> Int:\n    k = k + 1\n    return Int(k)",
    ),
    "vector-compound-assignment": Template(
        "var v = $vec4(Float32(iin[i]), 1.0, 2.0, 3.0)\nvar w = $vec4(0.5, 0.25, 4.0, 1.0)\n"
        "v += w\nv *= 2.0\nv -= w * 2.0\nv /= $vec4(2.0, 4.0, 8.0, 2.0)\nv *= w\nv[1] += 1.0\n"
        "outp[i] = v[0] + v[1] * 64.0 + v[2] * 4096.0 + v[3] * 65536.0"
    ),
    "nested-member-assignment": Template(
        "var item = Item($vec2(0.0, 0.0), Array[Float32, 2](fill=0.0))\n"
        "var b = Bag(Array[Item, 2](fill=item), 0)\nfor j in range(2):\n    b.items[j].pos[0] = Float32(j)\n"
        "    b.items[j].pos[1] = Float32(j) + 2.0\n    b.items[j].w[0] = 1.0\n    b.items[j].w[1] = 3.0\n"
        "var k = Int(iin[i] & 1)\nb.n = iin[i]\nb.items[k].pos[1] = Float32(b.n) * 2.0\nb.items[1 - k].w[k] += 8.0\n"
        "b.items[k].w[1 - k] = b.items[1 - k].pos[0] + 4.0\n"
        "outp[i] = b.items[0].pos[1] + b.items[1].pos[1] * 32.0 + b.items[0].w[1] * 1024.0"
        " + b.items[1].w[0] * 16384.0 + b.items[0].w[0] * 262144.0 + b.items[1].w[1] * 4194304.0",
        "@fieldwise_init\nstruct Item(Copyable, Movable):\n    var pos: $vec2\n    var w: Array[Float32, 2]\n\n\n"
        "@fieldwise_init\nstruct Bag(Copyable, Movable):\n    var items: Array[Item, 2]\n    var n: Int32",
    ),
    "bool-array": Template(
        "var flags = Array[Bool, 4](fill=False)\nfor j in range(4):\n    flags[j] = ((iin[i] >> Int32(j)) & 1) == 1\n"
        "var either = flags[0] or flags[3]\nflags[1] = not flags[1]\nvar r: Float32 = 1.0 if flags[1] else 0.0\n"
        "var s: Float32 = 2.0 if flags[2] else 0.0\nvar t: Float32 = 4.0 if either else 0.0\n"
        "outp[i] = r + s + t + Float32(iin[i]) * 8.0"
    ),
}
