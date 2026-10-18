#!/usr/bin/env python3
"""Deterministic expression kernels with independent integer-rule oracles.

This module generates fresh kernels from a fixed seed: random expression
trees over integer operations in several source languages, evaluated on edge
inputs. Only operations whose source rule is defined for every input are generated (no source undefined behaviour):

  GLSL   int +,-,* (wrap); int-vs-uint comparison (unsigned)
  HLSL   int +,-,* (wrap, reading A); uint <<,>> with any count (mask); uint / and % by possibly-zero (D3D:
         0xFFFFFFFF); int-vs-uint comparison
  MSL    uint <<,>> with any count (mask); int % by non-zero (observational control); int-vs-uint
         comparison; uchar arithmetic stored back to uchar (mod 256)
  CUDA   int-vs-uint comparison; unsigned char arithmetic (mod 256); int % by non-zero (truncated)
  OpenCL uint <<,>> (mask); int % by non-zero (truncated); uchar arithmetic (mod 256); int-vs-uint comparison
Each kernel reads a[i], b[i] (int32) and writes one int32. Oracles are Python models of the rules.
Usage (from run_suite): heldout_cases(n_per_family, seed) -> list[Case]
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from . import cases as C

HERE = Path(__file__).resolve().parent
EDGE = [
    0,
    1,
    -1,
    2,
    3,
    7,
    31,
    32,
    33,
    63,
    255,
    256,
    1000,
    -1000,
    2**30,
    2**31 - 1,
    -(2**31),
    2**31 - 2,
    -(2**31) + 1,
    123456789,
    -987654321,
    65535,
    65536,
]
MASK = 0xFFFFFFFF


def w32(x):  # signed 32-bit wrap
    x &= MASK
    return x - (1 << 32) if x >= 1 << 31 else x


def u32(x):
    return x & MASK


# ---- expression trees ------------------------------------------------------------------------------------------
# A node is (kind, ...) evaluated over the integer values of a and b; `text` renders it per language.
class Gen:
    def __init__(self, rng, lang, family):
        self.rng, self.lang, self.family = rng, lang, family

    def leaf_int(self):
        r = self.rng.random()
        if r < 0.4:
            return ("a",)
        if r < 0.8:
            return ("b",)
        return ("c", self.rng.choice(EDGE))

    def int_expr(self, depth, root=False):
        if depth == 0 or (not root and self.rng.random() < 0.25):
            return self.leaf_int()
        return (
            self.rng.choice(["+", "-", "*"]),
            self.int_expr(depth - 1),
            self.int_expr(depth - 1),
        )


def leaves(n):
    return [n] if n[0] in ("a", "b", "c") else leaves(n[1]) + leaves(n[2])


def ev_int(n, a, b):
    k = n[0]
    if k == "a":
        return a
    if k == "b":
        return b
    if k == "c":
        return w32(n[1])
    x, y = ev_int(n[1], a, b), ev_int(n[2], a, b)
    return w32({"+": x + y, "-": x - y, "*": x * y}[k])


def tx_int(n, lang):
    k = n[0]
    if k in ("a", "b"):
        return {
            "glsl": f"{k}v",
            "hlsl": f"{k}v",
            "metal": f"{k}v",
            "cuda": f"{k}v",
            "opencl": f"{k}v",
        }[lang]
    if k == "c":
        v = w32(n[1])
        if v == -(2**31):
            return "(-2147483647 - 1)"
        return f"({v})" if v < 0 else str(v)
    return f"({tx_int(n[1], lang)} {k} {tx_int(n[2], lang)})"


# ---- kernel families -------------------------------------------------------------------------------------------
def fam_wrap(rng, lang):
    g = Gen(rng, lang, "wrap")
    e = g.int_expr(3, root=True)
    while all(n[0] == "c" for n in leaves(e)):  # at least one input operand
        e = g.int_expr(3, root=True)
    body = f"int r = {tx_int(e, lang)};"
    return body, (lambda a, b: ev_int(e, a, b))


def fam_cmp(rng, lang):
    k = rng.choice(EDGE)
    u = u32(k)
    op = rng.choice(["<", "<=", ">", ">="])
    body = f"uint uk = {u}u;\n    int r = (av {op} uk) ? 1 : 0;"

    def f(a, b):
        x = u32(a)
        return int({"<": x < u, "<=": x <= u, ">": x > u, ">=": x >= u}[op])

    return body, f


def fam_shift(rng, lang):
    ops = [rng.choice(["<<", ">>"]) for _ in range(2)]
    body = f"uint x = uint(av); uint s = uint(bv);\n    uint t = (x {ops[0]} s) ^ (x {ops[1]} (s + 1u));\n    int r = int(t);"

    def f(a, b):
        x, s = u32(a), u32(b)

        def sh(o, v, c):
            c &= 31
            return u32(v << c) if o == "<<" else (v >> c)

        return w32(sh(ops[0], x, s) ^ sh(ops[1], x, u32(s + 1)))

    return body, f


def fam_udiv(rng, lang):
    op = rng.choice(["/", "%"])
    body = f"uint x = uint(av); uint y = uint(bv) & 7u;\n    int r = int(x {op} y);"

    def f(a, b):
        x, y = u32(a), u32(b) & 7
        if y == 0:
            return w32(0xFFFFFFFF)
        return w32(x // y if op == "/" else x % y)

    return body, f


def fam_srem(rng, lang):
    body = "int y = (bv & 15) - 7;\n    if (y == 0) { y = 3; }\n    if (av == (-2147483647 - 1) && y == -1) { y = 5; }\n    int r = av % y;"

    def f(a, b):
        y = (b & 15) - 7
        if y == 0:
            y = 3
        if a == -(2**31) and y == -1:
            y = 5
        q = abs(a) // abs(y) * (1 if (a >= 0) == (y > 0) else -1)
        return w32(a - y * q)

    return body, f


def fam_narrow(rng, lang):
    steps = [rng.choice(["+", "-", "*"]) for _ in range(3)]
    ks = [rng.choice([1, 3, 7, 100, 200, 255, 128]) for _ in range(3)]
    t = {"metal": "uchar", "cuda": "unsigned char", "opencl": "uchar"}[lang]
    lines = [f"{t} v = ({t})av;"]
    for s, k in zip(steps, ks):
        lines.append(f"v = v {s} ({t}){k};")
    lines.append("int r = (int)v;")
    body = "\n    ".join(lines)

    def f(a, b):
        v = a & 255
        for s, k in zip(steps, ks):
            v = {"+": v + k, "-": v - k, "*": v * k}[s] & 255
        return v

    return body, f


FAMILIES = {
    "glsl": [fam_wrap, fam_cmp],
    "hlsl": [fam_wrap, fam_shift, fam_udiv, fam_cmp],
    "metal": [fam_shift, fam_srem, fam_cmp, fam_narrow],
    "cuda": [fam_cmp, fam_narrow, fam_srem],
    "opencl": [fam_shift, fam_srem, fam_narrow, fam_cmp],
}


def wrap_source(lang, body):
    if lang == "glsl":
        return (
            "#version 450\nlayout(local_size_x = 64) in;\n"
            "layout(std430, binding = 0) buffer Inp { int ab[]; };\n"
            "layout(std430, binding = 1) buffer Outp { int o[]; };\n"
            "void main() {\n    uint i = gl_GlobalInvocationID.x;\n    int av = ab[2u * i];\n    int bv = ab[2u * i + 1u];\n"
            f"    {body}\n    o[i] = r;\n}}\n"
        )
    if lang == "hlsl":
        return (
            "StructuredBuffer<int> ab : register(t0);\nRWStructuredBuffer<int> o : register(u1);\n"
            "[numthreads(64, 1, 1)]\nvoid main(uint3 id : SV_DispatchThreadID) {\n"
            "    int av = ab[2 * id.x];\n    int bv = ab[2 * id.x + 1];\n"
            f"    {body}\n    o[id.x] = r;\n}}\n"
        )
    if lang == "metal":
        return (
            "#include <metal_stdlib>\nusing namespace metal;\n"
            "kernel void hazard(device const int *inp [[buffer(0)]], device int *outp [[buffer(1)]],\n"
            "                   uint i [[thread_position_in_grid]]) {\n"
            "    int av = inp[2 * i];\n    int bv = inp[2 * i + 1];\n"
            f"    {body}\n    outp[i] = r;\n}}\n"
        )
    if lang == "cuda":
        return (
            "__global__ void hazard(const int *inp, int *outp) {\n"
            "    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
            "    int av = inp[2 * i];\n    int bv = inp[2 * i + 1];\n"
            f"    {body.replace('uint(', '(unsigned int)(').replace('uint ', 'unsigned int ').replace('int(t)', '(int)(t)')}\n"
            "    outp[i] = r;\n}\n"
        )
    if lang == "opencl":
        return (
            "__kernel void hazard(__global const int *inp, __global int *outp) {\n"
            "    size_t i = get_global_id(0);\n    int av = inp[2 * i];\n    int bv = inp[2 * i + 1];\n"
            f"    {body.replace('uint(', '(uint)(').replace('int(t)', '(int)(t)')}\n"
            "    outp[i] = r;\n}\n"
        )
    raise ValueError(lang)


EXT = {
    "glsl": "src.comp",
    "hlsl": "src.hlsl",
    "metal": "src.metal",
    "cuda": "src.cu",
    "opencl": "src.cl",
}


def heldout_cases(directory, n_per_family=6, seed=20261017):
    """Write generated fixtures into a caller-owned scratch directory."""
    rng = random.Random(seed)
    out = []
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    inputs = np.array(
        [v for _ in range(4) for v in rng.sample(EDGE, 16)], dtype=np.int64
    )
    pairs = np.array(
        [w32(int(x)) for x in np.concatenate([inputs, inputs[::-1]])], dtype=np.int64
    )[:128]
    pairs = pairs.astype(np.int32)
    for lang, fams in FAMILIES.items():
        for fam in fams:
            for k in range(n_per_family):
                body, f = fam(rng, lang)
                cid = f"x_{lang}_{fam.__name__[4:]}_{k}"
                d = directory / cid
                d.mkdir(exist_ok=True)
                (d / EXT[lang]).write_text(wrap_source(lang, body))
                oracle = lambda inp, f=f: np.array(
                    [f(int(a), int(b)) for a, b in inp.reshape(-1, 2)], dtype=np.int64
                ).astype(np.int32)
                case = C.Case(
                    cid,
                    "heldout:" + fam.__name__[4:],
                    f"random {fam.__name__[4:]}",
                    "carried rule",
                    pairs,
                    np.int32,
                    oracle,
                    readings=(
                        {"A": True, "B": fam.__name__ not in ("fam_wrap", "fam_udiv")}
                        if lang == "hlsl"
                        else {"spec": not (lang == "metal" and fam == fam_srem)}
                    ),
                    lang=lang,
                    src=d / EXT[lang],
                )
                out.append(case)
    return out
