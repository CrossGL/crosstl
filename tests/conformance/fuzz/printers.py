"""Print a :class:`~tests.conformance.fuzz.program.Program` as source in each C-family shading language.

Every binary, unary and conditional expression is fully parenthesized, so no printer depends on the target's
precedence table. Names are unique per program, so no printer depends on the target's scoping rules either.
"""

from __future__ import annotations

import math
import struct

from .program import (
    Assign,
    Break,
    Continue,
    Declare,
    For,
    If,
    Literal,
    Return,
    Switch,
    Var,
)


def float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


class CPrinter:
    """The shared C-family printer. Subclasses set the spellings that differ."""

    language = "c"
    float_suffix = "f"
    uint_suffix = "u"
    vector_names = {"int": "int", "uint": "uint", "float": "float", "bool": "bool"}
    builtin_names: dict = {}

    def type(self, type_) -> str:
        base = self.vector_names[type_.kind]
        return base if type_.size == 1 else f"{base}{type_.size}"

    def scalar_type(self, kind) -> str:
        return {"int": "int", "uint": "uint", "float": "float", "bool": "bool"}[kind]

    # -- expressions ---------------------------------------------------------------------------------------------
    def literal(self, node: Literal) -> str:
        kind, value = node.type.kind, node.value
        if kind == "bool":
            return "true" if value else "false"
        if kind == "int":
            if value == -2147483648:
                return "(-2147483647 - 1)"
            return f"({value})" if value < 0 else str(value)
        if kind == "uint":
            return f"{value}{self.uint_suffix}"
        value = float32(value)
        if math.copysign(1.0, value) < 0:
            return f"(-{self.float_text(-value)})"
        return self.float_text(value)

    def float_text(self, value: float) -> str:
        text = repr(value)
        if "e" not in text and "." not in text:
            text += ".0"
        return text + self.float_suffix

    def expr(self, node) -> str:
        method = getattr(self, "expr_" + type(node).__name__.lower())
        return method(node)

    def expr_literal(self, node):
        return self.literal(node)

    def expr_var(self, node):
        return node.name

    def expr_load(self, node):
        return "a[i]"

    def expr_unary(self, node):
        return f"({node.op}{self.expr(node.operand)})"

    def expr_binary(self, node):
        return f"({self.expr(node.left)} {node.op} {self.expr(node.right)})"

    def expr_select(self, node):
        return (
            f"({self.expr(node.condition)} ? {self.expr(node.then)} : "
            f"{self.expr(node.otherwise)})"
        )

    def expr_swizzle(self, node):
        return f"{self.atom(node.base)}.{node.components}"

    def atom(self, node) -> str:
        text = self.expr(node)
        return text if isinstance(node, Var) else f"({text})"

    def expr_construct(self, node):
        args = ", ".join(self.expr(arg) for arg in node.args)
        return f"{self.type(node.type)}({args})"

    def expr_convert(self, node):
        return f"{self.type(node.type)}({self.expr(node.operand)})"

    def expr_builtin(self, node):
        name = self.builtin_names.get(node.name, node.name)
        return f"{name}({', '.join(self.expr(arg) for arg in node.args)})"

    def expr_call(self, node):
        return f"{node.function}({', '.join(self.expr(arg) for arg in node.args)})"

    # -- statements ----------------------------------------------------------------------------------------------
    def block(self, statements, indent) -> list:
        lines = []
        for statement in statements:
            lines += self.statement(statement, indent)
        return lines

    def statement(self, node, indent) -> list:
        pad = "    " * indent
        if isinstance(node, Declare):
            return [
                f"{pad}{self.type(node.type)} {node.name} = {self.expr(node.value)};"
            ]
        if isinstance(node, Assign):
            op = f"{node.op}=" if node.op else "="
            return [f"{pad}{node.name} {op} {self.expr(node.value)};"]
        if isinstance(node, If):
            lines = [f"{pad}if ({self.expr(node.condition)}) {{"]
            lines += self.block(node.then, indent + 1)
            if node.otherwise is not None:
                lines.append(f"{pad}}} else {{")
                lines += self.block(node.otherwise, indent + 1)
            return lines + [f"{pad}}}"]
        if isinstance(node, For):
            lines = [
                f"{pad}for ({self.scalar_type('int')} {node.name} = 0; {node.name} < {node.count}; "
                f"{node.name}++) {{"
            ]
            return lines + self.block(node.body, indent + 1) + [f"{pad}}}"]
        if isinstance(node, Switch):
            lines = [f"{pad}switch ({self.expr(node.selector)}) {{"]
            for label, body in node.cases:
                lines.append(
                    f"{pad}case {self.literal(Literal(node.selector.type, label))}: {{"
                )
                lines += self.block(body, indent + 1) + [f"{pad}    break;", f"{pad}}}"]
            lines.append(f"{pad}default: {{")
            lines += self.block(node.default, indent + 1) + [
                f"{pad}    break;",
                f"{pad}}}",
            ]
            return lines + [f"{pad}}}"]
        if isinstance(node, Break):
            return [f"{pad}break;"]
        if isinstance(node, Continue):
            return [f"{pad}continue;"]
        if isinstance(node, Return):
            return [f"{pad}return {self.expr(node.value)};"]
        raise TypeError(f"unknown statement {node!r}")

    def function(self, node) -> list:
        params = ", ".join(f"{self.type(t)} {name}" for name, t in node.params)
        lines = [f"{self.type(node.result)} {node.name}({params}) {{"]
        return lines + self.block(node.body, 1) + ["}", ""]

    def kernel_body(self, program) -> list:
        lines = self.block(program.body, 1)
        return lines + [f"    o[i] = {self.expr(program.output)};"]

    def print(self, program) -> str:
        lines = list(self.prelude())
        for function in program.functions:
            lines += self.function(function)
        lines += self.kernel(program)
        return "\n".join(lines) + "\n"


class MetalPrinter(CPrinter):
    language = "metal"

    def prelude(self):
        return ["#include <metal_stdlib>", "using namespace metal;", ""]

    def kernel(self, program):
        return [
            "kernel void fuzz_main(device const int* a [[buffer(0)]], device int* o [[buffer(1)]],",
            "                      uint i [[thread_position_in_grid]]) {",
            *self.kernel_body(program),
            "}",
        ]


class HLSLPrinter(CPrinter):
    language = "directx"
    builtin_names = {"mix": "lerp", "fract": "frac"}

    def expr_convert(self, node):
        return f"(({self.type(node.type)})({self.expr(node.operand)}))"

    def expr_construct(self, node):
        if (
            node.type.is_vector and len(node.args) == 1
        ):  # HLSL splats by cast, not by constructor
            return f"(({self.type(node.type)})({self.expr(node.args[0])}))"
        return super().expr_construct(node)

    def prelude(self):
        return [
            "StructuredBuffer<int> a : register(t0);",
            "RWStructuredBuffer<int> o : register(u1);",
            "",
        ]

    def kernel(self, program):
        return [
            "[numthreads(64, 1, 1)]",
            "void main(uint3 tid : SV_DispatchThreadID) {",
            "    uint i = tid.x;",
            *self.kernel_body(program),
            "}",
        ]


class SlangPrinter(HLSLPrinter):
    language = "slang"


class GLSLPrinter(CPrinter):
    language = "opengl"
    float_suffix = ""
    vector_names = {"int": "ivec", "uint": "uvec", "float": "vec", "bool": "bvec"}

    def __init__(self, version: str = "450"):
        self.version = version

    def type(self, type_):
        if type_.size == 1:
            return type_.kind
        return f"{self.vector_names[type_.kind]}{type_.size}"

    def prelude(self):
        lines = [f"#version {self.version}"]
        if self.version.endswith(" es"):
            lines += ["precision highp float;", "precision highp int;"]
        return lines + [
            "layout(local_size_x = 64) in;",
            "layout(std430, binding = 0) readonly buffer Inp { int a[]; };",
            "layout(std430, binding = 1) writeonly buffer Outp { int o[]; };",
            "",
        ]

    def kernel(self, program):
        return [
            "void main() {",
            "    uint i = gl_GlobalInvocationID.x;",
            *self.kernel_body(program),
            "}",
        ]


class CrossGLPrinter(GLSLPrinter):
    language = "cgl"

    def prelude(self):
        return [
            "shader main {",
            "    StructuredBuffer<int> a;",
            "    RWStructuredBuffer<int> o;",
            "",
        ]

    def kernel(self, program):
        return [
            "    compute {",
            "    @ numthreads(64, 1, 1)",
            "    void main(uvec3 gid @ gl_GlobalInvocationID) {",
            "    uint i = gid.x;",
            *self.kernel_body(program),
            "    }",
            "    }",
            "}",
        ]


class CUDAPrinter(CPrinter):
    """Scalar programs only (``Config(vectors=False)``); CUDA vector types have no operators."""

    language = "cuda"
    builtin_names = {"floor": "floorf", "ceil": "ceilf", "sqrt": "sqrtf", "sin": "sinf"}
    builtin_names.update({"cos": "cosf", "exp": "expf"})

    scalars = {"int": "int", "uint": "unsigned int", "float": "float", "bool": "bool"}

    def type(self, type_):
        if type_.is_vector:
            raise ValueError("the CUDA printer takes scalar programs")
        return self.scalars[type_.kind]

    def scalar_type(self, kind):
        return self.scalars[kind]

    def expr_convert(self, node):
        return f"(({self.type(node.type)})({self.expr(node.operand)}))"

    def expr_builtin(self, node):
        args = [self.expr(arg) for arg in node.args]
        if node.name == "abs" and node.type.kind == "float":
            return f"fabsf({args[0]})"
        if node.name == "fract":
            return f"({args[0]} - floorf({args[0]}))"
        if node.name == "mix":
            return f"({args[0]} + ({args[1]} - {args[0]}) * {args[2]})"
        if node.name == "clamp":
            return f"min(max({args[0]}, {args[1]}), {args[2]})"
        if node.name in ("min", "max") and node.type.kind == "float":
            return f"f{node.name}f({args[0]}, {args[1]})"
        return f"{self.builtin_names.get(node.name, node.name)}({', '.join(args)})"

    def function(self, node):
        lines = super().function(node)
        return ["__device__ " + lines[0], *lines[1:]]

    def prelude(self):
        return []

    def kernel(self, program):
        return [
            "__global__ void fuzz_main(const int* a, int* o) {",
            "    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;",
            *self.kernel_body(program),
            "}",
        ]


class HIPPrinter(CUDAPrinter):
    language = "hip"

    def prelude(self):
        return ["#include <hip/hip_runtime.h>", ""]


class OpenCLPrinter(CPrinter):
    """Scalar programs only: OpenCL's ``abs`` of ``int`` returns ``uint``, and vector literals differ."""

    language = "opencl"

    def type(self, type_):
        if type_.is_vector:
            raise ValueError("the OpenCL printer takes scalar programs")
        return type_.kind

    def expr_convert(self, node):
        return f"(({self.type(node.type)})({self.expr(node.operand)}))"

    def expr_builtin(self, node):
        args = [self.expr(arg) for arg in node.args]
        if node.name == "abs" and node.type.kind == "int":
            return f"((int)abs({args[0]}))"
        if node.name == "abs":
            return f"fabs({args[0]})"
        if node.name in ("min", "max") and node.type.kind == "float":
            return f"f{node.name}({args[0]}, {args[1]})"
        if node.name == "fract":
            return f"({args[0]} - floor({args[0]}))"
        return f"{node.name}({', '.join(args)})"

    def prelude(self):
        return []

    def kernel(self, program):
        return [
            "__kernel void fuzz_main(__global const int* a, __global int* o) {",
            "    uint i = get_global_id(0);",
            *self.kernel_body(program),
            "}",
        ]


PRINTERS = {
    "metal": MetalPrinter,
    "directx": HLSLPrinter,
    "slang": SlangPrinter,
    "opengl": GLSLPrinter,
    "cgl": CrossGLPrinter,
    "cuda": CUDAPrinter,
    "hip": HIPPrinter,
    "opencl": OpenCLPrinter,
}
