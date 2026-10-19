"""Token-level mutation of well-formed sources, to reach every importer's error paths.

A mutation deletes, duplicates, swaps or replaces tokens, inserts a token from the language's vocabulary, or
truncates the text. Hypothesis draws the positions, so a crash found on a mutant shrinks to fewer and smaller
edits of a smaller base program.
"""

from __future__ import annotations

import re

from hypothesis import strategies as st

_TOKEN = re.compile(r"[A-Za-z_]\w*|\d[\w.]*|\s+|\"(?:\\.|[^\"\\\n])*\"|.", re.S)

# fmt: off
COMMON = (
    "{", "}", "(", ")", "[", "]", "<", ">", ";", ",", ".", ":", "::", "?", "=", "==", "+", "-", "*", "/",
    "%", "&", "|", "^", "~", "!", "<<", ">>", "&&", "||", "++", "--", "->", "#", "##", "@", "\\", "\"", "'",
    "\n", " ", "0", "1", "-1", "0x", "0x7fffffff", "4294967296", "99999999999999999999", "1e", "1e400",
    "1.0", ".5", "1.5f", "0.0h", "1u", "'a'", '"s"', "/*", "*/", "//", "\\\n", "\t", "\x00", "\u00e9",
    "int", "uint", "float", "bool", "void", "struct", "return", "if", "else", "for", "while", "do",
    "switch", "case", "default", "break", "continue", "const", "static", "inline", "template", "typename",
    "true", "false", "main", "a", "o", "i",
)
VOCABULARY = {
    "metal": COMMON + (
        "kernel", "device", "constant", "threadgroup", "thread", "[[", "]]", "buffer(0)", "texture(0)",
        "thread_position_in_grid", "using", "namespace", "metal", "#include", "<metal_stdlib>", "float4",
        "half", "packed_float3", "simd_sum", "atomic_int", "constexpr", "sampler", "texture2d<float>",
    ),
    "directx": COMMON + (
        "[numthreads(1,1,1)]", "SV_DispatchThreadID", "register(u0)", "RWStructuredBuffer<int>", "cbuffer",
        "groupshared", "float4", "min16float", "half", "uint3", "Texture2D", "SamplerState", "[unroll]",
        "WaveActiveSum", "InterlockedAdd", "row_major", "precise", "nointerpolation", "#pragma",
    ),
    "opengl": COMMON + (
        "#version", "450", "310 es", "layout", "local_size_x", "std430", "std140", "binding", "buffer",
        "uniform", "in", "out", "inout", "readonly", "writeonly", "shared", "vec4", "ivec2", "uvec3", "mat4",
        "gl_GlobalInvocationID", "barrier()", "precision", "highp", "#extension", "subroutine", "sampler2D",
    ),
    "vulkan": (
        "\n", " ", "%", "%1", "%2", "%a", "=", "OpCapability", "Shader", "OpLabel", "OpReturn", "OpBranch",
        "OpFunction", "OpFunctionEnd", "OpTypeInt", "32", "0", "1", "OpConstant", "OpLoad", "OpStore",
        "OpIAdd", "OpPhi", "OpSelectionMerge", "OpLoopMerge", "OpSwitch", "None", "OpDecorate", "Binding",
        "OpTypeVector", "OpTypePointer", "Function", "StorageBuffer", "OpVariable", "OpAccessChain",
        "OpExtInst", "%glsl", "OpTypeStruct", "OpTypeRuntimeArray", "4294967296", "-1", "0.5", '"main"',
        "OpEntryPoint", "GLCompute", "OpExecutionMode", "LocalSize", "OpUnreachable", "OpKill", ";",
    ),
    "cgl": COMMON + (
        "shader", "compute", "vertex", "fragment", "@", "numthreads", "gl_GlobalInvocationID", "vec3",
        "ivec2", "uvec4", "mat3", "StructuredBuffer<int>", "RWStructuredBuffer<int>", "cbuffer", "generic",
    ),
}
# fmt: on


def tokens(text: str):
    return _TOKEN.findall(text)


@st.composite
def mutated(draw, base, language: str):
    """``base`` (a strategy of source text) with one to four token mutations."""
    text = draw(base)
    pieces = tokens(text)
    vocabulary = VOCABULARY.get(language, COMMON)
    for _ in range(draw(st.integers(1, 4))):
        if not pieces:
            pieces = [draw(st.sampled_from(vocabulary))]
            continue
        at = draw(st.integers(0, len(pieces) - 1))
        width = draw(st.integers(1, min(8, len(pieces) - at)))
        action = draw(
            st.sampled_from(
                ("delete", "duplicate", "swap", "replace", "insert", "truncate")
            )
        )
        if action == "delete":
            del pieces[at : at + width]
        elif action == "duplicate":
            pieces[at:at] = pieces[at : at + width]
        elif action == "swap":
            other = draw(st.integers(0, len(pieces) - 1))
            pieces[at], pieces[other] = pieces[other], pieces[at]
        elif action == "replace":
            pieces[at] = draw(st.sampled_from(vocabulary))
        elif action == "insert":
            pieces.insert(at, draw(st.sampled_from(vocabulary)))
        else:
            del pieces[at:]
    return "".join(pieces)
