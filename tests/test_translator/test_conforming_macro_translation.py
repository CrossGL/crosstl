"""Imported sources whose meaning depends on conforming macro replacement translate to the compiler's meaning.

An argument is fully macro-replaced before substitution, a pasted operand is not, and a function-like macro name
produced at the end of a replacement takes the arguments that follow the invocation. Before the front ends used
the shared expander, ``F(F(1))`` translated to a call of an undeclared ``F``, ``CAT(v, N)`` to ``vN`` and
``G(1)`` (``#define G F``) to ``F(1)``.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl

HLSL = """
#define F(x) ((x) + 1)
#define CAT_I(a, b) a##b
#define CAT(a, b) CAT_I(a, b)
#define N 2
#define G F
#define E(first, ...) combine(first, ##__VA_ARGS__)
int v2() { return 7; }
int combine(int a) { return a; }
int combine(int a, int b) { return a * b; }
RWStructuredBuffer<int> o : register(u0);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    o[0] = F(F(1));
    o[1] = CAT(v, N)();
    o[2] = G(5);
    o[3] = E(3) + E(3, 4);
}
"""

METAL = """
#include <metal_stdlib>
using namespace metal;
#define F(x) ((x) + 1)
#define CAT_I(a, b) a##b
#define CAT(a, b) CAT_I(a, b)
#define N 2
#define G F
#define OPT(a, ...) (a __VA_OPT__(* __VA_ARGS__))
int v2() { return 7; }
kernel void k(device int* o [[buffer(0)]]) {
    o[0] = F(F(1));
    o[1] = CAT(v, N)();
    o[2] = G(5);
    o[3] = OPT(3) + OPT(3, 4);
}
"""


def translate(tmp_path, name, text, backend):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def statements(code):
    return {
        line.strip().split(" = ", 1)[0]: line.strip()
        for line in code.splitlines()
        if line.strip().startswith("o[")
    }


def validate_glsl(tmp_path, glsl):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")
    artifact = tmp_path / "shader.comp"
    artifact.write_text(glsl, encoding="utf-8")
    result = subprocess.run(
        [glslang, str(artifact)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_hlsl_macros_translate_to_their_replacement(tmp_path):
    glsl = translate(tmp_path, "macros.hlsl", HLSL, "opengl")
    found = statements(glsl)
    assert found["o[0]"] == "o[0] = ((1 + 1) + 1);"
    assert found["o[1]"] == "o[1] = v2();"
    assert found["o[2]"] == "o[2] = (5 + 1);"
    assert found["o[3]"] == "o[3] = (combine(3) + combine(3, 4));"
    validate_glsl(tmp_path, glsl)


def test_metal_macros_translate_to_their_replacement(tmp_path):
    glsl = translate(tmp_path, "macros.metal", METAL, "opengl")
    found = statements(glsl)
    assert found["o[0]"] == "o[0] = ((1 + 1) + 1);"
    assert found["o[1]"] == "o[1] = v2();"
    assert found["o[2]"] == "o[2] = (5 + 1);"
    assert found["o[3]"] == "o[3] = (3 + (3 * 4));"
    validate_glsl(tmp_path, glsl)
