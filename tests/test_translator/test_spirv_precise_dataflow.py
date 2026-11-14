"""SPIR-V NoContraction covers every operation whose value reaches a precise variable.

GLSL 4.60 section 4.7.1 and HLSL ``precise`` apply to all operations contributing to a precise value, including
those computed into other variables first (crosstl/translator/codegen/precise_dataflow.py).
"""

import re
import shutil
import subprocess

import pytest

from crosstl import translate
from crosstl.translator.codegen.precise_dataflow import (
    declaration_is_precise,
    precise_contributors,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

GLSL_HEADER = """#version 450
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Inp { float abc[]; };
layout(std430, binding = 1) buffer Outp { float o[]; };
"""


def spirv(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    code = translate(str(path), backend="vulkan", format_output=False)
    validator = shutil.which("spirv-as") and shutil.which("spirv-val")
    if validator:
        assembled = tmp_path / "module.spv"
        subprocess.run(
            ["spirv-as", "--target-env", "vulkan1.1", "-o", str(assembled), "-"],
            input=code.encode(),
            check=True,
        )
        subprocess.run(
            ["spirv-val", "--target-env", "vulkan1.1", str(assembled)], check=True
        )
    return code


def decorated_operations(code):
    """Map each float arithmetic result id to whether it is decorated NoContraction, in emission order."""
    decorated = set(re.findall(r"OpDecorate %(\w+) NoContraction", code))
    return [
        (opcode, result in decorated)
        for result, opcode in re.findall(r"%(\w+) = (OpF(?:Mul|Add|Sub|Div)) ", code)
    ]


def test_hlsl_precise_local_qualifier_reaches_spirv(tmp_path):
    code = spirv(
        tmp_path,
        "precise.hlsl",
        """StructuredBuffer<float> abc : register(t0);
RWStructuredBuffer<float> o : register(u1);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    float a = abc[3 * id.x];
    float b = abc[3 * id.x + 1];
    float c = abc[3 * id.x + 2];
    precise float r = a * b - c;
    o[id.x] = r;
}
""",
    )
    assert decorated_operations(code) == [("OpFMul", True), ("OpFSub", True)]


def test_values_computed_into_other_locals_first_are_precise(tmp_path):
    code = spirv(
        tmp_path,
        "flow.comp",
        GLSL_HEADER + """void main() {
    uint i = gl_GlobalInvocationID.x;
    float a = abc[3u * i];
    float b = abc[3u * i + 1u];
    float c = abc[3u * i + 2u];
    float t = a * b;
    float u = a * c;
    precise float r = t - c;
    o[i] = r + u * b;
}
""",
    )
    # glslang decorates the same two operations: a * b reaches r through t, a * c and u * b do not reach r.
    assert decorated_operations(code) == [
        ("OpFMul", True),
        ("OpFMul", False),
        ("OpFSub", True),
        ("OpFMul", False),
        ("OpFAdd", False),
    ]


def test_assignments_and_compound_stores_propagate_backwards(tmp_path):
    code = spirv(
        tmp_path,
        "compound.comp",
        GLSL_HEADER + """void main() {
    uint i = gl_GlobalInvocationID.x;
    float a = abc[3u * i];
    float b = abc[3u * i + 1u];
    float t;
    t = a * b;
    t += b * b;
    precise float r;
    r = t * 0.5;
    o[i] = r;
}
""",
    )
    assert decorated_operations(code) == [
        ("OpFMul", True),
        ("OpFMul", True),
        ("OpFAdd", True),
        ("OpFMul", True),
    ]


def parse_body(source):
    lexer = Lexer(source)
    tokens = lexer.tokenize() or lexer.tokens
    program = Parser(tokens).parse()
    return program.functions[0].body


def test_contributors_follow_definitions_to_a_fixpoint():
    body = parse_body("""shader main {
    float f(float a, float b, float c) {
        float x = a * b;
        float y = x + c;
        float z = c * c;
        float r @precise = y;
        return z;
    }
}
""")
    assert precise_contributors(body) == frozenset({"r", "y", "x", "a", "b", "c"})


def test_contributors_are_empty_without_a_precise_variable():
    body = parse_body("""shader main {
    float f(float a) {
        float x = a * a;
        return x;
    }
}
""")
    assert precise_contributors(body) == frozenset()


def test_seeds_name_precise_variables_declared_outside_the_body():
    body = parse_body("""shader main {
    void f(float a) {
        float x = a * a;
        g = x;
    }
}
""")
    assert precise_contributors(body, seeds=["g"]) == frozenset({"g", "x", "a"})


@pytest.mark.parametrize(
    "declaration, expected",
    (
        ("precise float r = 1.0;", True),
        ("float r @precise = 1.0;", True),
        ("float r = 1.0;", False),
    ),
)
def test_both_spellings_declare_a_precise_variable(declaration, expected):
    body = parse_body(
        "shader main {\n    void f() {\n        " + declaration + "\n    }\n}\n"
    )
    statements = getattr(body, "statements", body)
    assert declaration_is_precise(statements[0]) is expected
