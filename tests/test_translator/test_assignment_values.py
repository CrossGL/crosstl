"""Assignments used as values keep their value and their grouping.

A CrossGL assignment is an expression that yields the stored value, so ``(e = 4) + 1`` stores four and yields five.
The parser drops source parentheses, and assignment binds more loosely than every other C-family operator, so an
emitter must parenthesize an assignment that appears as an operand. Without that, ``(e = 4) + 1`` becomes
``e = 4 + 1`` and stores five. Rust assignments yield ``()``, so Rust reads the target back in a block.
"""

import re
import shutil
import struct
import subprocess

import pytest

import crosstl.translator
from crosstl.translator.ast import (
    ArrayAccessNode,
    BinaryOpNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    UnaryOpNode,
)
from crosstl.translator.codegen.assignment_values import (
    assignment_value_expression,
    is_fully_parenthesized,
    is_reevaluable_place,
)
from crosstl.translator.codegen.cuda_codegen import CudaCodeGen
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.hip_codegen import HipCodeGen
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.codegen.rust_codegen import RustCodeGen
from crosstl.translator.codegen.rust_refusals import RustLoweringRefused
from crosstl.translator.codegen.slang_codegen import SlangCodeGen

PROBE = """
shader AssignmentValueProbe {
    compute {
        layout(std430, binding = 0) buffer Results { int data[]; } results;
        layout(local_size_x = 1) in;
        void main() {
            int e = 1;
            int g = 2;
            int f = (e = 4) + 1;
            int h = (g += 3) * 2;
            int p = 7;
            int q = -(p -= 2);
            int r = 3;
            int s = (r == 3) ? (r = 9) : 0;
            int t = 2;
            int u = (t *= 3) > 5 ? 1 : 0;
            int n = 4;
            int count = 0;
            while ((n -= 1) > 0) {
                count += 1;
            }
            int w = 0;
            int z = (w = 3) + 2 * 4;
            results.data[0] = e;
            results.data[1] = f;
            results.data[2] = g;
            results.data[3] = h;
            results.data[4] = p;
            results.data[5] = q;
            results.data[6] = r;
            results.data[7] = s;
            results.data[8] = t;
            results.data[9] = u;
            results.data[10] = n;
            results.data[11] = count;
            results.data[12] = w;
            results.data[13] = z;
        }
    }
}
"""

# CrossGL: each assignment stores its own right operand and yields the stored value.
EXPECTED = (4, 5, 5, 10, 5, -5, 9, 9, 6, 1, 0, 3, 3, 11)

STATEMENTS = """
shader AssignmentStatementProbe {
    compute {
        layout(std430, binding = 0) buffer Results { int data[]; } results;
        layout(local_size_x = 1) in;
        void main() {
            int total = 0;
            for (int i = 0; i < 8; i += 2) {
                total += i;
            }
            total = total * 3;
            results.data[0] = total;
        }
    }
}
"""

C_FAMILY = {
    "glsl": GLSLCodeGen,
    "hlsl": HLSLCodeGen,
    "metal": MetalCodeGen,
    "slang": SlangCodeGen,
    "cuda": CudaCodeGen,
    "hip": HipCodeGen,
}

VALUE_FORMS = (
    r"\(e = 4\) \+ 1",
    r"\(g \+= 3\) \* 2",
    r"-\s*\(p -= 2\)",
    r"\? \(r = 9\) :",
    r"\(t \*= 3\) > 5",
    r"\(n -= 1\) > 0",
    r"\(w = 3\) \+ ",
)


def _generate(source, generator):
    return generator().generate(crosstl.translator.parse(source))


@pytest.mark.parametrize(
    "code, expected",
    [
        ("(a = b)", True),
        ("((a) = (b))", True),
        ("(a) = (b)", False),
        ("a = (b)", False),
        ('(f(")"))', True),
        ("()", True),
        ("(", False),
    ],
)
def test_full_parenthesization_recognition(code, expected):
    assert is_fully_parenthesized(code) is expected


def test_assignment_value_expression_parenthesizes_only_values():
    assert assignment_value_expression("e = 4") == "(e = 4)"
    assert assignment_value_expression("(e = 4)") == "(e = 4)"
    assert assignment_value_expression("e = 4", discarded=True) == "e = 4"


@pytest.mark.parametrize(
    "node, expected",
    [
        (IdentifierNode("e"), True),
        (MemberAccessNode(IdentifierNode("p"), "x"), True),
        (ArrayAccessNode(IdentifierNode("a"), LiteralNode(1, "int")), True),
        (
            ArrayAccessNode(
                IdentifierNode("a"),
                BinaryOpNode(IdentifierNode("i"), "+", LiteralNode(1, "int")),
            ),
            True,
        ),
        (
            ArrayAccessNode(
                IdentifierNode("a"),
                UnaryOpNode("++", IdentifierNode("i"), is_postfix=True),
            ),
            False,
        ),
        (
            ArrayAccessNode(IdentifierNode("a"), FunctionCallNode("next", [])),
            False,
        ),
    ],
)
def test_reevaluable_place_recognition(node, expected):
    assert is_reevaluable_place(node) is expected


@pytest.mark.parametrize("backend", sorted(C_FAMILY))
def test_assignment_operands_are_parenthesized(backend):
    code = _generate(PROBE, C_FAMILY[backend])
    for form in VALUE_FORMS:
        assert re.search(form, code), (backend, form)
    # The unparenthesized forms assign the wrong value or do not compile.
    assert not re.search(r"(?<!\()e = 4 \+ 1", code)
    assert not re.search(r"(?<!\()g \+= 3 \* 2", code)


@pytest.mark.parametrize("backend", sorted(C_FAMILY))
def test_statement_assignments_gain_no_parentheses(backend):
    code = _generate(STATEMENTS, C_FAMILY[backend])
    assert re.search(r"; i \+= 2\)", code), backend
    assert re.search(r"^\s*total = \(?total \* 3\)?;", code, re.MULTILINE), backend
    assert "(i += 2)" not in code
    assert "(total = " not in code


RUST_PROBE = """
shader RustAssignmentValueProbe {
    struct Pair { int x; int y; }
    int calc(int a) {
        int e = a;
        int f = (e = 4) + 1;
        int g = 2;
        int h = (g += 3) * 2;
        int values[3];
        values[1] = 0;
        int k = (values[1] = 6) - 1;
        Pair pair;
        pair.x = 0;
        int m = (pair.x += 2) * 3;
        for (int i = 0; i < 3; i += 1) {
            e += 1;
        }
        return e + f + g + h + k + m;
    }
    compute {
        layout(local_size_x = 1) in;
        void main() {
            int v = calc(1);
        }
    }
}
"""

# e = 7, f = 5, g = 5, h = 10, k = 5, m = 6.
RUST_EXPECTED = 38


def _compact(code):
    return re.sub(r"\s+", " ", code)


def test_rust_reads_the_assigned_target_back():
    code = _compact(_generate(RUST_PROBE, RustCodeGen))
    assert "({ e = 4; e } + 1)" in code
    assert "({ g += 3; g } * 2)" in code
    assert "({ values[1] = 6; values[1] } - 1)" in code
    assert "({ pair.x += 2; pair.x } * 3)" in code
    assert " i += 1; }" in code


def test_rust_refuses_assignment_values_with_effectful_targets():
    source = """
    shader RustEffectfulAssignmentValue {
        int calc(int a) {
            int values[3];
            int i = 0;
            int k = (values[i++] = a) + 1;
            return k + i;
        }
        compute {
            layout(local_size_x = 1) in;
            void main() {
                int v = calc(1);
            }
        }
    }
    """
    with pytest.raises(RustLoweringRefused, match="assignment used as a value"):
        _generate(source, RustCodeGen)


def test_rust_assignment_values_match_crossgl(tmp_path):
    rustc = shutil.which("rustc")
    if rustc is None:
        pytest.skip("rustc is not installed")
    code = _generate(RUST_PROBE, RustCodeGen)
    start = code.index("#[derive")
    end = code.index("// Compute Shader")
    program = tmp_path / "probe.rs"
    program.write_text(
        code[start:end] + '\nfn main() { println!("{}", calc(1)); }\n',
        encoding="utf-8",
    )
    binary = tmp_path / "probe"
    compiled = subprocess.run(
        [rustc, "--edition", "2021", "-A", "warnings", str(program), "-o", str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
    ran = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
    assert int(ran.stdout) == RUST_EXPECTED


def _tool(name):
    path = shutil.which(name)
    if path is None:
        pytest.skip(f"{name} is not installed")
    return path


def _run(command):
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed


def _run_metal_kernel(metal_source, entry, result_count):
    Metal = pytest.importorskip("Metal")
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")
    library, error = device.newLibraryWithSource_options_error_(
        metal_source, Metal.MTLCompileOptions.new(), None
    )
    assert library is not None, error
    function = library.newFunctionWithName_(entry)
    assert function is not None, list(library.functionNames())
    pipeline, error = device.newComputePipelineStateWithFunction_error_(function, None)
    assert pipeline is not None, error
    result_size = 4 * result_count
    results = device.newBufferWithLength_options_(
        result_size, Metal.MTLResourceStorageModeShared
    )
    command = device.newCommandQueue().commandBuffer()
    encoder = command.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(results, 0, 0)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1), Metal.MTLSizeMake(1, 1, 1)
    )
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    assert command.error() is None, command.error()
    data = bytes(results.contents().as_buffer(result_size))
    return struct.unpack(f"<{result_count}i", data)


def _spirv_to_msl(spirv):
    spirv_cross = _tool("spirv-cross")
    return _run([spirv_cross, "--msl", "--msl-version", "20100", str(spirv)]).stdout


def test_metal_assignment_values_match_crossgl():
    code = _generate(PROBE, MetalCodeGen)
    entry = re.search(r"kernel void (\w+)", code).group(1)
    assert _run_metal_kernel(code, entry, len(EXPECTED)) == EXPECTED


def test_glsl_assignment_values_match_crossgl(tmp_path):
    glslang = _tool("glslangValidator")
    shader = tmp_path / "probe.comp"
    shader.write_text(_generate(PROBE, GLSLCodeGen), encoding="utf-8")
    spirv = tmp_path / "probe.spv"
    _run([glslang, "-V", "-S", "comp", str(shader), "-o", str(spirv)])
    msl = _spirv_to_msl(spirv)
    assert _run_metal_kernel(msl, "main0", len(EXPECTED)) == EXPECTED


def test_hlsl_assignment_values_match_crossgl(tmp_path):
    dxc = _tool("dxc")
    code = _generate(PROBE, HLSLCodeGen)
    entry = re.search(r"\]\s*void (\w+)\(", code).group(1)
    shader = tmp_path / "probe.hlsl"
    shader.write_text(code, encoding="utf-8")
    spirv = tmp_path / "probe.spv"
    _run([dxc, "-T", "cs_6_0", "-E", entry, "-spirv", str(shader), "-Fo", str(spirv)])
    msl = _spirv_to_msl(spirv)
    assert _run_metal_kernel(msl, entry, len(EXPECTED)) == EXPECTED
