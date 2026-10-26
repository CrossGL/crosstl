"""Slang and WGSL keep the parentheses that postfix and prefix unary operands need.

The parser drops source parentheses, so an emitter must restore them from operator binding. In Slang,
``(p + q).yx`` must not become ``p + q.yx``, which compiles and swizzles only ``q``, and ``-(-a)`` must not become
``--a``, which compiles as a pre-decrement and changes ``a``. WGSL has no prefix decrement, so ``--a`` is rejected.
"""

import shutil
import struct
import subprocess

import pytest

import crosstl.translator
from crosstl.translator.codegen.c_family_operands import (
    is_postfix_chain,
    postfix_operand,
    prefix_unary_operand,
)
from crosstl.translator.codegen.slang_codegen import SlangCodeGen
from crosstl.translator.codegen.wgsl_codegen import WGSLCodeGen

PROBE = """
shader SlangOperandProbe {
    compute {
        layout(std430, binding = 0) buffer Inputs { float values[]; } inputs;
        layout(std430, binding = 1) buffer Results { float data[]; } results;
        [numthreads(1, 1, 1)]
        void main() {
            float a = inputs.values[0];
            vec2 p = vec2(inputs.values[1], inputs.values[2]);
            vec2 q = vec2(inputs.values[3], inputs.values[4]);
            int i = int(inputs.values[5]);
            float negated = -(-a);
            vec2 swapped = (p + q).yx;
            int twice = -(-i);
            results.data[0] = negated;
            results.data[1] = a;
            results.data[2] = swapped.x;
            results.data[3] = swapped.y;
            results.data[4] = float(twice);
            results.data[5] = float(i);
        }
    }
}
"""

INPUTS = (5.0, 1.0, 2.0, 10.0, 20.0, 7.0)
# CrossGL: a and i are read only, and the swizzle applies to p + q.
EXPECTED = (5.0, 5.0, 22.0, 11.0, 7.0, 7.0)

COMPILE_PROBE = """
shader SlangOperandCompileProbe {
    compute {
        layout(std430, binding = 0) buffer Results { float data[]; } results;
        uniform sampler2D tex;
        [numthreads(1, 1, 1)]
        void main() {
            vec3 p = vec3(1.0, 2.0, 3.0);
            vec3 q = vec3(4.0, 5.0, 6.0);
            float scaled = (p * 2.0).y;
            float indexed = (p + q)[1];
            float chained = (p + q).xy.y;
            float negatedMember = (-p).z;
            int i = 2;
            int decremented = -(--i);
            int thrice = -(-(-i));
            float constructed = (vec2(scaled, indexed) + p.xy).x;
            vec4 projected = textureProjLod(tex, vec3(p.xy, 2.0) * 0.5, 0.0);
            results.data[0] = scaled + indexed + chained + negatedMember
                + float(decremented) + float(thrice) + constructed + projected.x;
        }
    }
}
"""


def _generate(source, generator=SlangCodeGen):
    return generator().generate(crosstl.translator.parse(source))


def _assignment(code, name):
    for line in code.splitlines():
        stripped = line.strip()
        if f" {name} = " in f" {stripped}":
            return stripped
    raise AssertionError(f"no assignment to {name} in:\n{code}")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("p", True),
        ("p.x", True),
        ("f(a, b).y", True),
        ("a[i][j].x", True),
        ("(p + q)", True),
        ("(p + q).x", True),
        ("s[i]->m", True),
        ("ns::value", True),
        ("float2(a, b)", True),
        ("p + q", False),
        ("-a", False),
        ("(float)x", False),
        ("(T)(x)", False),
        ("(a) + (b)", False),
        ("c ? a : b", False),
        ("vector<float, 2>(a)", False),
        ("a.", False),
        ("", False),
    ],
)
def test_postfix_chain_recognition(text, expected):
    assert is_postfix_chain(text) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("p + q", "(p + q)"),
        ("(p + q)", "(p + q)"),
        ("p.x", "p.x"),
        ("2.0", "(2.0)"),
        ("1u", "(1u)"),
        ("-a", "(-a)"),
        ("(float)x", "((float)x)"),
    ],
)
def test_postfix_operand(text, expected):
    assert postfix_operand(text) == expected


@pytest.mark.parametrize(
    ("operator", "text", "expected"),
    [
        ("-", "a", "a"),
        ("-", "-a", "(-a)"),
        ("-", "--a", "(--a)"),
        ("+", "+a", "(+a)"),
        ("&", "&a", "(&a)"),
        ("--", "-a", "(-a)"),
        ("-", "~a", "~a"),
        ("~", "~a", "~a"),
        ("!", "!a", "!a"),
        ("-", "+a", "+a"),
        ("-", "a - b", "(a - b)"),
        ("-", "(float)x", "((float)x)"),
        ("-", "2.0", "2.0"),
    ],
)
def test_prefix_unary_operand(operator, text, expected):
    assert prefix_unary_operand(operator, text) == expected


def test_member_access_on_binary_keeps_parentheses():
    code = _generate(PROBE)

    assert _assignment(code, "swapped") == "float2 swapped = (p + q).yx;"


def test_nested_negation_is_not_a_decrement():
    code = _generate(PROBE)

    assert _assignment(code, "negated") == "float negated = -(-a);"
    assert _assignment(code, "twice") == "int twice = -(-i);"
    assert "--a" not in code
    assert "--i" not in code


def test_postfix_and_prefix_operands_in_every_position():
    code = _generate(COMPILE_PROBE)

    assert _assignment(code, "scaled") == "float scaled = (p * 2.0).y;"
    assert _assignment(code, "indexed") == "float indexed = (p + q)[1];"
    assert _assignment(code, "chained") == "float chained = (p + q).xy.y;"
    assert _assignment(code, "negatedMember") == "float negatedMember = (-p).z;"
    assert _assignment(code, "decremented") == "int decremented = -(--i);"
    assert _assignment(code, "thrice") == "int thrice = -(-(-i));"
    assert (
        _assignment(code, "constructed")
        == "float constructed = (float2(scaled, indexed) + p.xy).x;"
    )
    assert "(float3(p.xy, 2.0) * 0.5).xy / (float3(p.xy, 2.0) * 0.5).z" in code


def test_simple_chains_gain_no_parentheses():
    code = _generate(COMPILE_PROBE)

    assert "float3 p = float3(1.0, 2.0, 3.0);" in code
    assert "(p).x" not in code
    assert "(p.xy)" not in code


def _slangc():
    slangc = shutil.which("slangc")
    if slangc is None:
        pytest.skip("slangc is not installed")
    return slangc


def _compile_spirv(tmp_path, code):
    slangc = _slangc()
    shader = tmp_path / "probe.slang"
    shader.write_text(code, encoding="utf-8")
    spirv = tmp_path / "probe.spv"
    completed = subprocess.run(
        [
            slangc,
            str(shader),
            "-target",
            "spirv",
            "-entry",
            "main",
            "-stage",
            "compute",
            "-o",
            str(spirv),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return spirv


@pytest.mark.parametrize("source", [PROBE, COMPILE_PROBE], ids=["runtime", "forms"])
def test_parenthesized_operands_compile_with_slangc(tmp_path, source):
    _compile_spirv(tmp_path, _generate(source))


def _run_metal_kernel(metal_source, inputs, result_count):
    Metal = pytest.importorskip("Metal")
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")
    library, error = device.newLibraryWithSource_options_error_(
        metal_source, Metal.MTLCompileOptions.new(), None
    )
    assert library is not None, error
    function = library.newFunctionWithName_("main0")
    pipeline, error = device.newComputePipelineStateWithFunction_error_(function, None)
    assert pipeline is not None, error

    shared = Metal.MTLResourceStorageModeShared
    result_size = 4 * result_count
    buffers = [
        device.newBufferWithBytes_length_options_(inputs, len(inputs), shared),
        device.newBufferWithLength_options_(result_size, shared),
    ]
    command = device.newCommandQueue().commandBuffer()
    encoder = command.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    for index, buffer in enumerate(buffers):
        encoder.setBuffer_offset_atIndex_(buffer, 0, index)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1), Metal.MTLSizeMake(1, 1, 1)
    )
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    assert command.error() is None, command.error()
    data = bytes(buffers[-1].contents().as_buffer(result_size))
    return struct.unpack(f"<{result_count}f", data)


def test_parenthesized_operands_match_crossgl_on_the_gpu(tmp_path):
    spirv = _compile_spirv(tmp_path, _generate(PROBE))
    spirv_cross = shutil.which("spirv-cross")
    if spirv_cross is None:
        pytest.skip("spirv-cross is not installed")
    converted = subprocess.run(
        [spirv_cross, "--msl", "--msl-version", "20100", str(spirv)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert converted.returncode == 0, converted.stderr
    inputs = struct.pack(f"<{len(INPUTS)}f", *INPUTS)

    assert _run_metal_kernel(converted.stdout, inputs, len(EXPECTED)) == EXPECTED


def test_wgsl_nested_negation_keeps_parentheses():
    code = _generate(PROBE, WGSLCodeGen)

    assert "var negated: f32 = -(-a);" in code
    assert "var twice: i32 = -(-i);" in code
    assert "--a" not in code
    assert "--i" not in code


def test_wgsl_nested_negation_validates_with_naga(tmp_path):
    naga = shutil.which("naga")
    if naga is None:
        pytest.skip("naga is not installed")
    shader = tmp_path / "probe.wgsl"
    shader.write_text(_generate(PROBE, WGSLCodeGen), encoding="utf-8")

    completed = subprocess.run(
        [naga, str(shader)], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
