"""CrossGL structure member functions become free functions that take their receiver first."""

from __future__ import annotations

import re
import shutil
import struct
import subprocess

import pytest

import crosstl
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics import StructMethodUnsupportedError

SPAN = """shader P {
    struct Span {
        float lo;
        float hi;
        float width() {
            return hi - lo;
        }
        float at(float t) {
            return lo + t * width();
        }
        void widen(float d) {
            hi += d;
        }
        void grow(float d) {
            widen(d);
            this.lo -= d;
        }
        float clampTo(float lo) {
            return min(lo, this.hi);
        }
        static Span make(float a, float b) {
            Span s;
            s.lo = a;
            s.hi = b;
            return s;
        }
    };
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* iin;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            Span s = Span::make(float(iin[i]), float(iin[i]) * 2.0 + 1.0);
            float w = s.width();
            s.widen(3.0);
            Span t = s;
            t.grow(1.0);
            outp[i * 4u] = w + s.at(0.5) * 64.0;
            outp[i * 4u + 1u] = s.hi;
            outp[i * 4u + 2u] = t.lo + t.hi * 16.0;
            outp[i * 4u + 3u] = t.clampTo(100.0) + Span::make(1.0, 4.0).width();
        }
    }
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _flat(code):
    return re.sub(r"\s+", " ", code)


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_parser_keeps_member_functions_on_the_structure():
    program = Parser(Lexer(SPAN).get_tokens()).parse()
    (span,) = program.structs
    assert [member.name for member in span.members] == ["lo", "hi"]
    assert [method.name for method in span.methods] == [
        "width",
        "at",
        "widen",
        "grow",
        "clampTo",
        "make",
    ]
    assert span.methods[-1].qualifiers == ["static"]
    assert not any(function.name == "width" for function in program.functions)


def test_hlsl_receives_free_functions_that_take_the_receiver(tmp_path):
    code = _flat(_translate(tmp_path, SPAN, "directx"))
    assert "float Span_width(Span receiver) { return (receiver.hi - receiver.lo); }" in code
    assert "return (receiver.lo + (t * Span_width(receiver)));" in code
    # Only the methods that write their receiver take it inout, including through a writing method.
    assert "void Span_widen(inout Span receiver, float d) { receiver.hi += d; }" in code
    assert (
        "void Span_grow(inout Span receiver, float d) { Span_widen(receiver, d); receiver.lo -= d; }"
        in code
    )
    # A parameter named like a member hides the member.
    assert "float Span_clampTo(Span receiver, float lo) { return min(lo, receiver.hi); }" in code
    assert "Span Span_make(float a, float b)" in code
    assert "Span s = Span_make(float(iin[i]), ((float(iin[i]) * 2.0) + 1.0));" in code
    assert "float w = Span_width(s);" in code
    assert "Span_widen(s, 3.0);" in code
    assert "Span_grow(t, 1.0);" in code
    assert "Span_width(Span_make(1.0, 4.0))" in code
    assert ".width(" not in code and "::" not in code


@pytest.mark.parametrize(
    "backend", ["metal", "opengl", "vulkan", "cuda", "hip", "slang", "wgsl", "mojo"]
)
def test_every_target_calls_the_lowered_member_functions(tmp_path, backend):
    code = _translate(tmp_path, SPAN, backend)
    assert "Span_widen" in code and "Span_make" in code or backend == "vulkan"
    assert ".width(" not in code and ".widen(" not in code


def test_opengl_output_validates(tmp_path):
    if not shutil.which("glslangValidator"):
        pytest.skip("glslangValidator is not on PATH")
    path = tmp_path / "shader.comp"
    path.write_text(_translate(tmp_path, SPAN, "opengl"), encoding="utf-8")
    _run(["glslangValidator", "-S", "comp", str(path)])


def test_spirv_output_validates(tmp_path):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("spirv-as and spirv-val are not on PATH")
    path = tmp_path / "shader.spvasm"
    path.write_text(_translate(tmp_path, SPAN, "vulkan"), encoding="utf-8")
    binary = tmp_path / "shader.spv"
    _run(["spirv-as", "--target-env", "vulkan1.1", str(path), "-o", str(binary)])
    _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])


def test_metal_output_computes_the_member_function_results(tmp_path):
    Metal = pytest.importorskip("Metal")
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")
    code = _translate(tmp_path, SPAN, "metal")
    slots = dict(re.findall(r"(\w+) \[\[buffer\((\d+)\)\]\]", code))
    library, error = device.newLibraryWithSource_options_error_(
        code, Metal.MTLCompileOptions.new(), None
    )
    assert library is not None, error
    entry = re.search(r"kernel void (\w+)", code).group(1)
    pipeline, error = device.newComputePipelineStateWithFunction_error_(
        library.newFunctionWithName_(entry), None
    )
    assert pipeline is not None, error
    inputs = struct.pack("<2i", 2, 5)
    iin = device.newBufferWithBytes_length_options_(
        inputs, len(inputs), Metal.MTLResourceStorageModeShared
    )
    results = device.newBufferWithLength_options_(32, Metal.MTLResourceStorageModeShared)
    command = device.newCommandQueue().commandBuffer()
    encoder = command.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(iin, 0, int(slots["iin"]))
    encoder.setBuffer_offset_atIndex_(results, 0, int(slots["outp"]))
    encoder.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake(2, 1, 1), Metal.MTLSizeMake(1, 1, 1)
    )
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    assert command.error() is None, command.error()
    data = struct.unpack("<8f", bytes(results.contents().as_buffer(32)))
    expected = []
    for value in (2.0, 5.0):
        lo, hi = value, value * 2.0 + 1.0
        w = hi - lo
        hi += 3.0
        at = lo + 0.5 * (hi - lo)
        t_lo, t_hi = lo - 1.0, hi + 1.0
        expected += [w + at * 64.0, hi, t_lo + t_hi * 16.0, min(100.0, t_hi) + 3.0]
    assert data == tuple(expected)


def test_a_writing_member_function_needs_an_lvalue_object(tmp_path):
    source = SPAN.replace("t.grow(1.0);", "Span::make(1.0, 2.0).widen(1.0);")
    with pytest.raises(StructMethodUnsupportedError, match="not an lvalue"):
        _translate(tmp_path, source, "directx")


def test_generic_structures_with_member_functions_are_refused(tmp_path):
    source = """shader P {
    struct Box<T> {
        T value;
        T get() { return value; }
    };
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        void main() {
            Box<float> b;
            b.value = 1.0;
            outp[0] = b.get();
        }
    }
}
"""
    with pytest.raises(StructMethodUnsupportedError, match="generic structure 'Box'"):
        _translate(tmp_path, source, "directx")


def test_a_member_function_name_taken_by_a_user_function_gets_a_fresh_name(
    tmp_path,
):
    source = SPAN.replace(
        "    compute {", "    float Span_width(float x) { return x; }\n    compute {", 1
    )
    code = _flat(_translate(tmp_path, source, "directx"))
    assert "float Span_width(float x) { return x; }" in code
    assert "float Span_width_2(Span receiver)" in code
    assert "float w = Span_width_2(s);" in code


def test_texture_methods_named_like_member_functions_are_left_alone(tmp_path):
    source = """shader P {
    struct Lookup {
        float bias;
        float Sample(float x) { return x + bias; }
    };
    fragment {
        uniform Texture2D tex;
        uniform SamplerState samp;
        vec4 main(vec2 uv @ TEXCOORD0) @ SV_Target0 {
            Lookup l;
            l.bias = 0.5;
            return tex.Sample(samp, uv) + vec4(l.Sample(1.0));
        }
    }
}
"""
    code = _flat(_translate(tmp_path, source, "directx"))
    assert "tex.Sample(samp, uv)" in code
    assert "Lookup_Sample(l, 1.0)" in code
