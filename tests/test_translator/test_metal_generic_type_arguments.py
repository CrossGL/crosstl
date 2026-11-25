"""Leading type arguments of a generic call select its specialization and are not passed to it.

``bump(vec2, w, w)`` names ``T = vec2`` with its first argument, the same call as ``bump<vec2>(w, w)``. Metal and
GLSL output call the specialization with the value arguments only, the Metal compiler and glslang accept the result,
and the Metal output computes on the GPU what the CrossGL program computes. Leading arguments that are values are
still passed.
"""

import os
import re
import shutil
import struct
import subprocess

import pytest

import crosstl

SOURCE = """
shader main {
    buffer float* results;
    void bump<T>(inout T value, T step) { value = value + step; }
    T twice<T>(T value) { return value + value; }
    compute {
    @ numthreads(1, 1, 1)
    void main() {
        vec2 w = vec2(1.0, 3.0);
        bump(vec2, w, w);
        float s = 0.5;
        bump(s, s);
        int n = 3;
        bump(int, n, 4);
        results[0] = w.x;
        results[1] = w.y;
        results[2] = s;
        results[3] = twice(float, s);
        results[4] = float(twice(int, n));
        results[5] = twice(s);
    }
    }
}
"""

# w = (2, 6); s = 1; n = 7; twice(float, s) = 2; twice(int, n) = 14; twice(s) = 2.
EXPECTED = (2.0, 6.0, 1.0, 2.0, 14.0, 2.0)


def _translate(tmp_path, backend, source=SOURCE):
    path = tmp_path / "generic.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(os.fspath(path), backend=backend, format_output=False)


def _calls(code, name):
    return re.findall(rf"\b{name}_\w+\(([^()]*)\)\s*;", code)


@pytest.mark.parametrize("backend", ["metal", "opengl"])
def test_leading_type_arguments_are_not_passed(tmp_path, backend):
    code = _translate(tmp_path, backend)
    for arguments in _calls(code, "bump"):
        assert not re.match(r"\s*(vec2|float2|int)\b", arguments), code
    body = code[code.index("main") :]
    assert not re.search(
        r"\b(twice|bump)\w*\(\s*(vec2|float2|float|int)\s*,", body
    ), code
    assert re.search(r"\bbump_\w+\(\s*w\s*,\s*w\s*\)", code), code


def test_a_leading_value_argument_is_still_passed(tmp_path):
    source = SOURCE.replace("bump(vec2, w, w);", "bump(w, w);")
    code = _translate(tmp_path, "metal", source)
    assert re.search(r"\bbump_\w+\(\s*w\s*,\s*w\s*\)", code), code


def test_glsl_output_validates(tmp_path):
    validator = shutil.which("glslangValidator")
    if validator is None:
        pytest.skip("glslangValidator is not installed")
    path = tmp_path / "generic.comp"
    path.write_text(_translate(tmp_path, "opengl"), encoding="utf-8")
    result = subprocess.run(
        [validator, "-S", "comp", os.fspath(path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _run_on_gpu(metal_source):
    Metal = pytest.importorskip("Metal")
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")
    options = Metal.MTLCompileOptions.new()
    options.setMathMode_(Metal.MTLMathModeSafe)
    library, error = device.newLibraryWithSource_options_error_(
        metal_source, options, None
    )
    assert library is not None, f"{error}\n{metal_source}"
    entry = re.search(r"\bkernel\s+void\s+(\w+)\s*\(", metal_source)
    assert entry, metal_source
    function = library.newFunctionWithName_(entry.group(1))
    pipeline, error = device.newComputePipelineStateWithFunction_error_(function, None)
    assert pipeline is not None, error

    size = 4 * len(EXPECTED)
    output = device.newBufferWithLength_options_(
        size, Metal.MTLResourceStorageModeShared
    )
    command = device.newCommandQueue().commandBuffer()
    encoder = command.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(output, 0, 0)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1), Metal.MTLSizeMake(1, 1, 1)
    )
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    assert command.error() is None, command.error()
    return struct.unpack(f"<{len(EXPECTED)}f", bytes(output.contents().as_buffer(size)))


def test_metal_output_computes_the_program_on_the_gpu(tmp_path):
    metal = _translate(tmp_path, "metal")
    assert _run_on_gpu(metal) == EXPECTED
