"""Objects of a class-template partial specialization in Metal input (#2261).

``Block<float2, 8> special; special.run(out);`` names the partial specialization ``Block<float2, M>``
([temp.spec.partial.match]). Metal import used to materialize only the primary template's uses: with no use of
the primary, the declaration kept its template spelling and the member call had no definition, and with one it
refused the receiver. The selected specialization's methods and static constants are now materialized for every
concrete object, and every target computes the specialization's body.
"""

import shutil
import struct
import subprocess

import pytest

import crosstl

HEADER = "#include <metal_stdlib>\nusing namespace metal;\n"

BLOCK = """
template <typename T, int M>
struct Block {
  void run(device float* out) { out[0] = float(M); }
};

template <int M>
struct Block<float2, M> {
  void run(device float* out) { out[0] = float(M) + 0.5f; }
};
"""

SPECIALIZATION_ONLY = HEADER + BLOCK + """
kernel void probe(device float* out [[buffer(0)]]) {
  Block<float2, 8> special;
  special.run(out);
}
"""

WITH_PRIMARY = HEADER + BLOCK + """
kernel void probe(device float* out [[buffer(0)]]) {
  Block<float2, 8> special;
  special.run(out);
  Block<float, 4> primary;
  primary.run(out + 1);
}
"""

STATIC_CONSTANT = HEADER + """
template <typename T, int M>
struct Block {
  static constant constexpr const short TM = M;
  void run(device float* out, const threadgroup T* tile) { out[0] = float(TM); }
};

template <int M>
struct Block<float2, M> {
  static constant constexpr const short TM = M / 2;
  void run(device float* out, const threadgroup float2* tile) {
    float v[TM];
    for (short i = 0; i < TM; ++i) { v[i] = tile[i].x + tile[i].y; }
    float s = 0.0f;
    for (short i = 0; i < TM; ++i) { s += v[i]; }
    out[0] = s + float(TM);
  }
};

kernel void probe(device float* out [[buffer(0)]],
                  uint lid [[thread_position_in_threadgroup]]) {
  threadgroup float2 tile[8];
  if (lid < 8) { tile[lid] = float2(float(lid), 0.25f); }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    Block<float2, 8> special;
    special.run(out, tile);
  }
}
"""

CASES = {
    "specialization-only": (SPECIALIZATION_ONLY, ["Block_float2_8__run"], [8.5]),
    "with-primary": (
        WITH_PRIMARY,
        ["Block_float2_8__run", "Block_float_4__run"],
        [8.5, 4.0],
    ),
    # TM = 8 / 2 = 4: (0 + 1 + 2 + 3) + 4 * 0.25 + 4.
    "static-constant": (STATIC_CONSTANT, ["Block_float2_8__run"], [11.0]),
}


def _translate(tmp_path, source, backend):
    path = tmp_path / "probe.metal"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def _run(source, count):
    """Dispatch one threadgroup of eight threads and return the first ``count`` floats of buffer 0."""
    metal = pytest.importorskip("Metal")
    device = metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")
    library, error = device.newLibraryWithSource_options_error_(source, None, None)
    assert library is not None, error
    pipeline, error = device.newComputePipelineStateWithFunction_error_(
        library.newFunctionWithName_("probe"), None
    )
    assert pipeline is not None, error
    data = struct.pack("<8f", *([-1.0] * 8))
    out = device.newBufferWithBytes_length_options_(data, len(data), 0)
    commands = device.newCommandQueue().commandBuffer()
    encoder = commands.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(out, 0, 0)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        metal.MTLSizeMake(8, 1, 1), metal.MTLSizeMake(8, 1, 1)
    )
    encoder.endEncoding()
    commands.commit()
    commands.waitUntilCompleted()
    raw = bytes(out.contents().as_buffer(len(data)))
    return list(struct.unpack("<8f", raw))[:count]


def _compile_metal(tmp_path, source):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")
    artifact = tmp_path / "generated.metal"
    artifact.write_text(source, encoding="utf-8")
    result = subprocess.run(
        [xcrun, "--sdk", "macosx", "metal", "-Werror", "-c", str(artifact)]
        + ["-o", str(tmp_path / "generated.air")],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 and "unable to find utility" in result.stderr:
        pytest.skip("the Metal compiler is not installed")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("case", sorted(CASES))
def test_metal_round_trip_computes_the_partial_specialization(tmp_path, case):
    source, methods, expected = CASES[case]

    metal = _translate(tmp_path, source, "metal")

    assert "Block<" not in metal
    for method in methods:
        assert f"{method}(" in metal
    _compile_metal(tmp_path, metal)
    assert _run(source, len(expected)) == expected
    assert _run(metal, len(expected)) == expected


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("backend", ["directx", "opengl"])
def test_other_targets_define_the_specialization_methods(tmp_path, case, backend):
    source, methods, _ = CASES[case]

    code = _translate(tmp_path, source, backend)

    assert "Block<" not in code
    assert "float(8) + 0.5" in code or "float(4)" in code
    for method in methods:
        # Prototype, definition and call. GLSL output joins the receiver and method with one underscore
        # and appends the parameter spellings.
        name = method.replace("__", "_") if backend == "opengl" else method
        assert code.count(name) >= 3, code
    if backend == "opengl":
        compiler = shutil.which("glslangValidator")
        if compiler is None:
            pytest.skip("glslangValidator is not installed")
        path = tmp_path / "generated.comp"
        path.write_text(code, encoding="utf-8")
        result = subprocess.run(
            [compiler, "-l", "-S", "comp", str(path)]
            # A Metal kernel leaves its threadgroup size to the host.
            + [
                "-DCROSSGL_WORKGROUP_SIZE_X=8",
                "-DCROSSGL_WORKGROUP_SIZE_Y=1",
                "-DCROSSGL_WORKGROUP_SIZE_Z=1",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
