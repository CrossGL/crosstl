"""MSL ``clz``/``ctz`` and CrossGL ``findMSB``/``findLSB`` across Metal and the other targets.

MSL has only the bit counts ``clz`` and ``ctz`` (MSL 3.2 section 6.13) and GLSL only the bit indices ``findMSB``
and ``findLSB`` (GLSL 4.60 section 8.8). Each used to be emitted verbatim on targets that lack it: ``clz(x)`` in
GLSL and HLSL, ``metal::clz`` as the mangled ``metal_u3a_u3aclz``, and ``findMSB(x)`` in MSL. The conformance cases
``m_bit_scans`` and ``g_bit_scans`` run both lowerings against their oracles.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.Metal.bit_scan_import import MetalBitScanError

HEADER = "#include <metal_stdlib>\nusing namespace metal;\n"

SCANS = """
kernel void scans(device uint* out [[buffer(0)]], device int2* pairs [[buffer(1)]],
                  device ushort* halves [[buffer(2)]], device char* bytes [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) {
    out[i] = ::metal::clz(out[i]) + metal::ctz(out[i]) + clz(out[i]);
    pairs[i] = clz(pairs[i]) + ctz(pairs[i]);
    halves[i] = clz(halves[i]) + ctz(halves[i]);
    bytes[i] = clz(bytes[i]) + ctz(bytes[i]);
}
"""

FIND = """
shader Find {
    compute {
        layout(std430, binding = 0) buffer Data { ivec2 signedValues[4]; uint values[]; };
        void main() @ numthreads(1, 1, 1) {
            uint x = values[0];
            values[1] = uint(findMSB(x)) + uint(findLSB(x));
            signedValues[1] = findMSB(signedValues[0]) + findLSB(signedValues[0]);
        }
    }
}
"""


def _translate(tmp_path, source, backend, name="shader.metal"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def test_metal_bit_counts_import_as_crossgl_bit_scans(tmp_path):
    cgl = _translate(tmp_path, HEADER + SCANS, "cgl")
    assert "metal_u3a" not in cgl
    assert "clz(" not in cgl and "ctz(" not in cgl
    assert "uint(31 - findMSB(buffer_load(out_, i)))" in cgl
    assert "uint(min(uint(findLSB(buffer_load(out_, i))), 32u))" in cgl
    # Signed and vector components scan their unsigned bit pattern.
    assert "ivec2(31 - findMSB(uvec2(buffer_load(pairs, i))))" in cgl
    # Narrow components are widened, masked and counted against their own width.
    assert "uint16(15 - findMSB((uint(buffer_load(halves, i)) & 0xffffu)))" in cgl
    assert "int8(min(uint(findLSB((uint(buffer_load(bytes, i)) & 0xffu))), 8u))" in cgl


@pytest.mark.parametrize("backend", ["opengl", "directx", "vulkan", "metal"])
def test_metal_bit_counts_reach_every_target(tmp_path, backend):
    code = _translate(tmp_path, HEADER + SCANS, backend)
    assert "metal_u3a" not in code
    if backend in {"opengl", "directx"}:
        assert "clz(" not in code and "ctz(" not in code


def test_a_64_bit_bit_count_is_refused(tmp_path):
    source = HEADER + (
        "kernel void k(device ulong* v [[buffer(0)]], uint i [[thread_position_in_grid]]) "
        "{ v[i] = clz(v[i]); }\n"
    )
    with pytest.raises(MetalBitScanError, match="ulong"):
        _translate(tmp_path, source, "cgl")


def test_a_user_function_named_clz_is_called_as_written(tmp_path):
    source = HEADER + (
        "uint clz(uint x) { return x + 1u; }\n"
        "kernel void k(device uint* v [[buffer(0)]], uint i [[thread_position_in_grid]]) "
        "{ v[i] = clz(v[i]); }\n"
    )
    cgl = _translate(tmp_path, source, "cgl")
    assert "findMSB" not in cgl


def test_metal_spells_bit_scans_through_helpers(tmp_path):
    metal = _translate(tmp_path, FIND, "metal", "find.cgl")
    assert (
        "static inline int crosstl_findMSB_uint(uint x) {\n    return int(31) - int(clz(x));"
        in metal
    )
    assert "return select(int(ctz(x)), int(-1), x == uint(0));" in metal
    assert "int2(31) - int2(clz(select(x, ~x, x < int2(0))));" in metal
    assert "uint(crosstl_findMSB_uint(x)) + uint(crosstl_findLSB_uint(x))" in metal
    assert "findMSB(" not in metal.replace("crosstl_findMSB", "")
    if shutil.which("xcrun") is None:
        return
    path = tmp_path / "find.metal"
    path.write_text(metal)
    result = subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(path),
            "-o",
            str(tmp_path / "find.air"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + metal


def test_metal_refuses_a_bit_scan_of_a_non_32_bit_operand(tmp_path):
    source = FIND.replace("uint(findLSB(x))", "uint(findLSB(1.5))")
    with pytest.raises(Exception, match="findLSB: the operand"):
        _translate(tmp_path, source, "metal", "find.cgl")
