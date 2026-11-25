"""Ordering of the partial struct specializations a Metal template-id matches (#1987).

A template-id that several partial specializations match names the most specialized one, and is ill-formed when
none is more specialized than all the others ([temp.spec.partial.order]). Metal import used to refuse every such
template-id with an enable_if candidate and to fall back to the primary template for the others, which changed the
field types silently. A partial specialization that spells fewer arguments than the primary also takes the
primary's defaults, so ``Cell<T, device U*>`` of ``Cell<T, U, Enable = void>`` is matched as
``Cell<T, device U*, void>``.
"""

import shutil
import struct
import subprocess

import pytest

import crosstl
from crosstl.backend.Metal.preprocessor import MetalTemplateSpecializationError

HEADER = "#include <metal_stdlib>\nusing namespace metal;\n"

KERNEL = """
kernel void probe(device const uint* in [[buffer(0)]], device uint* out [[buffer(1)]],
                  uint i [[thread_position_in_grid]]) {
BODY
}
"""

# Cell<T, uint, void> is more specialized than Cell<T, U, enable_if_t<true>>.
CONSTRAINED = (
    HEADER
    + """
template <typename T, typename U, typename = void>
struct Cell { uint value; };
template <typename T, typename U>
struct Cell<T, U, enable_if_t<true>> { int value; };
template <typename T>
struct Cell<T, uint, void> { float value; };
"""
    + KERNEL.replace(
        "BODY",
        """  Cell<float, uint> cell;
  cell.value = 1.5f + float(in[i] & 1u);
  out[i] = as_type<uint>(cell.value);""",
    )
)

# Cell<T, device uint*> is more specialized than Cell<T, device U*>.
POINTER = (
    HEADER
    + """
template <typename T, typename U>
struct Cell { uint value; };
template <typename T, typename U>
struct Cell<T, device U*> { int value; };
template <typename T>
struct Cell<T, device uint*> { float value; };
"""
    + KERNEL.replace(
        "BODY",
        """  Cell<float, device uint*> cell;
  cell.value = 1.5f + float(in[i] & 1u);
  out[i] = as_type<uint>(cell.value);""",
    )
)

# Three uses: Cell<T, device T*> wins over Cell<T, device U*> for <float, device float*>, the latter alone matches
# <float, device int*>, and the primary is used for <float, uint>. The partial specializations omit the defaulted
# Enable parameter, and each selected specialization keeps its own member function.
DEFAULTED = (
    HEADER
    + """
template <typename T, typename U, typename Enable = void>
struct Cell {
  uint value;
  uint bits() const { return value; }
};
template <typename T, typename U>
struct Cell<T, device U*> {
  int value;
  uint bits() const { return uint(value); }
};
template <typename T>
struct Cell<T, device T*> {
  half value;
  uint bits() const { return uint(as_type<ushort>(value)); }
};
"""
    + KERNEL.replace(
        "BODY",
        """  float x = 1.5f + float(in[i] & 1u);
  Cell<float, device float*> same;
  same.value = half(x);
  Cell<float, device int*> other;
  other.value = int(x * 2.0f);
  Cell<float, uint> primary;
  primary.value = uint(x * 4.0f);
  out[i] = same.bits() + (other.bits() << 16) + (primary.bits() << 24);""",
    )
)


def _defaulted_expected(word):
    x = 1.5 + (word & 1)
    half_bits = struct.unpack("<H", struct.pack("<e", x))[0]
    return half_bits + (int(x * 2.0) << 16) + (int(x * 4.0) << 24)


CASES = {
    "constrained": (
        CONSTRAINED,
        "float value;",
        lambda word: struct.unpack("<I", struct.pack("<f", 1.5 + (word & 1)))[0],
    ),
    "pointer": (
        POINTER,
        "float value;",
        lambda word: struct.unpack("<I", struct.pack("<f", 1.5 + (word & 1)))[0],
    ),
    "defaulted": (DEFAULTED, "half value;", _defaulted_expected),
}

COUNT = 8

# Clang (and so the Metal compiler) rejects both: "ambiguous partial specializations of 'Cell<...>'".
AMBIGUOUS = (
    HEADER
    + """
template <typename T, typename U>
struct Cell { uint value; };
template <typename U>
struct Cell<float, U> { int value; };
template <typename T>
struct Cell<T, device uint*> { float value; };
"""
    + KERNEL.replace(
        "BODY",
        "  Cell<float, device uint*> cell;\n  cell.value = 1;\n  out[i] = uint(cell.value);",
    )
)

DEPENDENT_CONSTRAINT = CONSTRAINED.replace(
    "enable_if_t<true>", "enable_if_t<is_same_v<T, float>>"
)


def _translate(tmp_path, source, backend):
    path = tmp_path / "probe.metal"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def _run(source):
    """Dispatch eight threads with in[i] = i and return out[0..7]."""
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
    data = struct.pack(f"<{COUNT}I", *range(COUNT))
    inputs = device.newBufferWithBytes_length_options_(data, len(data), 0)
    outputs = device.newBufferWithLength_options_(len(data), 0)
    commands = device.newCommandQueue().commandBuffer()
    encoder = commands.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(inputs, 0, 0)
    encoder.setBuffer_offset_atIndex_(outputs, 0, 1)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        metal.MTLSizeMake(COUNT, 1, 1), metal.MTLSizeMake(COUNT, 1, 1)
    )
    encoder.endEncoding()
    commands.commit()
    commands.waitUntilCompleted()
    return list(
        struct.unpack(f"<{COUNT}I", bytes(outputs.contents().as_buffer(len(data))))
    )


def _metal_compile(tmp_path, source):
    """Compile with the Metal compiler; None when it is not installed."""
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        return None
    artifact = tmp_path / "generated.metal"
    artifact.write_text(source, encoding="utf-8")
    result = subprocess.run(
        [xcrun, "--sdk", "macosx", "metal", "-Werror", "-c", str(artifact)]
        + ["-o", str(tmp_path / "generated.air")],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 and "unable to find utility" in result.stderr:
        return None
    return result


@pytest.mark.parametrize("case", sorted(CASES))
def test_metal_round_trip_uses_the_most_specialized_partial(tmp_path, case):
    source, field, expected = CASES[case]

    metal = _translate(tmp_path, source, "metal")

    assert "Cell<" not in metal
    assert field in metal
    result = _metal_compile(tmp_path, metal)
    if result is None:
        pytest.skip("the Metal compiler is not installed")
    assert result.returncode == 0, result.stdout + result.stderr
    wanted = [expected(word) for word in range(COUNT)]
    assert _run(source) == wanted
    assert _run(metal) == wanted


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("backend", ["directx", "opengl"])
def test_other_targets_declare_the_selected_fields(tmp_path, case, backend):
    source, field, _ = CASES[case]

    code = _translate(tmp_path, source, backend)

    selected = {
        ("directx", "half value;"): "float16_t value;",
        ("opengl", "half value;"): "float value;",
    }.get((backend, field), field)
    assert "Cell<" not in code
    assert selected in code
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


@pytest.mark.parametrize(
    "source,message",
    [
        (AMBIGUOUS, "is ambiguous: it matches the partial specializations"),
        (DEPENDENT_CONSTRAINT, "whose ordering is not supported"),
    ],
    ids=["ambiguous", "dependent-constraint"],
)
@pytest.mark.parametrize("backend", ["metal", "directx", "opengl"])
def test_unordered_partials_are_refused(tmp_path, source, message, backend):
    native = _metal_compile(tmp_path, source)
    if native is not None:
        assert native.returncode != 0
        assert "ambiguous partial specializations" in native.stderr

    with pytest.raises(MetalTemplateSpecializationError, match=message):
        _translate(tmp_path, source, backend)
