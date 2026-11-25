"""Generic calls on elements of pointer buffer globals in Metal output (#3764).

Metal output passes a global ``buffer float* data`` to the functions that use it and registers it there as
``float*``, so ``pick(data[i])`` deduces ``T = float`` while it is emitted. Generic collection, which runs first, did
not know these globals, so the call had no argument type until emission and the generator refused the program
(before that, it called a ``pick_float`` it never wrote). The specializations are now collected with the type that
emission registers; the output declares each one, compiles with warnings as errors and computes the source's
results on the GPU.
"""

import re
import shutil
import struct
import subprocess

import pytest

import crosstl

SCALARS = """shader GenericBufferElements {
    buffer float* data;
    buffer int* counts;
    buffer float* outp;
    buffer int* outc;
    T pick<T>(T x) { return x; }
    generic<T> T sel2(T a, T b) { return a > b ? a : b; }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = pick(data[i]) + 1.0 + sel2(data[0], data[1]);
            outc[i] = pick(counts[i]) * 2 + sel2(counts[0], counts[1]);
        }
    }
}
"""

STRUCTS = """shader GenericStructElements {
    struct Pt { float a; float b; };
    buffer Pt* pts;
    buffer float* outp;
    T pick<T>(T x) { return x; }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            Pt p = pick(pts[i]);
            outp[i] = p.a * 10.0 + pick(pts[i].b);
        }
    }
}
"""

DATA = [2.5, -1.0, 4.0, 0.5]
COUNTS = [3, 7, -2, 5]
POINTS = [(1.0, 2.0), (3.0, 4.0), (-5.0, 0.25), (6.5, -1.5)]


def _translate(tmp_path, source, name):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="metal")


def _compile(tmp_path, source, stem):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")
    artifact = tmp_path / f"{stem}.metal"
    artifact.write_text(source, encoding="utf-8")
    result = subprocess.run(
        [
            xcrun,
            "--sdk",
            "macosx",
            "metal",
            "-Werror",
            "-c",
            str(artifact),
            "-o",
            str(tmp_path / f"{stem}.air"),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 and "unable to find utility" in result.stderr:
        pytest.skip("the Metal compiler is not installed")
    assert result.returncode == 0, result.stdout + result.stderr


def _bindings(metal):
    """``{parameter name: buffer index}`` of the kernel's buffer parameters."""
    return {
        name: int(index)
        for name, index in re.findall(r"\*\s*(\w+)\s*\[\[buffer\((\d+)\)\]\]", metal)
    }


def _run(metal, inputs, outputs, count):
    """Dispatch ``count`` threads; ``inputs`` maps names to bytes, ``outputs`` maps names to struct formats."""
    mtl = pytest.importorskip("Metal")
    device = mtl.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")
    library, error = device.newLibraryWithSource_options_error_(metal, None, None)
    assert library is not None, error
    entry = re.search(r"\bkernel\s+void\s+(\w+)\s*\(", metal).group(1)
    pipeline, error = device.newComputePipelineStateWithFunction_error_(
        library.newFunctionWithName_(entry), None
    )
    assert pipeline is not None, error
    bindings = _bindings(metal)
    assert set(bindings) == set(inputs) | set(outputs), bindings
    commands = device.newCommandQueue().commandBuffer()
    encoder = commands.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    buffers = {}
    for name, words in inputs.items():
        buffers[name] = device.newBufferWithBytes_length_options_(words, len(words), 0)
    for name, fmt in outputs.items():
        buffers[name] = device.newBufferWithLength_options_(
            struct.calcsize(f"<{count}{fmt}"), 0
        )
    for name, buffer in buffers.items():
        encoder.setBuffer_offset_atIndex_(buffer, 0, bindings[name])
    encoder.dispatchThreads_threadsPerThreadgroup_(
        mtl.MTLSizeMake(count, 1, 1), mtl.MTLSizeMake(1, 1, 1)
    )
    encoder.endEncoding()
    commands.commit()
    commands.waitUntilCompleted()
    results = {}
    for name, fmt in outputs.items():
        size = struct.calcsize(f"<{count}{fmt}")
        results[name] = struct.unpack(
            f"<{count}{fmt}", buffers[name].contents().as_buffer(size)
        )
    return results


def test_metal_output_writes_each_specialization_it_calls(tmp_path):
    metal = _translate(tmp_path, SCALARS, "scalars.cgl")
    for name, signature in (
        ("pick_float", "float pick_float(float x)"),
        ("pick_int", "int pick_int(int x)"),
        ("sel2_float", "float sel2_float(float a, float b)"),
        ("sel2_int", "int sel2_int(int a, int b)"),
    ):
        assert signature in metal, metal
        assert f"{name}(" in metal.split("kernel", 1)[1], metal
    assert "pick_float(data[i])" in metal
    assert "sel2_int(counts[0], counts[1])" in metal
    _compile(tmp_path, metal, "scalars")


def test_struct_elements_deduce_the_struct_type(tmp_path):
    metal = _translate(tmp_path, STRUCTS, "structs.cgl")
    assert "Pt pick_Pt(Pt x)" in metal, metal
    assert "float pick_float(float x)" in metal, metal
    _compile(tmp_path, metal, "structs")


def test_scalar_elements_compute_the_source_results_on_the_gpu(tmp_path):
    metal = _translate(tmp_path, SCALARS, "scalars.cgl")
    _compile(tmp_path, metal, "scalars")
    results = _run(
        metal,
        {
            "data": struct.pack(f"<{len(DATA)}f", *DATA),
            "counts": struct.pack(f"<{len(COUNTS)}i", *COUNTS),
        },
        {"outp": "f", "outc": "i"},
        len(DATA),
    )
    assert results["outp"] == tuple(
        value + 1.0 + max(DATA[0], DATA[1]) for value in DATA
    )
    assert results["outc"] == tuple(
        value * 2 + max(COUNTS[0], COUNTS[1]) for value in COUNTS
    )


def test_struct_elements_compute_the_source_results_on_the_gpu(tmp_path):
    metal = _translate(tmp_path, STRUCTS, "structs.cgl")
    _compile(tmp_path, metal, "structs")
    words = b"".join(struct.pack("<2f", a, b) for a, b in POINTS)
    results = _run(metal, {"pts": words}, {"outp": "f"}, len(POINTS))
    assert results["outp"] == tuple(a * 10.0 + b for a, b in POINTS)
