"""Metal output declares a CrossGL ``readonly buffer T* name`` as a ``const device T*`` argument.

HLSL's ``StructuredBuffer``, a GLSL ``readonly buffer`` block and a WGSL-form ``var<storage, read>`` parameter already
became ``const device`` arguments, but a CrossGL ``readonly buffer float* fin;`` stage global became
``device float* fin``. That compiles and computes the same, but the Metal library reflects the argument as
read-write (``air.read_write``) where the source declares a read-only input, and a store through it reached the Metal
compiler as a valid store. The argument is now ``const device``, both in the entry and in the helpers the global is
passed to, its AIR access is ``air.read``, and a store or atomic through it is refused with a typed diagnostic.
"""

import re
import shutil
import struct
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused

COPY = """
shader ReadOnlyCopy {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        readonly buffer float* fin;
        buffer float* outp;

        float twice(uint i) { return fin[i] * 2.0; }

        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = twice(i) + fin[i];
        }
    }
}
"""

STORE = """
shader ReadOnlyStore {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        readonly buffer float* fin;
        buffer float* outp;

        void main() {
            uint i = gl_GlobalInvocationID.x;
            fin[i] = 1.0;
            outp[i] = fin[i];
        }
    }
}
"""

ATOMIC = """
shader ReadOnlyAtomic {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        readonly buffer int* fin;
        buffer int* outp;

        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = atomicAdd(fin[i], 1);
        }
    }
}
"""


def _translate(tmp_path, source):
    path = tmp_path / "probe.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend="metal")


def _air(tmp_path, metal):
    if shutil.which("xcrun") is None:
        pytest.skip("xcrun is unavailable")
    source = tmp_path / "probe.metal"
    source.write_text(metal)
    result = subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-S",
            "-emit-llvm",
            "-o",
            "-",
            str(source),
        ],
        capture_output=True,
        text=True,
    )
    if "unable to find utility" in result.stderr:
        pytest.skip("the Metal compiler is unavailable")
    assert result.returncode == 0, result.stderr
    return result.stdout


def _argument_access(air, name):
    """The AIR access (``air.read``, ``air.read_write``) of the buffer argument ``name``."""
    for line in air.splitlines():
        if '!"air.buffer"' in line and f'!"air.arg_name", !"{name}"' in line:
            match = re.search(r'!"(air\.(?:read|write|read_write))"', line)
            assert match, line
            return match.group(1)
    raise AssertionError(f"no buffer argument {name}")


def _run(metal, kernel, inputs, count):
    metal_api = pytest.importorskip("Metal")
    device = metal_api.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")
    options = metal_api.MTLCompileOptions.new()
    options.setMathMode_(metal_api.MTLMathModeSafe)
    library, error = device.newLibraryWithSource_options_error_(metal, options, None)
    assert library is not None, error
    pipeline, error = device.newComputePipelineStateWithFunction_error_(
        library.newFunctionWithName_(kernel), None
    )
    assert pipeline is not None, error
    commands = device.newCommandQueue().commandBuffer()
    encoder = commands.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    handles = {}
    for index, values in inputs.items():
        data = struct.pack(f"{len(values)}f", *values)
        handles[index] = device.newBufferWithBytes_length_options_(data, len(data), 0)
        encoder.setBuffer_offset_atIndex_(handles[index], 0, index)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        metal_api.MTLSizeMake(count, 1, 1), metal_api.MTLSizeMake(1, 1, 1)
    )
    encoder.endEncoding()
    commands.commit()
    commands.waitUntilCompleted()
    return {
        index: list(
            struct.unpack(
                f"{len(inputs[index])}f",
                handle.contents().as_buffer(4 * len(inputs[index])),
            )
        )
        for index, handle in handles.items()
    }


def test_readonly_buffer_pointer_is_a_const_device_argument(tmp_path):
    metal = _translate(tmp_path, COPY)
    assert re.search(r"const device float\s*\*\s*fin \[\[buffer\(0\)\]\]", metal), metal
    assert re.search(r"(?<!const )device float\s*\*\s*outp \[\[buffer\(1\)\]\]", metal)
    # The helper the global is passed to takes it as const device too.
    assert re.search(r"float twice\(uint i, const device float\s*\*\s*fin\)", metal), metal


def test_readonly_buffer_pointer_reflects_as_read_only(tmp_path):
    air = _air(tmp_path, _translate(tmp_path, COPY))
    assert _argument_access(air, "fin") == "air.read"
    assert _argument_access(air, "outp") == "air.read_write"


def test_readonly_buffer_pointer_computes_the_source(tmp_path):
    metal = _translate(tmp_path, COPY)
    results = _run(metal, "kernel_main", {0: [1.0, 2.0, 3.5], 1: [0.0] * 3}, 3)
    assert results[0] == [1.0, 2.0, 3.5]
    assert results[1] == [3.0, 6.0, 10.5]


@pytest.mark.parametrize(
    "source, construct",
    [
        (STORE, "parameter store: parameter 'fin' is readonly"),
        (ATOMIC, "buffer atomic: atomicAdd requires writable storage"),
    ],
)
def test_writes_through_a_readonly_buffer_pointer_are_refused(
    tmp_path, source, construct
):
    with pytest.raises(MetalLoweringRefused, match=re.escape(construct)):
        _translate(tmp_path, source)
