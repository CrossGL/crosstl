"""SPIR-V resource parameters of Metal entry points and helpers.

Explicit storage-buffer bindings of entry-point parameters stay free for those parameters.

MSL numbers buffers, textures and samplers in separate index spaces (MSL 5.2.1), so a kernel can take both
``texture2d<float> source [[texture(0)]]`` and ``device float *out [[buffer(0)]]``. SPIR-V keeps the buffer's
index as its Vulkan binding and allocates the texture a free one. The texture parameter is processed first and used
to take binding 0, so the buffer was then refused as a duplicate binding.

A helper's storage-image parameter is a value, not a memory object declaration, so only the caller's variable takes
the ``NonWritable``/``NonReadable`` decoration of its access; spirv-val rejected the decoration on the parameter.
"""

import re
import shutil
import subprocess

import pytest

import crosstl

TEXTURE_THEN_BUFFER = """
#include <metal_stdlib>
using namespace metal;

kernel void copy_red(texture2d<float> source [[texture(0)]],
                     texture2d<float> mask [[texture(1)]],
                     device float *destination [[buffer(0)]],
                     device float *weights [[buffer(1)]],
                     uint2 gid [[thread_position_in_grid]]) {
    uint index = gid.y * 16 + gid.x;
    destination[index] = source.read(gid).x * mask.read(gid).x * weights[index];
}
"""

LIBRARY = """
#include <metal_stdlib>
using namespace metal;

kernel void read_texture(texture2d<float> source [[texture(0)]],
                         device float *destination [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    destination[gid.y * 16 + gid.x] = source.read(gid).x;
}

kernel void scale_buffer(device float *values [[buffer(0)]],
                         uint gid [[thread_position_in_grid]]) {
    values[gid] = values[gid] * 2.0f;
}
"""


IMAGE_HELPERS = """
#include <metal_stdlib>
using namespace metal;

static half3 fetch(texture2d<half, access::read> source, uint2 at) {
    return source.read(at).rgb;
}

static void put(texture2d<half, access::write> destination, uint2 at, half3 value) {
    destination.write(half4(value, 1.0h), at);
}

kernel void copy(texture2d<half, access::read> source [[texture(0)]],
                 texture2d<half, access::write> destination [[texture(1)]],
                 uint2 gid [[thread_position_in_grid]]) {
    put(destination, gid, fetch(source, gid));
}
"""


def _translate(tmp_path, source):
    path = tmp_path / "kernels.metal"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(
        str(path), backend="spirv", source_options={"threadgroup_size": (8, 8, 1)}
    )


def _bindings(spirv):
    names = dict(re.findall(r'OpName (%\w+) "(\w+)"', spirv))
    return {
        names.get(variable, variable): int(binding)
        for variable, binding in re.findall(r"OpDecorate (%\w+) Binding (\d+)", spirv)
    }


def _validate(spirv, tmp_path):
    spirv_as, spirv_val = shutil.which("spirv-as"), shutil.which("spirv-val")
    if not spirv_as or not spirv_val:
        return
    assembly, module = tmp_path / "kernels.spvasm", tmp_path / "kernels.spv"
    assembly.write_text(spirv, encoding="utf-8")
    for command in (
        [spirv_as, "--target-env", "vulkan1.1", str(assembly), "-o", str(module)],
        [spirv_val, "--target-env", "vulkan1.1", str(module)],
    ):
        completed = subprocess.run(command, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stdout + completed.stderr


def test_textures_take_bindings_the_buffers_leave_free(tmp_path):
    spirv = _translate(tmp_path, TEXTURE_THEN_BUFFER)
    bindings = _bindings(spirv)
    assert bindings["destination"] == 0
    assert bindings["weights"] == 1
    assert {bindings["source"], bindings["mask"]} == {2, 3}
    _validate(spirv, tmp_path)


def test_each_kernel_of_a_library_keeps_its_buffer_bindings(tmp_path):
    spirv = _translate(tmp_path, LIBRARY)
    bindings = _bindings(spirv)
    assert bindings["destination"] == 0
    assert bindings["values"] == 0
    assert bindings["source"] != 0
    _validate(spirv, tmp_path)


def test_two_buffers_with_one_binding_are_still_refused(tmp_path):
    source = TEXTURE_THEN_BUFFER.replace("[[buffer(1)]]", "[[buffer(0)]]")
    with pytest.raises(Exception, match="Duplicate SPIR-V resource binding"):
        _translate(tmp_path, source)


def test_image_helper_parameters_take_no_memory_decorations(tmp_path):
    spirv = _translate(tmp_path, IMAGE_HELPERS)
    decorated = dict(re.findall(r"OpDecorate (%\w+) (NonWritable|NonReadable)", spirv))
    variables = set(re.findall(r"(%\w+) = OpVariable ", spirv))
    parameters = set(re.findall(r"(%\w+) = OpFunctionParameter ", spirv))
    assert sorted(decorated.values()) == ["NonReadable", "NonWritable"]
    assert set(decorated) <= variables
    assert not set(decorated) & parameters
    _validate(spirv, tmp_path)
