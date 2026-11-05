"""Arrays sized by a specialization constant inside uniform and storage blocks.

GL_KHR_vulkan_glsl, "Specialization Constants": an array inside a block may be sized with a specialization
constant, "but the block will have a static layout ... the layout will be based on the default size of the array".
Metal lays such a block out from the default, so it is lowered rather than refused, while ``length()`` and every
other value use read the specialized count. ``length()`` of any array sized by program constants becomes
``int(<size>)`` instead of the default's literal.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused
from crosstl.translator.semantics.array_length import ArrayLengthUnresolvedError

UNIFORM_BLOCK = """#version 450
layout (constant_id = 0) const int KERNEL_SIZE = 64;
layout (binding = 1) uniform SampleKernel { float scale; vec4 samples[KERNEL_SIZE]; } kern;
layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 color;
void main() {
    vec4 acc = vec4(0.0);
    for (int i = 0; i < kern.samples.length(); i++) {
        acc += kern.samples[i];
    }
    color = acc * uv.x * kern.scale;
}
"""

STORAGE_BLOCK = """#version 450
layout (constant_id = 0) const int MAX_LEVEL = 5;
layout (binding = 3) buffer Counters {
    uint total;
    uint perLevel[MAX_LEVEL + 1];
} counters;
layout (local_size_x = 16) in;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < uint(counters.perLevel.length())) {
        atomicAdd(counters.perLevel[i], 1u);
    }
    atomicAdd(counters.total, 1u);
}
"""

LOCAL_ARRAY = """#version 450
layout (constant_id = 0) const int COUNT = 4;
layout (local_size_x = 1) in;
layout (binding = 0) buffer Out { float values[]; } outp;
void main() {
    float scratch[COUNT];
    for (int i = 0; i < COUNT; i++) {
        scratch[i] = float(i);
    }
    outp.values[0] = scratch[COUNT - 1];
}
"""

SHADOWED = """#version 450
layout (constant_id = 0) const int COUNT = 4;
layout (binding = 0) uniform Weights { vec4 weights[COUNT]; } w;
layout (location = 0) out vec4 color;
int measure(int COUNT) {
    return w.weights.length() + COUNT;
}
void main() {
    color = vec4(float(w.weights.length()), float(measure(2)), 0.0, 1.0);
}
"""


def _translate(tmp_path, source, backend, name):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _assert_metal_compiles(tmp_path, metal):
    if shutil.which("xcrun") is None:
        return
    path = tmp_path / "out.metal"
    path.write_text(metal, encoding="utf-8")
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(path), "-o", str(path) + ".air"],
        capture_output=True,
        text=True,
    )
    if "unable to find utility" in result.stderr:
        return
    assert result.returncode == 0, result.stderr[-2000:]


def test_uniform_block_member_keeps_the_default_layout_and_the_specialized_length(
    tmp_path,
):
    metal = _translate(tmp_path, UNIFORM_BLOCK, "metal", "kernel.frag")
    assert "[[function_constant(0)]]" in metal, metal
    # The block layout follows the default size; the loop follows the specialized count.
    assert re.search(r"float4 samples\[64\];", metal), metal
    assert "i < int(KERNEL_SIZE)" in metal, metal
    assert ".length()" not in metal, metal
    _assert_metal_compiles(tmp_path, metal)


def test_storage_block_member_lowers_with_the_specialized_length(tmp_path):
    metal = _translate(tmp_path, STORAGE_BLOCK, "metal", "counters.comp")
    assert "int(MAX_LEVEL + 1)" in metal, metal
    # The member after the scalar sits at the default layout's offset.
    assert "counters + (4 + i * 4)" in metal, metal
    _assert_metal_compiles(tmp_path, metal)


def test_local_array_sized_by_a_specialization_constant_is_still_refused(tmp_path):
    with pytest.raises(
        MetalLoweringRefused, match="array sized by specialization constant 'COUNT'"
    ):
        _translate(tmp_path, LOCAL_ARRAY, "metal", "local.comp")


@pytest.mark.parametrize("backend", ["directx", "metal"])
def test_length_of_a_specialization_sized_array_is_not_the_default(tmp_path, backend):
    output = _translate(tmp_path, UNIFORM_BLOCK, backend, "kernel.frag")
    assert "int(KERNEL_SIZE)" in output, output
    assert "< 64" not in output, output


def test_length_is_refused_where_a_parameter_shadows_the_constant(tmp_path):
    # Copying ``COUNT`` into ``measure`` would read the parameter, not the specialization constant.
    with pytest.raises(ArrayLengthUnresolvedError, match="sized by 'COUNT'"):
        _translate(tmp_path, SHADOWED, "directx", "shadowed.frag")


KEYWORD_BLOCK = """#version 450
layout (binding = 0) uniform Kernel { vec4 tint; } k;
layout (location = 0) out vec4 color;
void main() {
    color = k.tint;
}
"""


def test_uniform_block_named_like_an_msl_keyword_gets_a_legal_parameter(tmp_path):
    # The parameter is named after the block, and ``kernel`` is an MSL keyword.
    metal = _translate(tmp_path, KEYWORD_BLOCK, "metal", "keyword.frag")
    assert "constant Kernel& kernel_ [[buffer(0)]]" in metal, metal
    assert "kernel_.tint" in metal, metal
    _assert_metal_compiles(tmp_path, metal)
