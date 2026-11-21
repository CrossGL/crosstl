"""GLSL output of pointer views of a ``shared`` array (#4823).

``threadgroup float* words = (threadgroup float*)tile;`` over ``threadgroup float tile[128]`` views the array with
its own element type, so it is the array itself plus an offset. GLSL output refused every such view with "without a
concrete shared-array backing object" because the pointer binding did not look through the view. A view that
changes the element type stays refused with a typed diagnostic: a GLSL ``shared`` array has one element type and
cannot be aliased.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.pointer_reinterpret import (
    PointerReinterpretationError,
)

SOURCE = """
shader main {
    compute {
        layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
        buffer float* inp;
        buffer float* out_;
        void main() {
            uint lane = gl_LocalInvocationIndex;
            threadgroup %(element)s[128] tile;
            tile[lane] = %(element)s(inp[lane]);
            workgroupBarrier();
            threadgroup float* upper = (threadgroup float*)(tile + 64);
            threadgroup const float* view =
                (threadgroup const float*)((threadgroup float*)&tile[32]);
            float value = 0.0;
            if (lane < 64) { value = upper[lane] + view[lane]; }
            workgroupBarrier();
            out_[lane] = value;
        }
    }
}
"""

METAL_SOURCE = """
#include <metal_stdlib>
using namespace metal;
kernel void tg_pun_float(device const float* input [[buffer(0)]],
                         device float* output [[buffer(1)]],
                         uint lane [[thread_index_in_threadgroup]]) {
  threadgroup float tile[128];
  tile[lane] = input[lane];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float value = tile[lane];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup float* words = reinterpret_cast<threadgroup float*>(tile);
  if (lane < 64) { words[lane] = value; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  output[lane] = value + words[lane & 63];
}
"""


def _translate(tmp_path, source, suffix=".cgl"):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="opengl", format_output=False)


def _glslang(tmp_path, code, *defines):
    if shutil.which("glslangValidator") is None:
        return
    version, body = code.split("\n", 1)
    path = tmp_path / "out.comp"
    path.write_text(
        "\n".join([version, *(f"#define {d}" for d in defines), body]),
        encoding="utf-8",
    )
    result = subprocess.run(
        ["glslangValidator", "-S", "comp", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_same_type_views_index_the_shared_array(tmp_path):
    glsl = _translate(tmp_path, SOURCE % {"element": "float"})

    assert "shared float main_tile[128];" in glsl
    assert "int upper_offset = int(64);" in glsl
    assert "int view_offset = int(32);" in glsl
    assert "main_tile[(upper_offset + int(lane))]" in glsl
    assert "main_tile[(view_offset + int(lane))]" in glsl
    _glslang(tmp_path, glsl)


def test_metal_reinterpret_cast_to_its_own_element_type(tmp_path):
    glsl = _translate(tmp_path, METAL_SOURCE, ".metal")

    assert "int words_offset = int(0);" in glsl
    assert (
        "tg_pun_float_tile[(words_offset + int(gl_LocalInvocationIndex))] = value;" in (
            glsl
        )
    )
    _glslang(
        tmp_path,
        glsl,
        "CROSSGL_WORKGROUP_SIZE_X 128",
        "CROSSGL_WORKGROUP_SIZE_Y 1",
        "CROSSGL_WORKGROUP_SIZE_Z 1",
    )


@pytest.mark.parametrize("element", ["half", "uint", "int"])
def test_views_that_change_the_element_type_are_refused(tmp_path, element):
    with pytest.raises(PointerReinterpretationError) as raised:
        _translate(tmp_path, SOURCE % {"element": element})

    error = raised.value
    assert error.reason == "workgroup-view-type-change"
    assert error.address_space == "workgroup"
    assert error.source_type == element
    assert error.target_type == "float"
    assert "cannot be aliased" in str(error)
