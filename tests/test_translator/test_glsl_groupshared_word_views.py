"""GLSL output packs a shared array that a pointer of another scalar width views into 32-bit words.

GLSL cannot alias a ``shared`` array as another element type, so ``threadgroup half tile[128]`` viewed through
``(threadgroup float*)tile`` becomes one ``shared uint`` array: 16- and 8-bit elements are lanes of a word,
written with ``atomicAnd`` and ``atomicOr`` so neighbouring lanes do not race, and read with
``unpackHalf2x16`` for ``half`` or ``bitfieldExtract`` for integers. The DirectX half of #4823 packs the same
arrays the same way (``hlsl_groupshared_word_views``); this is its GLSL mirror.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.pointer_reinterpret import (
    PointerReinterpretationError,
)

HALF_TILE = """
shader main {
    compute {
        layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
        buffer float* inp;
        buffer float* out_;
        void main() {
            uint lane = gl_LocalInvocationIndex;
            threadgroup half[128] tile;
            tile[lane] = half(inp[lane]);
            workgroupBarrier();
            float value = float(tile[lane]);
            workgroupBarrier();
            threadgroup float* words = (threadgroup float*)tile;
            if (lane < 64u) {
                words[lane] = value;
            }
            workgroupBarrier();
            out_[lane] = value + words[lane & 63u];
        }
    }
}
"""

BYTE_TILE = """
shader main {
    compute {
        layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
        buffer float* out_;
        void main() {
            uint lane = gl_LocalInvocationIndex;
            threadgroup uint8[256] bytes;
            bytes[lane] = uint8(lane * 3u);
            bytes[lane + 128u] = uint8(lane);
            workgroupBarrier();
            bytes[lane] += uint8(1);
            bytes[lane + 128u]++;
            workgroupBarrier();
            threadgroup uint* quads = (threadgroup uint*)&bytes[4];
            out_[lane] = float(quads[lane & 31u]);
        }
    }
}
"""

# One element type, one view type and a body that reads, writes, compound-assigns and steps both.
VIEW_TEMPLATE = """
shader main {
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        buffer float* out_;
        void main() {
            uint lane = gl_LocalInvocationIndex;
            threadgroup ELEMENT[128] tile;
            tile[lane] = ELEMENT(lane);
            tile[lane + 64u] = ELEMENT(1);
            workgroupBarrier();
            tile[lane] += ELEMENT(2);
            tile[lane + 64u]++;
            float value = float(tile[lane]);
            workgroupBarrier();
            threadgroup VIEW* view = (threadgroup VIEW*)tile;
            if (lane < 8u) {
                view[lane] = VIEW(value);
                view[lane + 8u] -= VIEW(1);
            }
            workgroupBarrier();
            out_[lane] = value + float(view[lane & 7u]);
        }
    }
}
"""

VIEW_PAIRS = [
    ("float16", "float"),
    ("float16", "uint"),
    ("float16", "uint16"),
    ("int16", "int"),
    ("uint16", "uint"),
    ("uint8", "uint"),
    ("int8", "int"),
    ("uint8", "float"),
    ("uint", "uint16"),
    ("float", "int16"),
    ("int", "uint8"),
]

ELEMENT_BITS = {
    "float16": 16,
    "int16": 16,
    "uint16": 16,
    "uint8": 8,
    "int8": 8,
    "uint": 32,
    "float": 32,
    "int": 32,
}


def _translate(tmp_path, source, name="kernel.cgl", **kwargs):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="opengl", format_output=False, **kwargs)


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


def test_half_tile_viewed_as_float_is_one_word_array(tmp_path):
    glsl = _translate(tmp_path, HALF_TILE)

    assert "shared uint main_tile[64];" in glsl
    assert "shared float main_tile" not in glsl and "threadgroup" not in glsl
    assert "packHalf2x16(vec2(" in glsl
    assert "unpackHalf2x16(" in glsl
    assert "atomicAnd(" in glsl and "atomicOr(" in glsl
    assert "uintBitsToFloat(" in glsl and "floatBitsToUint(" in glsl
    assert " words" not in glsl
    _glslang(tmp_path, glsl)


def test_byte_tile_lanes_keep_their_neighbours_and_view_at_an_offset(tmp_path):
    glsl = _translate(tmp_path, BYTE_TILE)

    assert "shared uint main_bytes[64];" in glsl
    assert "bitfieldExtract(" in glsl
    assert "atomicAnd(" in glsl and "atomicOr(" in glsl
    assert re.search(r"main_bytes\[\(uint\(\(lane & 31u\)\) \+ 1u\)\]", glsl)
    assert " quads" not in glsl
    _glslang(tmp_path, glsl)


@pytest.mark.parametrize("element, view", VIEW_PAIRS)
def test_views_of_another_width_compile(tmp_path, element, view):
    source = VIEW_TEMPLATE.replace("ELEMENT", element).replace("VIEW", view)
    glsl = _translate(tmp_path, source)

    words = 128 * ELEMENT_BITS[element] // 32
    assert f"shared uint main_tile[{words}];" in glsl
    assert " view" not in glsl
    _glslang(tmp_path, glsl)


@pytest.mark.parametrize(
    "element, view",
    [("uint", "float"), ("int", "float"), ("float", "int")],
)
def test_same_width_views_of_another_type_still_refuse(tmp_path, element, view):
    source = VIEW_TEMPLATE.replace("ELEMENT", element).replace("VIEW", view)
    with pytest.raises(PointerReinterpretationError) as raised:
        _translate(tmp_path, source)

    assert raised.value.reason == "workgroup-view-type-change"


METAL_REPRODUCER = """#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void tg_pun_impl(
    device const T* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint lane [[thread_index_in_threadgroup]]) {
  threadgroup T tile[128];
  tile[lane] = input[lane];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float value = float(tile[lane]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup float* words = reinterpret_cast<threadgroup float*>(tile);
  if (lane < 64) {
    words[lane] = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  output[lane] = value + words[lane & 63];
}

template [[host_name("tg_pun_half")]] [[kernel]] void tg_pun_impl<half>(
    device const half*, device float*, uint);
template [[host_name("tg_pun_float")]] [[kernel]] void tg_pun_impl<float>(
    device const float*, device float*, uint);
"""


def test_metal_threadgroup_half_tile_reinterpreted_as_float_translates(tmp_path):
    glsl = _translate(
        tmp_path, METAL_REPRODUCER, "tg_pun.metal", entry_point="tg_pun_half"
    )

    assert "shared uint tg_pun_half_tile[64];" in glsl
    assert "packHalf2x16(vec2(" in glsl
    assert "unpackHalf2x16(" in glsl
    _glslang(
        tmp_path,
        glsl,
        "CROSSGL_WORKGROUP_SIZE_X 128",
        "CROSSGL_WORKGROUP_SIZE_Y 1",
        "CROSSGL_WORKGROUP_SIZE_Z 1",
    )


ESCAPES = {
    "view passed to a helper": (
        "float first(threadgroup float* p) { return p[0]; }",
        "buffer_store(output, lane, first(view));",
    ),
    "view moved by pointer arithmetic": (
        "",
        "threadgroup float* next = view + 1; buffer_store(output, lane, next[lane & 7u]);",
    ),
    "element address taken": (
        "",
        "threadgroup float16* one = &tile[lane]; buffer_store(output, lane, float(one[0]));",
    ),
}


@pytest.mark.parametrize("helper, use", ESCAPES.values(), ids=list(ESCAPES))
def test_unsupported_uses_still_refuse(tmp_path, helper, use):
    source = f"""
shader main {{
    {helper}
    compute {{
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        void main(RWStructuredBuffer<float> output @buffer(0), uint lane @gl_LocalInvocationIndex) {{
            threadgroup float16[128] tile;
            tile[lane] = float16(lane);
            workgroupBarrier();
            threadgroup float* view = (threadgroup float*)tile;
            {use}
        }}
    }}
}}
"""
    with pytest.raises(PointerReinterpretationError) as raised:
        _translate(tmp_path, source)

    assert raised.value.reason == "workgroup-view-type-change"
