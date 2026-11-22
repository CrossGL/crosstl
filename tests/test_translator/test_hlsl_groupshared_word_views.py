"""HLSL output packs a groupshared array that a pointer of another scalar width views into 32-bit words.

HLSL cannot alias a groupshared array as another element type, so ``threadgroup half tile[128]`` viewed through
``(threadgroup float*)tile`` becomes one ``groupshared uint`` array: 16- and 8-bit elements are lanes of a word,
written with ``InterlockedAnd`` and ``InterlockedOr`` so neighbouring lanes do not race (#4823).
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess

import pytest

import crosstl
from crosstl.project.directx_toolchain import dxc_compiler_arguments_for_source
from crosstl.translator.codegen.directx_values import DirectXWorkgroupPointerError
from crosstl.translator.codegen.pointer_reinterpret import PointerReinterpretationError

REQUIRE_ENV = "CROSSTL_REQUIRE_HLSL_GROUPSHARED_WORD_VIEW_EXECUTION"

HALF_TILE = """
shader main {
    compute {
        layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
        void main(StructuredBuffer<float16> input @buffer(0), RWStructuredBuffer<float> output @buffer(1),
                  uint lane @gl_LocalInvocationIndex) {
            threadgroup float16[128] tile;
            tile[lane] = buffer_load(input, lane);
            workgroupBarrier();
            float value = float(tile[lane]);
            workgroupBarrier();
            threadgroup float* words = (threadgroup float*)tile;
            if (lane < 64u) {
                words[lane] = value;
            }
            workgroupBarrier();
            buffer_store(output, lane, value + words[lane & 63u]);
        }
    }
}
"""

BYTE_TILE = """
shader main {
    compute {
        layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
        void main(RWStructuredBuffer<uint> output @buffer(0), uint lane @gl_LocalInvocationIndex) {
            threadgroup uint8[256] bytes;
            bytes[lane] = uint8(lane * 3u);
            bytes[lane + 128u] = uint8(lane);
            workgroupBarrier();
            bytes[lane] += uint8(1);
            bytes[lane + 128u]++;
            workgroupBarrier();
            threadgroup uint* quads = (threadgroup uint*)&bytes[4];
            buffer_store(output, lane, quads[lane & 31u]);
        }
    }
}
"""

# One element type, one view type and a body that reads, writes, compound-assigns and steps both.
VIEW_TEMPLATE = """
shader main {
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        void main(RWStructuredBuffer<float> output @buffer(0), uint lane @gl_LocalInvocationIndex) {
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
            buffer_store(output, lane, value + float(view[lane & 7u]));
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


def _translate(tmp_path, source, name="kernel.cgl"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="directx")


def _dxc(tmp_path, hlsl, *arguments, output):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not on PATH")
    source = tmp_path / "kernel.hlsl"
    source.write_text(hlsl, encoding="utf-8")
    flags = dxc_compiler_arguments_for_source(hlsl)
    result = subprocess.run(
        [dxc, "-T", "cs_6_2", "-E", "CSMain", "-WX", *flags, *arguments]
        + [str(source), "-Fo", str(output)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-4000:]
    return output


def _vulkan_runtime():
    """A host Vulkan runtime with ``vkrun``: lavapipe, else MoltenVK or KosmicKrisp."""
    try:
        from tests.conformance.runtimes import provision
    except ImportError:
        return None
    for runtime in ("vulkan-lavapipe", "vulkan-moltenvk", "vulkan-kosmickrisp"):
        try:
            provisioned = provision(runtime)
        except Exception:
            continue
        if provisioned.available and provisioned.mode == "host":
            env = dict(provisioned.env)
            if env.get("CROSSTL_CONFORMANCE_VKRUN"):
                return env
    return None


def _run_on_vulkan(tmp_path, hlsl, buffers):
    """Run ``hlsl`` once with 128 threads; ``buffers`` maps a binding to its initial bytes."""
    binary = _dxc(
        tmp_path,
        hlsl,
        "-spirv",
        "-fspv-target-env=vulkan1.1",
        output=tmp_path / "kernel.spv",
    )
    env = _vulkan_runtime()
    if env is None:
        if os.environ.get(REQUIRE_ENV) == "1":
            pytest.fail(
                f"{REQUIRE_ENV}=1 but no host Vulkan runtime with vkrun is available"
            )
        pytest.skip(f"no host Vulkan runtime; set {REQUIRE_ENV}=1 to require one")
    command = [env["CROSSTL_CONFORMANCE_VKRUN"], str(binary), "CSMain", "1", "1", "1"]
    paths = {}
    for binding, data in sorted(buffers.items()):
        paths[binding] = tmp_path / f"buffer{binding}.bin"
        paths[binding].write_bytes(data)
        command.append(f"{binding}:inout:{paths[binding]}")
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, **env},
    )
    assert result.returncode == 0 and result.stdout.startswith("ok"), (
        result.stdout + result.stderr
    )
    return {binding: path.read_bytes() for binding, path in paths.items()}


def _groupshared_body(hlsl):
    return hlsl[hlsl.index("groupshared") :]


def test_half_tile_viewed_as_float_is_one_word_array(tmp_path):
    hlsl = _translate(tmp_path, HALF_TILE)

    body = _groupshared_body(hlsl)
    assert re.search(r"groupshared uint \w*tile\[64\];", body)
    assert "float16_t main_tile" not in hlsl and "half " not in body
    assert "InterlockedAnd(" in body and "InterlockedOr(" in body
    assert re.search(r"asfloat\(\w*tile\[", body)
    _dxc(tmp_path, hlsl, output=tmp_path / "kernel.dxil")


def test_half_tile_viewed_as_float_computes_the_source_values(tmp_path):
    hlsl = _translate(tmp_path, HALF_TILE)
    halves = [((index * 37) % 101 - 50) * 0.375 for index in range(128)]
    raw = struct.pack("<128e", *halves)
    initial = struct.pack("<128I", *([0xA5A5A5A5] * 128))

    outputs = _run_on_vulkan(tmp_path, hlsl, {0: raw, 1: initial})

    values = struct.unpack("<128e", raw)
    expected = [values[lane] + values[lane & 63] for lane in range(128)]
    assert list(struct.unpack("<128f", outputs[1])) == expected


def test_byte_tile_lanes_keep_their_neighbours_and_view_at_an_offset(tmp_path):
    hlsl = _translate(tmp_path, BYTE_TILE)
    assert re.search(r"groupshared uint \w*bytes\[64\];", _groupshared_body(hlsl))

    outputs = _run_on_vulkan(
        tmp_path, hlsl, {0: struct.pack("<128I", *([0xA5A5A5A5] * 128))}
    )

    data = bytes(
        ((3 * index + 1) if index < 128 else (index - 128 + 1)) & 0xFF
        for index in range(256)
    )
    words = struct.unpack("<64I", data)
    expected = [words[(lane & 31) + 1] for lane in range(128)]
    assert list(struct.unpack("<128I", outputs[0])) == expected


@pytest.mark.parametrize("element, view", VIEW_PAIRS)
def test_views_of_another_width_compile(tmp_path, element, view):
    source = VIEW_TEMPLATE.replace("ELEMENT", element).replace("VIEW", view)
    hlsl = _translate(tmp_path, source)

    body = _groupshared_body(hlsl)
    assert re.search(r"groupshared uint \w*tile\[\d+\];", body)
    assert " view" not in body
    _dxc(tmp_path, hlsl, output=tmp_path / "kernel.dxil")


def test_same_width_views_of_word_arrays_keep_their_declaration(tmp_path):
    source = VIEW_TEMPLATE.replace("ELEMENT", "uint").replace("VIEW", "float")
    hlsl = _translate(tmp_path, source)

    assert "InterlockedAnd(" not in hlsl and "__crossgl_word_index" not in hlsl
    _dxc(tmp_path, hlsl, output=tmp_path / "kernel.dxil")


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
"""


def test_metal_threadgroup_half_tile_reinterpreted_as_float_translates(tmp_path):
    path = tmp_path / "tg_pun.metal"
    path.write_text(METAL_REPRODUCER, encoding="utf-8")

    hlsl = crosstl.translate(str(path), backend="directx")

    assert re.search(r"groupshared uint \w*tile\[64\];", hlsl)
    _dxc(
        tmp_path,
        hlsl,
        *(
            f"-DCROSSGL_WORKGROUP_SIZE_{axis}={size}"
            for axis, size in zip("XYZ", (128, 1, 1))
        ),
        output=tmp_path / "kernel.dxil",
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
    with pytest.raises((PointerReinterpretationError, DirectXWorkgroupPointerError)):
        _translate(tmp_path, source)
