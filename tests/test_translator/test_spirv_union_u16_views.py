"""SPIR-V, WGSL and Slang output of a union with a 16-bit lane view (``union { uint2 val; ushort2 halves[2]; }``).

The storage lowering (:mod:`crosstl.translator.codegen.spirv_union_storage`) keeps one ``uvec2`` and reads each
``u16vec2`` of the lane view as the two 16-bit lanes of one storage word, least significant first, with no ``Int16``
capability. A write of a whole ``u16vec2`` packs it into the word, and a write of one lane, including the Metal
importer's ``CrossGLMetalVectorIndex_u16vec2_set`` helper, replaces one lane. Narrow-integer legalization keeps the
union's members at their declared width, so the word count is the view's, not that of a widened ``ivec2[2]``.
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.spirv_union_storage import UnionLayoutError

REQUIRE_ENV = "CROSSTL_REQUIRE_UNION_U16_EXECUTION"
TARGETS = ("vulkan", "wgsl", "slang")
INVOCATIONS = 8
WORDS = 6

# The shape of MLX's random.metal: a union written through its words and read through its 16-bit lanes.
METAL = """#include <metal_stdlib>
using namespace metal;

union rbits {
  uint2 val;
  ushort2 halves[2];
};

rbits make(uint2 x) {
  rbits v;
  v.val.x = x.x;
  v.val.y = x.y;
  return v;
}

kernel void k(device uint* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  rbits v = make(uint2(0x04030201u * (i + 1u), 0x80706050u + i));
  uint acc = 0u;
  for (int j = 0; j < 2; ++j) {
    acc = acc * 7u + uint(v.halves[1][j]);
  }
  v.halves[1][0] = ushort(7);
  v.halves[0][i & 1u] = ushort(200u + i);
  ushort2 c = v.halves[0];
  out[i * 6u + 0u] = v.val.x;
  out[i * 6u + 1u] = v.val.y;
  out[i * 6u + 2u] = acc;
  out[i * 6u + 3u] = uint(c.x) + uint(c.y) * 65536u;
  v.halves[1] = ushort2(ushort(i), 250);
  out[i * 6u + 4u] = v.val.y;
  v.halves[0] = c.yx;
  out[i * 6u + 5u] = v.val.x;
}
"""


def expected_words():
    """What the Metal kernel stores, one little-endian union per invocation."""

    def word(lanes):
        return lanes[0] | (lanes[1] << 16)

    words = []
    for i in range(INVOCATIONS):
        first = (0x04030201 * (i + 1)) & 0xFFFFFFFF
        second = (0x80706050 + i) & 0xFFFFFFFF
        lanes = [
            [first & 0xFFFF, (first >> 16) & 0xFFFF],
            [second & 0xFFFF, (second >> 16) & 0xFFFF],
        ]
        acc = 0
        for value in lanes[1]:
            acc = (acc * 7 + value) & 0xFFFFFFFF
        lanes[1][0] = 7
        lanes[0][i & 1] = (200 + i) & 0xFFFF
        copy = list(lanes[0])
        row = [word(lanes[0]), word(lanes[1]), acc, copy[0] + copy[1] * 65536]
        lanes[1] = [i & 0xFFFF, 250]
        row.append(word(lanes[1]))
        row.append(word(copy[::-1]))
        words += row
    return words


def translate(tmp_path, source, backend, suffix=".metal"):
    path = tmp_path / f"union{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def run_tool(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def spirv_binary(tmp_path, code):
    """Assemble and validate SPIR-V assembly; skip when SPIRV-Tools is missing."""
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools is not installed")
    source, binary = tmp_path / "union.spvasm", tmp_path / "union.spv"
    source.write_text(code, encoding="utf-8")
    run_tool(["spirv-as", "--target-env", "vulkan1.1", str(source), "-o", str(binary)])
    run_tool(["spirv-val", "--target-env", "vulkan1.1", str(binary)])
    return binary


def wgsl_binary(tmp_path, code):
    """Validate WGSL with naga and convert it to SPIR-V; skip when naga is missing."""
    naga = shutil.which("naga")
    if naga is None:
        pytest.skip("naga is not installed")
    source, binary = tmp_path / "union.wgsl", tmp_path / "union-wgsl.spv"
    source.write_text(code, encoding="utf-8")
    run_tool([naga, str(source), str(binary)])
    return binary


def slang_binary(tmp_path, code):
    """Compile Slang to SPIR-V with slangc and validate it; skip when slangc is missing."""
    slangc = shutil.which("slangc")
    if slangc is None:
        pytest.skip("slangc is not installed")
    source, binary = tmp_path / "union.slang", tmp_path / "union-slang.spv"
    source.write_text(code, encoding="utf-8")
    run_tool([slangc, str(source), "-target", "spirv", "-o", str(binary)])
    if shutil.which("spirv-val"):
        run_tool(["spirv-val", "--target-env", "vulkan1.2", str(binary)])
    return binary


BINARIES = {"vulkan": spirv_binary, "wgsl": wgsl_binary, "slang": slang_binary}


def vulkan_runtime():
    """A host Vulkan runtime and ``vkrun``, or ``None``: lavapipe, else MoltenVK or KosmicKrisp."""
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


@pytest.mark.parametrize("backend", TARGETS)
def test_u16_view_lowers_to_the_storage_words(backend, tmp_path):
    code = translate(tmp_path, METAL, backend)
    assert "CrossGLUnionStorage" in code
    assert "Int16" not in code
    if backend == "vulkan":
        members = re.findall(r'OpMemberName %\w+ \d+ "(\w+)"', code)
        assert "CrossGLUnionStorage" in members
        assert not {"val", "halves"} & set(members)
    else:
        # Each u16vec2 is the two 16-bit lanes of one word, from the least significant up.
        assert "65535u" in code and "16u" in code
        assert ".halves" not in code and ".val" not in code
    BINARIES[backend](tmp_path, code)


@pytest.mark.parametrize("backend", TARGETS)
def test_u16_view_computes_what_the_metal_kernel_computes(backend, tmp_path):
    env = vulkan_runtime()
    if env is None:
        if os.environ.get(REQUIRE_ENV) == "1":
            pytest.fail(
                f"{REQUIRE_ENV}=1 but no host Vulkan runtime with vkrun is available"
            )
        pytest.skip(f"no host Vulkan runtime; set {REQUIRE_ENV}=1 to require one")
    binary = BINARIES[backend](tmp_path, translate(tmp_path, METAL, backend))
    disassembled = run_tool(["spirv-dis", str(binary)]).stdout
    entry = re.search(r'OpEntryPoint GLCompute %\w+ "(\w+)"', disassembled).group(1)
    result_path = tmp_path / "result.bin"
    result_path.write_bytes(bytes(4 * WORDS * INVOCATIONS))
    command = [
        env["CROSSTL_CONFORMANCE_VKRUN"],
        str(binary),
        entry,
        str(INVOCATIONS),
        "1",
        "1",
        f"0:inout:{result_path}",
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=120, env={**os.environ, **env}
    )
    assert result.returncode == 0, result.stdout + result.stderr
    count = WORDS * INVOCATIONS
    assert (
        list(struct.unpack(f"<{count}I", result_path.read_bytes())) == expected_words()
    )


def crossgl(body, *, helpers="", lanes="u16vec2"):
    return f"""shader s {{
    @union_layout(8, 8, little_endian, metal)
    struct Bits {{
        @union_member_layout(0, 8, 8) uvec2 words;
        @union_member_layout(0, 8, 4) {lanes}[2] halves;
    }}
    {helpers}
    compute {{
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer uint* outp;
        void main() {{
            uint idx = gl_GlobalInvocationID.x;
            Bits b;
            b.words = uvec2(idx, idx + 1u);
            {body}
            outp[idx] = b.words.y;
        }}
    }}
}}
"""


LANE_WRITES = {
    # A constant lane shifts by a constant; a dynamic one selects at most lane 1, as the importer's helper does.
    "wgsl": (
        "b.CrossGLUnionStorage[1] = ((b.CrossGLUnionStorage[1] & ~(65535u << 16u)) | "
        "((u32((i32(idx) & 65535)) & 65535u) << 16u));",
        "b.CrossGLUnionStorage[0] = ((b.CrossGLUnionStorage[0] & ~(65535u << (16u * min(u32((idx & 1u)), 1u)))) | "
        "((u32(5) & 65535u) << (16u * min(u32((idx & 1u)), 1u))));",
    ),
    "slang": (
        "b.CrossGLUnionStorage[1] = b.CrossGLUnionStorage[1] & ~(65535u << 16u) | "
        "(uint(uint16_t(idx)) & 65535u) << 16u;",
        "b.CrossGLUnionStorage[0] = b.CrossGLUnionStorage[0] & ~(65535u << 16u * min(uint(idx & 1u), 1u)) | "
        "(uint(uint16_t(5)) & 65535u) << 16u * min(uint(idx & 1u), 1u);",
    ),
    "vulkan": ("OpShiftLeftLogical", "OpNot", "UMin"),
}


@pytest.mark.parametrize("backend", TARGETS)
def test_lane_writes_replace_one_lane_of_the_word(backend, tmp_path):
    source = crossgl("b.halves[1].y = uint16(idx); b.halves[0][idx & 1u] = uint16(5);")
    code = translate(tmp_path, source, backend, suffix=".cgl")
    for expected in LANE_WRITES[backend]:
        assert expected in code, expected
    BINARIES[backend](tmp_path, code)


REFUSALS = {
    "lane-view-write-unsupported": crossgl(
        "b.halves[0] = next_halves(idx);",
        helpers="u16vec2 next_halves(uint i) { return u16vec2(1, uint16(i)); }",
    ),
    "lane-write-unsupported": crossgl(
        "b.halves[1][pick(idx)] = uint16(9);",
        helpers="uint pick(uint i) { return i & 1u; }",
    ),
    "interpreted-assignment-value-unsupported": crossgl(
        "uint16 r = (b.halves[1].y = uint16(3)); outp[idx + 1u] = uint(r);"
    ),
    "interpreted-subcomponent-write-unsupported": crossgl(
        "b.halves[1].xy = u16vec2(1, 2);"
    ),
    "interpreted-update-unsupported": crossgl("b.halves[1].x++;"),
    "whole-array-view-unsupported": crossgl(
        "u16vec2 copy[2] = b.halves; outp[idx + 1u] = uint(copy[0].x);"
    ),
    # Sign extension of a signed lane is not lowered, so a signed 16-bit lane view stays refused.
    "member-shape-unsupported": crossgl(
        "outp[idx + 1u] = uint(b.halves[1].x);", lanes="i16vec2"
    ),
}


@pytest.mark.parametrize("backend", TARGETS)
@pytest.mark.parametrize("reason", sorted(REFUSALS))
def test_u16_views_without_an_exact_lowering_are_refused(backend, reason, tmp_path):
    with pytest.raises(UnionLayoutError) as caught:
        translate(tmp_path, REFUSALS[reason], backend, suffix=".cgl")
    error = caught.value
    assert error.reason == reason
    assert error.union_name == "Bits"
    assert error.project_diagnostic_code == (
        f"project.translate.{backend}-union-layout-unsupported"
    )


def test_a_compound_lane_write_is_refused(tmp_path):
    # The read-modify-write of one lane would need the lane's old value; narrow legalization leaves it compound.
    with pytest.raises(UnionLayoutError) as caught:
        translate(tmp_path, crossgl("b.halves[1].x += uint16(1);"), "vulkan", ".cgl")
    assert caught.value.reason == "lane-write-unsupported"


def test_a_u16vec4_view_stays_refused(tmp_path):
    # A u16vec4 spans two words, so it is not the one-word lane view this lowering covers.
    with pytest.raises(UnionLayoutError) as caught:
        translate(
            tmp_path,
            crossgl("outp[idx + 1u] = uint(b.halves[1].x);", lanes="u16vec4"),
            "vulkan",
            ".cgl",
        )
    assert caught.value.reason == "member-shape-unsupported"
