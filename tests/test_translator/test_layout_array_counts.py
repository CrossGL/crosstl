"""Constant-expression array counts in buffer blocks keep their layout on every target.

A GLSL block member such as ``vec4 position[12 * 3]`` or ``float weights[COUNT + 1]`` is sized by a constant
integral expression (GLSL 4.60 section 4.1.9). The layout planners must size it like the literal count rather
than refuse the block, and must still refuse a count that is not a compile-time constant.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.array_utils import fold_array_type_counts
from crosstl.translator.codegen.buffer_member_layout import packoffset_bytes

SOURCE = """#version 450
const int COUNT = 3;
layout(std140, binding = 0) uniform Params {
    mat4 mvp;
    vec4 position[12 * 3];
    float weights[COUNT + 1];
    float tail;
} params;
layout(location = 0) out vec4 color;
void main() {
    color = params.mvp * params.position[5] + vec4(params.weights[COUNT] + params.tail);
}
"""

# std140 offsets of SOURCE: each float of ``weights`` has a 16-byte stride, so ``tail`` follows the last one at
# 640 + 4 * 16. glslangValidator -l -q reports the same offsets.
OFFSETS = {"mvp": 0, "position": 64, "weights": 640, "tail": 704}


def _translate(tmp_path, backend):
    path = tmp_path / "shader.frag"
    path.write_text(SOURCE, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, source_backend="opengl")


@pytest.mark.parametrize(
    ("type_name", "constants", "expected"),
    (
        ("vec4[(12 * 3)]", None, "vec4[36]"),
        ("float[COUNT + 1]", {"COUNT": 3}, "float[4]"),
        ("float4[2][N * 2]", {"N": 3}, "float4[2][6]"),
        ("float[8 / 3]", None, "float[2]"),
        ("float[16]", None, "float[16]"),
    ),
)
def test_constant_array_counts_fold_to_literals(type_name, constants, expected):
    assert fold_array_type_counts(type_name, constants) == expected


@pytest.mark.parametrize(
    "type_name", ("float[UNKNOWN]", "float[1 - 2]", "float[]", "float[n[0]]")
)
def test_non_constant_array_counts_are_kept(type_name):
    assert fold_array_type_counts(type_name, {"n": 1}) == type_name


def test_glslang_reports_the_std140_offsets(tmp_path):
    validator = shutil.which("glslangValidator")
    if validator is None:
        pytest.skip("glslangValidator is not available")
    path = tmp_path / "reflect.frag"
    path.write_text(SOURCE, encoding="utf-8")
    result = subprocess.run(
        [validator, "-l", "-q", str(path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
    reported = {
        match.group(1): int(match.group(2))
        for match in re.finditer(
            r"^(?:Params\.)?(\w+)(?:\[0\])?: offset (\d+)", result.stdout, re.M
        )
    }
    assert {name: reported[name] for name in OFFSETS} == OFFSETS


def test_opengl_keeps_the_std140_block(tmp_path):
    code = _translate(tmp_path, "opengl")
    assert re.search(r"layout\(std140, binding = 0\) uniform Params\b", code)


def test_hlsl_places_the_tail_after_the_strided_array(tmp_path):
    code = _translate(tmp_path, "directx")
    body = re.search(r"cbuffer Params\b[^{]*\{(.*?)\};", code, re.S)
    assert body is not None, code
    placed = {
        match.group(1): packoffset_bytes(match.group(2))
        for match in re.finditer(
            r"(\w+)(?:\[[^\]]*\])?\s*:\s*packoffset\(([^)]*)\)", body.group(1)
        )
    }
    # HLSL packing would put ``tail`` at 692, in the last register of ``weights``; std140 puts it at 704.
    assert placed == OFFSETS


def test_spirv_decorates_the_std140_offsets(tmp_path):
    code = _translate(tmp_path, "vulkan")
    members = {
        (struct, index): name
        for struct, index, name in re.findall(
            r'OpMemberName %(\w+) (\d+) "(\w+)"', code
        )
    }
    offsets = {}
    for struct, index, offset in re.findall(
        r"OpMemberDecorate %(\w+) (\d+) Offset (\d+)", code
    ):
        name = members.get((struct, index))
        if name in OFFSETS:
            offsets[name] = int(offset)
    assert offsets == OFFSETS
    assert re.search(r"OpDecorate %\w+ ArrayStride 16", code)


def test_metal_places_the_members_at_std140_offsets(tmp_path):
    code = _translate(tmp_path, "metal")
    if not shutil.which("xcrun"):
        pytest.skip("the Metal toolchain is not available")
    probe = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True, text=True
    )
    if probe.returncode != 0:
        pytest.skip("the Metal compiler is not available")
    asserts = "\n".join(
        f'static_assert(__builtin_offsetof(Params, {name}) == {offset}, "{name}");'
        for name, offset in OFFSETS.items()
    )
    path = tmp_path / "offsets.metal"
    path.write_text(f"{code}\n{asserts}\n", encoding="utf-8")
    result = subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(path),
            "-o",
            str(tmp_path / "o.air"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]
