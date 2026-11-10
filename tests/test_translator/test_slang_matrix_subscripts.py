"""Slang output reorders matrix element subscripts that are operators of names.

A CrossGL ``m[c][r]`` is ``m[r][c]`` in HLSL and Slang, so the generator evaluates the two subscripts in the other
order. It refuses that when neither subscript can be evaluated again safely. The Slang generator counted only names,
literals, members and elements as repeatable, so ``m[i % 2u][(i / 4u) % 2u][i % 3u]`` was refused although its
subscripts write nothing. Operators, conditionals and scalar conversions of repeatable operands now count too, as
they do in the HLSL generator; increments still do not.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl

SOURCE = """#version 450
layout(local_size_x = 64) in;
layout(std430, binding = 1) buffer Outp { float o[]; };
layout(std140, binding = 2) uniform Params { mat2x3 m[2]; } params;
void main() {
    uint i = gl_GlobalInvocationID.x;
    o[i] = params.m[i % 2u][(i / 4u) % 2u][i < 3u ? i : uint(float(i) * 0.0)];
}
"""


def _slang(tmp_path, source):
    path = tmp_path / "subscripts.comp"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="slang")


def test_operator_subscripts_are_reordered(tmp_path):
    code = re.sub(r"\s+", " ", _slang(tmp_path, SOURCE))
    assert "m[(i % 2u)][(i < 3u ? i : uint(float(i) * 0.0))][i / 4u % 2u]" in code, code


def test_increment_subscripts_are_still_refused(tmp_path):
    source = SOURCE.replace(
        "params.m[i % 2u][(i / 4u) % 2u][i < 3u ? i : uint(float(i) * 0.0)]",
        "params.m[0][i++ % 2u][i++ % 3u]",
    )
    with pytest.raises(Exception, match="cannot reorder the column and row subscripts"):
        _slang(tmp_path, source)


def test_slangc_compiles_the_reordered_subscripts(tmp_path):
    slangc = shutil.which("slangc")
    if slangc is None:
        pytest.skip("slangc is not installed")
    shader = tmp_path / "subscripts.slang"
    shader.write_text(_slang(tmp_path, SOURCE), encoding="utf-8")
    result = subprocess.run(
        [slangc, str(shader), "-entry", "main", "-stage", "compute", "-target", "spirv"]
        + ["-o", str(tmp_path / "subscripts.spv")],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
