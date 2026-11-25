"""A struct method that advances or reassigns a pointer member keeps that change in Metal input (#2501).

Direct pointer-member promotion passes the constructor's pointer argument to each method by value. It must not
apply to a struct whose methods change the member itself, or the change is lost and later methods read the
original pointer.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.Metal.preprocessing.pointer_member_writes import (
    pointer_member_is_written,
)
from tests.metal_compute_dispatch import dispatch_word_buffer

REQUIRE_ENV = "CROSSTL_REQUIRE_POINTER_MEMBER_ADVANCE_EXECUTION"

SOURCE = """#include <metal_stdlib>
using namespace metal;

struct Reader {{
  device int* src;
  Reader(device int* s) : src(s) {{}}
  int at(int i) const {{ return src[i]; }}
  void next() {{ {advance} }}
}};

kernel void read_words(device int* words [[buffer(0)]],
                       uint gid [[thread_position_in_grid]]) {{
  Reader r(words);
  r.next();
  r.next();
  words[0] = r.at(1);
}}
"""

# words[k] = 100 + k; each advance moves the reader by `step` words, so words[0] becomes 100 + 2 * step + 1.
ADVANCES = {
    "compound": ("src += 2;", 2),
    "prefix": ("++src;", 1),
    "postfix": ("src++;", 1),
    "this-arrow": ("this->src += 3;", 3),
    "assignment": ("src = src + 2;", 2),
}


@pytest.mark.parametrize(
    "body, written",
    [
        ("src += 2;", True),
        ("src -= 1;", True),
        ("++src;", True),
        ("--src;", True),
        ("src++;", True),
        ("src = other;", True),
        ("this->src += 1;", True),
        ("(*this).src = p;", True),
        ("auto q = &src;", True),
        ("return src[i];", False),
        ("src[i] = 1;", False),
        ("*src = 1;", False),
        ("return src == other;", False),
        ("return &src[i];", False),
        ("return a && src;", False),
        ("other.src += 1;", False),
        ("p->src = q;", False),
        ("return src(i);", False),
        ("// src += 1;\nreturn src[0];", False),
        ("return srcs[0] + my_src;", False),
    ],
)
def test_pointer_member_writes_are_found(body, written):
    assert pointer_member_is_written(body, {"src"}) is written


def _round_trip(tmp_path, advance):
    source = tmp_path / "reader.metal"
    source.write_text(SOURCE.format(advance=advance), encoding="utf-8")
    crossgl = tmp_path / "reader.cgl"
    crossgl.write_text(
        crosstl.translate(str(source), backend="cgl", source_backend="metal"),
        encoding="utf-8",
    )
    return crosstl.translate(str(crossgl), backend="metal", format_output=False)


@pytest.mark.parametrize("advance", sorted(ADVANCES))
def test_advanced_pointer_member_is_kept_in_the_struct(tmp_path, advance):
    metal = _round_trip(tmp_path, ADVANCES[advance][0])

    # The method takes the struct, which holds the pointer, by reference, and no by-value copy of the pointer.
    assert re.search(r"Reader__next\(thread Reader\s*&\s*self\)", metal), metal


@pytest.mark.parametrize("advance", sorted(ADVANCES))
def test_advanced_pointer_member_computes_like_the_source_on_metal(tmp_path, advance):
    body, step = ADVANCES[advance]
    words = [100 + index for index in range(16)]
    expected = 100 + 2 * step + 1
    (tmp_path / "native").mkdir()
    (tmp_path / "translated").mkdir()
    native = dispatch_word_buffer(
        SOURCE.format(advance=body), tmp_path / "native", words, REQUIRE_ENV
    )
    assert native[0] == expected

    translated = dispatch_word_buffer(
        _round_trip(tmp_path, body), tmp_path / "translated", words, REQUIRE_ENV
    )
    assert translated == native


def test_advanced_pointer_member_moves_the_reference_in_hlsl_and_spirv(tmp_path):
    source = tmp_path / "reader.metal"
    source.write_text(SOURCE.format(advance="src += 2;"), encoding="utf-8")

    # HLSL and SPIR-V output carry the pointer member as a buffer reference with an offset; the advance moves the
    # offset of the struct the method receives by reference, so the next call reads from the advanced position.
    hlsl = crosstl.translate(str(source), backend="directx", format_output=False)
    next_body = re.search(
        r"void Reader__next\(inout Reader self[^)]*\)\s*\{(?P<body>[^}]*)\}", hlsl
    )
    assert next_body is not None, hlsl
    assert re.search(r"self\.src\.offset\s*\+=", next_body.group("body")), hlsl

    spirv = crosstl.translate(str(source), backend="vulkan")
    assert "OpEntryPoint GLCompute" in spirv
    if shutil.which("spirv-as") is None or shutil.which("spirv-val") is None:
        pytest.skip("spirv-tools is not installed")
    (tmp_path / "reader.spvasm").write_text(spirv, encoding="utf-8")
    assembled = subprocess.run(
        ["spirv-as", "--target-env", "vulkan1.0", "reader.spvasm", "-o", "reader.spv"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=120,
    )
    assert assembled.returncode == 0, assembled.stderr[-2000:]
    validated = subprocess.run(
        ["spirv-val", "--target-env", "vulkan1.0", "reader.spv"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=120,
    )
    assert validated.returncode == 0, validated.stderr[-2000:]
