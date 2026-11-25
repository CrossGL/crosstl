"""GLSL output types a private member view through the struct its root has.

``bump(&box.bits)`` passes one member of a private struct value to a ``thread uint*`` parameter, and GLSL output
lowers it to an ``inout uint`` argument of the member lvalue. The private-pointer pre-pass runs before the caller's
local types are recorded, so it must look ``bits`` up in the struct ``box`` is declared with. Looking it up by name
in every struct picked ``uint bits[2]`` of an unrelated struct and refused the call (``missing-view-backing``).
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl

GLSLANG = shutil.which("glslangValidator")

WORD_BOX = "struct WordBox { uint bits; uint other; };"
PAIR = "struct Pair { uint bits[2]; };"

KERNEL = """
void bump_word(thread uint* word) { word[0] += 1u; }
kernel void k(device uint* results [[buffer(0)]]) {
    WordBox box; box.bits = 5u; box.other = 9u;
    bump_word(&box.bits);
    Pair pair; pair.bits[0] = 1u; pair.bits[1] = 2u;
    results[0] = box.bits + pair.bits[1];
}
"""

NESTED_KERNEL = """
struct Outer { WordBox inner; };
void bump_word(thread uint* word) { word[0] += 1u; }
kernel void k(device uint* results [[buffer(0)]]) {
    Outer outer; outer.inner.bits = 5u;
    bump_word(&outer.inner.bits);
    results[0] = outer.inner.bits;
}
"""


def _metal(tmp_path, structs, kernel):
    path = tmp_path / "member_view.metal"
    path.write_text(
        "#include <metal_stdlib>\nusing namespace metal;\n"
        + "\n".join(structs)
        + kernel,
        encoding="utf-8",
    )
    return path


def _assert_glslang_accepts(tmp_path, code):
    if GLSLANG is None:
        return
    shader = tmp_path / "member_view.comp"
    shader.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [GLSLANG, "-S", "comp", str(shader)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "structs",
    ((WORD_BOX, PAIR), (PAIR, WORD_BOX)),
    ids=("word-box-first", "pair-first"),
)
def test_member_view_uses_the_root_struct_not_a_same_named_member(tmp_path, structs):
    code = crosstl.translate(str(_metal(tmp_path, structs, KERNEL)), backend="opengl")

    assert "void bump_word(inout uint word)" in code
    assert "bump_word(box.bits);" in code
    _assert_glslang_accepts(tmp_path, code)


def test_nested_member_view_walks_each_struct(tmp_path):
    code = crosstl.translate(
        str(_metal(tmp_path, (PAIR, WORD_BOX), NESTED_KERNEL)), backend="opengl"
    )

    assert "bump_word(outer.inner.bits);" in code
    _assert_glslang_accepts(tmp_path, code)
