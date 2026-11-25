"""HLSL private-pointer views of ``&local.member`` take the member's type from the local's structure.

Another structure that declares a member of the same name with a different type
(``uint bits`` and ``uint bits[2]``) must not change how ``&box.bits`` is passed:
it is one scalar object, written back through an ``inout`` argument.

Each program stores constants, so DXC folds the stored value and the DXIL
listing shows the result.
"""

import re
import shutil
import subprocess

import pytest

import crosstl

METAL_MEMBER_NAME = """#include <metal_stdlib>
using namespace metal;
struct WordBox { uint bits; uint other; };
struct Pair { uint bits[2]; };
void bump_word(thread uint* word) { word[0] += 1u; }
kernel void k(device uint* results [[buffer(0)]]) {
    uint plain = 1u;
    bump_word(&plain);
    WordBox box; box.bits = 5u; box.other = 9u;
    bump_word(&box.bits);
    Pair pair; pair.bits[0] = 1u; pair.bits[1] = 2u;
    results[0] = plain + box.bits + pair.bits[1];
}
"""

CROSSGL_NESTED_MEMBER_NAME = """shader main {
    struct Inner {
        uint bits;
    }
    struct Outer {
        Inner inner;
        uint tail;
    }
    struct Pair {
        uint[2] bits;
        uint[3] inner;
    }
    void bump_word(thread uint* word) {
        word[0] += 3u;
    }
    compute {
        @ stage_entry
        void main(RWStructuredBuffer<uint> results @buffer(0)) {
            Outer outer;
            outer.inner.bits = 4u;
            outer.tail = 7u;
            bump_word((&outer.inner.bits));
            bump_word((&outer.tail));
            Pair pair;
            pair.bits[0] = 1u;
            pair.bits[1] = 2u;
            pair.inner[2] = 5u;
            buffer_store(results, 0, outer.inner.bits * 100u + outer.tail * 10u + pair.bits[1] + pair.inner[2]);
        }
    }
}
"""


def _hlsl(tmp_path, source, name):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend="directx", format_output=False)


def _stored_uint(tmp_path, code):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not on PATH")
    source = tmp_path / "out.hlsl"
    source.write_text(code)
    listing = tmp_path / "out.ll"
    result = subprocess.run(
        [dxc, "-T", "cs_6_0", "-E", "CSMain", "-WX", str(source), "-Fc", str(listing)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # BufferStore(opcode, uav, coord0, coord1, value0, ...): value0 is the result.
    stored = re.findall(
        r"call void @dx\.op\.(?:raw)?[bB]ufferStore\.i32\(i32 \d+, "
        r"%dx\.types\.Handle %[\w.]+, i32 \d+, i32 \d+, i32 (-?\d+),",
        listing.read_text(),
    )
    assert len(stored) == 1, stored
    return int(stored[0])


def test_member_view_ignores_same_named_array_member_of_another_struct(tmp_path):
    code = _hlsl(tmp_path, METAL_MEMBER_NAME, "member_name.metal")

    assert re.search(r"void bump_word\(inout uint word\)", code), code
    assert "bump_word(box.bits);" in code, code
    assert "bump_word(plain);" in code, code


def test_member_view_with_same_named_array_member_computes_source_value(tmp_path):
    code = _hlsl(tmp_path, METAL_MEMBER_NAME, "member_name.metal")

    # plain 1 -> 2, box.bits 5 -> 6, pair.bits[1] 2.
    assert _stored_uint(tmp_path, code) == 10


def test_nested_member_view_resolves_each_member_through_its_struct(tmp_path):
    code = _hlsl(tmp_path, CROSSGL_NESTED_MEMBER_NAME, "nested.cgl")

    assert "bump_word(outer.inner.bits);" in code, code
    assert "bump_word(outer.tail);" in code, code
    # inner.bits 4 -> 7, tail 7 -> 10, pair.bits[1] 2, pair.inner[2] 5.
    assert _stored_uint(tmp_path, code) == 700 + 100 + 2 + 5
