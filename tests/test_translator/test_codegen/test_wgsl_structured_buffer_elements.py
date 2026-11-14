"""WGSL structured-buffer loads and stores carry the element type, and narrow elements are 32-bit words.

- ``buffer_load(b, i)`` and ``b.Load(i)`` had no type in WGSL, so generic deduction from them failed with
  "cannot infer concrete type arguments ... (unknown)".
- ``buffer_store(b, i, v)`` and ``b.Store(i, v)`` wrote ``v`` without converting it to the element type, so an
  ``int`` stored into a ``uint`` buffer produced ``b[i] = (v + 1)`` with an ``i32`` value (invalid WGSL).
- ``RWStructuredBuffer<uint8>`` and Metal ``device uchar*`` were written verbatim as ``array<uint8>``, which no WGSL
  implementation accepts. WGSL has no 8- or 16-bit integers (WGSL section 6.2.4); like SPIR-V, the storage is laid out
  one 32-bit word per element.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics.narrow import NarrowIntegerUnsupportedError

GENERIC_LOADS = """
shader S {
    generic<T> T bump(T x) { return x + T(1); }
    compute {
        @ stage_entry
        void k(RWStructuredBuffer<uint> q @buffer(0), RWStructuredBuffer<float> f @buffer(1),
               uint id @gl_GlobalInvocationID) {
            buffer_store(q, id, bump(buffer_load(q, id)));
            buffer_store(f, id, bump(f.Load(id)));
        }
    }
}
"""

CONVERTED_STORES = """
shader S {
    compute {
        @ stage_entry
        void k(RWStructuredBuffer<uint> q @buffer(0), RWStructuredBuffer<int> r @buffer(1),
               uint id @gl_GlobalInvocationID) {
            int v = int(id) - 3;
            buffer_store(q, id, v + 1);
            r.Store(id, id * 2u);
        }
    }
}
"""

NARROW_BUFFERS = """
shader S {
    compute {
        @ stage_entry
        void k(RWStructuredBuffer<uint8> p @buffer(0), RWStructuredBuffer<int16> s @buffer(1),
               uint id @gl_GlobalInvocationID) {
            uint8 v = buffer_load(p, id);
            buffer_store(p, id, v + 1);
            int16 w = buffer_load(s, id);
            buffer_store(s, id, w - 1);
        }
    }
}
"""

METAL_NARROW_POINTERS = """
#include <metal_stdlib>
using namespace metal;
kernel void k(device uchar* p [[buffer(0)]], device short* s [[buffer(1)]],
              uint id [[thread_position_in_grid]]) {
    p[id] = p[id] + 1;
    s[id] = s[id] - 1;
}
"""

NARROW_SPECIALIZATIONS = """
shader S {
    generic<T> T bump(T x) { return x + T(1); }
    uint8 bump_u8(uint8 x) @explicit_specialization(bump, uint8) { return x + 2; }
    uint16 bump_u16(uint16 x) @explicit_specialization(bump, uint16) { return x + 3; }
    uint bump_u32(uint x) @explicit_specialization(bump, uint) { return x + 5u; }
    compute {
        @ stage_entry
        void k(RWStructuredBuffer<uint8> p @buffer(0), RWStructuredBuffer<uint16> s @buffer(1),
               RWStructuredBuffer<uint> q @buffer(2), uint id @gl_GlobalInvocationID) {
            buffer_store(p, id, bump(buffer_load(p, id)));
            buffer_store(s, id, bump(buffer_load(s, id)));
            buffer_store(q, id, bump(buffer_load(q, id)));
        }
    }
}
"""

NARROW_PRIMARY = """
shader S {
    generic<T> T bump(T x) { return x + T(1); }
    compute {
        @ stage_entry
        void k(RWStructuredBuffer<uint8> p @buffer(0), uint id @gl_GlobalInvocationID) {
            buffer_store(p, id, bump(buffer_load(p, id)));
        }
    }
}
"""


def _wgsl(tmp_path, source, suffix=".cgl"):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    code = crosstl.translate(str(path), backend="wgsl")
    naga = shutil.which("naga")
    if naga is not None:
        module = tmp_path / "out.wgsl"
        module.write_text(code)
        result = subprocess.run(
            [naga, str(module)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr + code
    return code


def _body(code):
    return code[code.index("fn compute_main") :]


def test_generic_calls_deduce_from_structured_buffer_loads(tmp_path):
    code = _wgsl(tmp_path, GENERIC_LOADS)

    assert "fn bump_u32(x: u32) -> u32" in code, code
    assert "fn bump_f32(x: f32) -> f32" in code, code
    body = _body(code)
    assert "q[id] = bump_u32(q[id]);" in body, code
    assert "f[id] = bump_f32(f[id]);" in body, code


def test_stores_convert_values_to_the_element_type(tmp_path):
    body = _body(_wgsl(tmp_path, CONVERTED_STORES))

    assert "q[id] = u32((v + 1));" in body, body
    assert re.search(r"r\[id\] = i32\(\(id \* 2u\)\);", body), body


def test_narrow_buffer_elements_are_32_bit_words(tmp_path):
    code = _wgsl(tmp_path, NARROW_BUFFERS)

    assert "var<storage, read_write> p: array<u32>;" in code, code
    assert "var<storage, read_write> s: array<i32>;" in code, code
    assert "uint8" not in code and "int16" not in code, code
    body = _body(code)
    assert "var v: i32 = (i32(p[id]) & 255);" in body, body
    assert "p[id] = u32((v + 1));" in body, body


def test_metal_narrow_device_pointers_are_32_bit_words(tmp_path):
    code = _wgsl(tmp_path, METAL_NARROW_POINTERS, suffix=".metal")

    assert "var<storage, read_write> p: array<u32>;" in code, code
    assert "var<storage, read_write> s: array<i32>;" in code, code
    assert not re.search(r"\b(uchar|uint8|short|int16)\b", code), code


def test_narrow_loads_select_their_explicit_specializations(tmp_path):
    code = _wgsl(tmp_path, NARROW_SPECIALIZATIONS)

    body = _body(code)
    # The words are deduced as their CrossGL element types, never as u32.
    assert "p[id] = u32(bump_u8(i32(p[id])));" in body, body
    assert "s[id] = u32(bump_u16(i32(s[id])));" in body, body
    assert "q[id] = bump_u32(q[id]);" in body, body
    assert set(re.findall(r"\bbump\w*", code)) == {
        "bump_u8",
        "bump_u16",
        "bump_u32",
    }, code


def test_a_primary_is_not_instantiated_for_a_narrow_type(tmp_path):
    path = tmp_path / "shader.cgl"
    path.write_text(NARROW_PRIMARY)

    with pytest.raises(NarrowIntegerUnsupportedError, match="'bump'.*'uint8'"):
        crosstl.translate(str(path), backend="wgsl")
