"""Explicit specializations are selected by type, not by how a scalar alias is spelled.

MSL spells each integer scalar two ways (MSL section 2.1, table 2.1): ``uchar`` and ``uint8_t``, ``ushort`` and
``uint16_t``, ``uint`` and ``uint32_t`` name one type each. A call ``bump(x)`` with ``x`` of type ``uchar`` deduces
``T = uchar`` and runs ``template <> uint8_t bump<uint8_t>(uint8_t)``. Targets without templates get one helper per
deduced type, and that helper must carry the specialization's body rather than a clone of the primary's.
"""

import re

import pytest

import crosstl

PREFIX = "#include <metal_stdlib>\nusing namespace metal;\n"

# Specializations spelled with the <stdint.h> names, calls deduced from the short names.
SPECIALIZED_AS_STDINT = """
template <typename T> T bump(T x) { return x + T(1); }
template <> uint8_t bump<uint8_t>(uint8_t x) { return x + 2; }
template <> uint16_t bump<uint16_t>(uint16_t x) { return x + 3; }
template <> uint32_t bump<uint32_t>(uint32_t x) { return x + 5u; }
kernel void k(device uchar* p [[buffer(0)]], device ushort* s [[buffer(1)]],
              device uint* q [[buffer(2)]], uint id [[thread_position_in_grid]]) {
  p[id] = bump(p[id]);
  s[id] = bump(s[id]);
  q[id] = bump(q[id]);
}
"""

# Specializations spelled with the short names, calls deduced from the <stdint.h> names.
SPECIALIZED_AS_SHORT = """
template <typename T> T bump(T x) { return x + T(1); }
template <> uchar bump<uchar>(uchar x) { return x + 2; }
template <> ushort bump<ushort>(ushort x) { return x + 3; }
template <> uint bump<uint>(uint x) { return x + 5u; }
kernel void k(device uint8_t* p [[buffer(0)]], device uint16_t* s [[buffer(1)]],
              device uint32_t* q [[buffer(2)]], uint id [[thread_position_in_grid]]) {
  p[id] = bump(p[id]);
  s[id] = bump(s[id]);
  q[id] = bump(q[id]);
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "kernel.metal"
    path.write_text(PREFIX + source)
    return crosstl.translate(
        str(path), backend=backend, format_output=False, source_backend="metal"
    )


@pytest.mark.parametrize(
    "source", [SPECIALIZED_AS_STDINT, SPECIALIZED_AS_SHORT], ids=["stdint", "short"]
)
@pytest.mark.parametrize("backend", ["directx", "opengl"])
def test_alias_spelled_specializations_keep_their_bodies(tmp_path, source, backend):
    code = _translate(tmp_path, source, backend)

    # Every deduced helper runs its specialization; none clones the primary ``x + T(1)``.
    assert re.search(r"\bx\)? \+ 2\)", code), code
    assert re.search(r"\bx\)? \+ 3\)", code), code
    assert re.search(r"\bx \+ 5u\)", code), code
    assert "(1)" not in code, code


@pytest.mark.parametrize(
    "source", [SPECIALIZED_AS_STDINT, SPECIALIZED_AS_SHORT], ids=["stdint", "short"]
)
def test_alias_spelled_specializations_keep_their_bodies_in_spirv(tmp_path, source):
    code = _translate(tmp_path, source, "vulkan")

    constants = dict(re.findall(r"(%\w+) = OpConstant %\w+ (\S+)", code))
    added = [
        constants.get(operand)
        for operand in re.findall(r"= OpIAdd %\w+ %\w+ (%\w+)", code)
    ]
    assert sorted(added) == ["2", "3", "5"], code
