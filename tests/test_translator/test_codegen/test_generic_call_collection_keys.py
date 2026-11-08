"""A generic call names the instantiation that specialization collection deduced for it.

``bump(buffer_load(p, id))`` with ``p`` a ``RWStructuredBuffer<uint8>`` deduces ``T = uint8``. HLSL stores
``uint8`` elements in ``uint`` and spells ``uint16`` as ``uint16_t``; emission used to deduce the call again from
those rendered types, so it called the ``uint`` specialization, or a late clone of the primary for ``uint16_t``,
instead of the explicit specializations for ``uint8`` and ``uint16`` (silently wrong results).
"""

import re

import pytest

import crosstl

SOURCE = """
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

# A primary instantiated for one storage type and specialized for another must still keep the two apart.
MIXED_SOURCE = """
shader S {
    generic<T> T bump(T x) { return x + T(1); }
    uint bump_u32(uint x) @explicit_specialization(bump, uint) { return x + 5u; }
    compute {
        @ stage_entry
        void k(RWStructuredBuffer<uint8> p @buffer(0), RWStructuredBuffer<uint> q @buffer(1),
               uint id @gl_GlobalInvocationID) {
            buffer_store(p, id, bump(buffer_load(p, id)));
            buffer_store(q, id, bump(buffer_load(q, id)));
        }
    }
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _callee_for_buffer(code, buffer_name):
    """The ``bump`` helper whose result is stored into ``buffer_name``."""
    for line in code.splitlines():
        if re.match(rf"\s*{buffer_name}\[", line) and "bump" in line:
            return re.search(r"\b(bump\w*)\(", line).group(1)
    raise AssertionError(f"no store into {buffer_name}:\n{code}")


@pytest.mark.parametrize("backend", ["directx", "opengl", "metal"])
def test_storage_widened_calls_select_their_explicit_specializations(tmp_path, backend):
    code = _translate(tmp_path, SOURCE, backend)

    assert _callee_for_buffer(code, "p") == "bump_u8", code
    assert _callee_for_buffer(code, "s") == "bump_u16", code
    assert _callee_for_buffer(code, "q") == "bump_u32", code
    # Every instantiation is an explicit specialization; no clone of the primary is emitted.
    assert set(re.findall(r"\bbump\w*", code)) == {
        "bump_u8",
        "bump_u16",
        "bump_u32",
    }, code


def _spirv_stored_increments(code):
    """Map each stored buffer to the constant its called helper adds, following stores through bitcasts."""
    increments, current, results, aliases, chains, stored = {}, None, {}, {}, {}, {}
    constants = {}
    for line in code.splitlines():
        words = line.split()
        if len(words) >= 3 and words[1] == "=":
            result, op, operands = words[0], words[2], words[3:]
            if op == "OpConstant":
                constants[result] = operands[-1]
            elif op == "OpFunction":
                current = result
            elif op == "OpIAdd" and current not in increments:
                increments[current] = operands[-1]
            elif op == "OpFunctionCall":
                results[result] = operands[1]
            elif op == "OpBitcast":
                aliases[result] = operands[1]
            elif op == "OpAccessChain":
                chains[result] = operands[1]
        elif words[:1] == ["OpStore"] and words[1] in chains:
            value = words[2]
            while value in aliases:
                value = aliases[value]
            if value in results:
                stored[chains[words[1]]] = constants.get(increments.get(results[value]))
    return stored


def test_spirv_calls_select_their_explicit_specializations(tmp_path):
    # SPIR-V used to reject the call as an unspecialized generic helper: emission could not deduce ``T``.
    code = _translate(tmp_path, SOURCE, "vulkan")

    names = dict(re.findall(r'OpName (%\w+) "(\w+)"', code))
    stored = {
        names.get(buffer, buffer): value
        for buffer, value in _spirv_stored_increments(code).items()
    }
    assert stored == {"p": "2", "s": "3", "q": "5"}, code


@pytest.mark.parametrize("backend", ["directx", "opengl", "metal"])
def test_storage_widened_primary_instantiation_stays_apart_from_a_specialization(
    tmp_path, backend
):
    code = _translate(tmp_path, MIXED_SOURCE, backend)

    narrow = _callee_for_buffer(code, "p")
    assert narrow != "bump_u32", code
    assert _callee_for_buffer(code, "q") == "bump_u32", code
    body = re.search(rf"\b{narrow}\([^)]*\)\s*\{{(.*?)\}}", code, re.S).group(1)
    assert re.search(r"\b1\b", body) and not re.search(r"\b5u\b", body), code
