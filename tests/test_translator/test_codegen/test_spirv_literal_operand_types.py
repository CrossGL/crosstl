"""Integer literals inside operators and calls keep their own type in SPIR-V.

The SPIR-V generator types a non-negative ``int`` literal as ``uint`` when the expression it initializes or converts
is unsigned. That context used to reach literals nested in operators and calls, so in ``uint(v % 3)`` the 3 became
``3u`` and the remainder of a negative ``v`` was computed unsigned (GLSL 4.60 section 5.9: an operation on ``int``
operands is signed; the result is converted afterwards). A literal now takes the type of the other operand, or of
the parameter it binds to.
"""

from __future__ import annotations

import re

import pytest

import crosstl

PROGRAM = """
shader Literals {{
    compute probe {{
        layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
        buffer int* data;
        buffer uint* outp;
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            int v = data[i];
            uint b = uint(i);
            uint a = {expression};
            outp[i] = a;
        }}
    }}
}}
"""


def operations(tmp_path, expression):
    path = tmp_path / "probe.cgl"
    path.write_text(PROGRAM.format(expression=expression))
    generated = crosstl.translate(str(path), backend="vulkan", format_output=False)
    return re.findall(
        r"Op(?:[SU]GreaterThan|[SU]Div|[SU]Rem|UMod|SMod)\b|\b[SU](?:Max|Min|Clamp)\b",
        generated,
    )


@pytest.mark.parametrize(
    "expression,signed",
    [
        ("(v > 0) ? 1u : 0u", "OpSGreaterThan"),
        ("(v > -990) ? 1u : 0u", "OpSGreaterThan"),
        ("uint(max(-1, v))", "SMax"),
        ("uint((v > 0) ? 1 : 0)", "OpSGreaterThan"),
        ("uint(v % 3)", "OpSRem"),
        ("uint(v / 2)", "OpSDiv"),
        ("uint(max(0, v))", "SMax"),
        ("uint(clamp(v, 0, 9))", "SClamp"),
    ],
)
def test_signed_operations_stay_signed_in_an_unsigned_context(
    tmp_path, expression, signed
):
    found = operations(tmp_path, expression)
    assert signed in found, found
    assert not any(name.startswith(("OpU", "U")) for name in found), found


@pytest.mark.parametrize(
    "expression,unsigned",
    [("b % 3", "OpUMod"), ("max(b, 1u)", "UMax"), ("min(b, 7)", "UMin")],
)
def test_a_literal_still_takes_its_unsigned_operands_type(
    tmp_path, expression, unsigned
):
    assert unsigned in operations(tmp_path, expression)
