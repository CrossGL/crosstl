"""Integer atomics on a typed buffer convert their operands to the destination's signedness for HLSL.

DXC declares ``InterlockedAdd(ref int32_only result, in uint value)``, so HLSL accepts an ``int`` destination with a
``uint`` value; the memory operand fixes the operation's signedness, as in GLSL, SPIR-V and Metal. The DirectX
generator raised ``ValueError`` for that mix, so an HLSL round trip of a shader DXC accepts failed (#2608).
"""

import re
import shutil
import subprocess

import pytest

import crosstl

OPERATIONS = {
    "InterlockedAdd": "atomicAdd",
    "InterlockedAnd": "atomicAnd",
    "InterlockedOr": "atomicOr",
    "InterlockedXor": "atomicXor",
    "InterlockedMin": "atomicMin",
    "InterlockedMax": "atomicMax",
    "InterlockedExchange": "atomicExchange",
}


def _shader(destination, value, body):
    return f"""
RWStructuredBuffer<{value}> values : register(u0);
RWStructuredBuffer<{destination}> totals : register(u1);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {{
{body}
}}
"""


def _translate(tmp_path, source, backend="directx"):
    path = tmp_path / "shader.hlsl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def _dxc(tmp_path, code):
    dxc = shutil.which("dxc")
    if dxc is None:
        return
    path = tmp_path / "out.hlsl"
    path.write_text(code, encoding="utf-8")
    entry = re.search(r"\]\s*void\s+(\w+)\s*\(", code).group(1)
    result = subprocess.run(
        [dxc, "-T", "cs_6_0", "-E", entry, str(path), "-Fo", str(tmp_path / "o")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("intrinsic", sorted(OPERATIONS))
@pytest.mark.parametrize("destination, value", [("int", "uint"), ("uint", "int")])
def test_interlocked_operand_takes_the_destination_signedness(
    tmp_path, intrinsic, destination, value
):
    source = _shader(
        destination, value, f"    {intrinsic}(totals[id.x], values[id.x]);"
    )
    code = _translate(tmp_path, source)
    # InterlockedExchange always carries its original-value operand.
    assert re.search(
        rf"{intrinsic}\(totals\[id\.x\], {destination}\(values\[id\.x\]\)(, \w+)?\);",
        code,
    ), code
    _dxc(tmp_path, code)


@pytest.mark.parametrize("destination, value", [("int", "uint"), ("uint", "int")])
def test_compare_exchange_converts_compare_and_replacement(
    tmp_path, destination, value
):
    body = (
        f"    {destination} original;\n"
        "    InterlockedCompareExchange(totals[id.x], values[id.x], values[id.x + 1], original);\n"
        "    totals[id.x + 2] = original;"
    )
    code = _translate(tmp_path, _shader(destination, value, body))
    assert re.search(
        rf"InterlockedCompareExchange\(totals\[id\.x\], {destination}\(values\[id\.x\]\), "
        rf"{destination}\(values\[\(?id\.x \+ 1u?\)?\]\), original\);",
        code,
    ), code
    _dxc(tmp_path, code)


def test_opengl_converts_the_same_operand(tmp_path):
    code = _translate(
        tmp_path,
        _shader("int", "uint", "    InterlockedAdd(totals[id.x], values[id.x]);"),
        "opengl",
    )
    assert re.search(r"atomicAdd\([^;]*,\s*int\([^;]*values[^;]*\)\);", code), code
