"""Early fragment tests ([[early_fragment_tests]], [earlydepthstencil]) reach every target."""

import re
import shutil
import subprocess

import pytest

import crosstl

METAL = """#include <metal_stdlib>
using namespace metal;
struct VOut { float4 position [[position]]; float4 color; };
[[early_fragment_tests]] fragment float4 f(VOut in [[stage_in]]) { return in.color; }
"""
HLSL = """struct VSOutput { float4 Pos : SV_POSITION; };
RWStructuredBuffer<uint> counter : register(u1);
[earlydepthstencil]
void main(VSOutput input)
{
    uint previous;
    InterlockedAdd(counter[0], 1, previous);
}
"""
EXPECTED = {
    "opengl": r"layout\(early_fragment_tests\) in;",
    "directx": r"\[earlydepthstencil\]\s*(?:float4|void) PSMain",
    "vulkan": r"OpExecutionMode %\w+ EarlyFragmentTests\n",
    "metal": r"\[\[early_fragment_tests\]\] fragment ",
}


def translate(tmp_path, source, suffix, backend, source_backend):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(
        str(path), backend=backend, source_backend=source_backend, format_output=False
    )


@pytest.mark.parametrize("backend", sorted(EXPECTED))
@pytest.mark.parametrize(
    "source, suffix, source_backend",
    [(METAL, ".metal", "metal"), (HLSL, ".frag.hlsl", "directx")],
    ids=["metal", "hlsl"],
)
def test_early_fragment_tests_are_kept(
    tmp_path, source, suffix, source_backend, backend
):
    # Only GLSL kept them on main; HLSL, SPIR-V and Metal dropped them, and the HLSL import read the attribute as
    # a semantic of a function it did not take for a pixel shader.
    output = translate(tmp_path, source, suffix, backend, source_backend)
    assert re.search(EXPECTED[backend], output), output
    if backend == "directx" and shutil.which("dxc"):
        (tmp_path / "out.hlsl").write_text(output)
        result = subprocess.run(
            ["dxc", "-T", "ps_6_0", "-E", "PSMain", "out.hlsl"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
