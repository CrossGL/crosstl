"""HLSL's and Slang's ``globallycoherent`` buffers keep their device-wide coherence on every target.

The Slang importer dropped ``globallycoherent``, and the Metal generator ignored it on the HLSL importer's output,
so a UAV coherent across the device became an ordinary buffer. It now becomes ``@globallycoherent`` in CrossGL,
``coherent(device)`` in Metal, ``coherent`` in GLSL and ``Coherent`` in SPIR-V.
"""

import re
import shutil
import subprocess

import pytest

import crosstl

SOURCE = """globallycoherent RWStructuredBuffer<float> outp : register(u1);
RWStructuredBuffer<float> fin : register(u0);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    outp[id.x] = fin[id.x] * 2.0;
}
"""

SLANG = """RWStructuredBuffer<float> fin;
globallycoherent RWStructuredBuffer<float> outp;
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    outp[id.x] = fin[id.x] * 2.0;
}
"""


def _translate(tmp_path, source, suffix, backend):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def _coherent_blocks(spirv):
    names = dict(re.findall(r'OpName (%\w+) "(\w+)"', spirv))
    return {
        names.get(target)
        for target, decoration in re.findall(
            r"Op(?:Member)?Decorate (%\w+)(?: \d+)? (\w+)", spirv
        )
        if decoration == "Coherent"
    }


def test_slang_import_keeps_globallycoherent(tmp_path):
    crossgl = _translate(tmp_path, SLANG, ".slang", "cgl")
    assert re.search(
        r"RWStructuredBuffer<float>\s+outp\b[^;]*@globallycoherent", crossgl
    )
    assert not re.search(
        r"RWStructuredBuffer<float>\s+fin\b[^;]*@globallycoherent", crossgl
    )


@pytest.mark.parametrize(
    "source, suffix", [(SOURCE, ".hlsl"), (SLANG, ".slang")], ids=["hlsl", "slang"]
)
def test_globallycoherent_reaches_every_target(tmp_path, source, suffix):
    metal = _translate(tmp_path, source, suffix, "metal")
    assert re.search(r"coherent\(device\)\s+device\s+float\s*\*\s*outp\b", metal), metal
    assert not re.search(r"coherent\(device\)\s+device\s+float\s*\*\s*fin\b", metal)
    glsl = _translate(tmp_path, source, suffix, "opengl")
    assert re.search(r"\bcoherent\s+buffer\s+outpBuffer\b", glsl), glsl
    spirv = _translate(tmp_path, source, suffix, "vulkan")
    assert _coherent_blocks(spirv) == {"outpBuffer"}
    slang = _translate(tmp_path, source, suffix, "slang")
    assert re.search(r"globallycoherent\s+RWStructuredBuffer<float>\s+outp\b", slang)
    _compile_metal(tmp_path, metal)


def _compile_metal(tmp_path, code):
    if not shutil.which("xcrun"):
        return
    found = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "--version"], capture_output=True
    )
    if found.returncode != 0:
        return
    source, output = tmp_path / "out.metal", tmp_path / "out.ll"
    source.write_text(code)
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-std=metal3.2", "-S", "-emit-llvm"]
        + ["-c", str(source), "-o", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + code
    assert re.search(r"@air\.\w+\.device_coherent\.", output.read_text())
