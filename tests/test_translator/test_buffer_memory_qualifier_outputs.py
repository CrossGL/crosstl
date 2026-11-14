"""Buffer memory qualifiers across Metal, GLSL, SPIR-V and HLSL output.

* Metal output dropped ``restrict``, ``volatile`` and ``coherent`` on ``buffer T*`` storage pointers, and ``restrict``
  on structured buffers; it now writes ``volatile device T*``, ``coherent device T*`` and ``device T* __restrict``.
* The Metal importer dropped ``__restrict`` on a buffer argument; it now writes CrossGL's ``restrict``.
* SPIR-V and HLSL output ignored Metal's scoped ``coherent(device)``; it is ``Coherent`` and ``globallycoherent``.
* OpenGL output refused ``volatile`` and ``coherent`` on a ``buffer T*`` pointer, which it declares as a
  shader-storage block, where GLSL has both qualifiers (GLSL 4.60 section 4.10).
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.memory_qualifier_names import memory_qualifier_name

POINTERS = """shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        restrict buffer float* fin;
        volatile buffer float* outp;
        coherent buffer float* co;
        float load(uint i) { return fin[i]; }
        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = load(i) * 2.0;
            co[i] = 1.0;
        }
    }
}
"""

STRUCTURED = """shader main {
    @ restrict
    @ set(0)
    @ binding(0)
    StructuredBuffer<float> fin;
    @ set(0)
    @ binding(1)
    RWStructuredBuffer<float> outp;
    compute {
    @ numthreads(1, 1, 1)
    void main(uvec3 id @ gl_GlobalInvocationID) {
        outp[id.x] = fin[id.x] * 2.0;
    }
    }
}
"""

METAL = """#include <metal_stdlib>
using namespace metal;
kernel void probe(const device float* __restrict fin [[buffer(0)]],
                  coherent(device) device float* co [[buffer(1)]],
                  device float* __restrict__ r2 [[buffer(2)]],
                  uint3 id [[thread_position_in_grid]]) {
    co[id.x] = fin[id.x] * 2.0;
    r2[id.x] = 3.0;
}
"""


def _translate(tmp_path, source, suffix, backend):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def _air(tmp_path, code):
    """The AIR the Metal compiler writes for ``code``, or ``None`` without the compiler."""
    if not shutil.which("xcrun"):
        return None
    found = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "--version"], capture_output=True
    )
    if found.returncode != 0:
        return None
    source, output = tmp_path / "out.metal", tmp_path / "out.ll"
    source.write_text(code)
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-std=metal3.2", "-S", "-emit-llvm"]
        + ["-c", str(source), "-o", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + code
    return output.read_text()


def _memory_decorations(spirv, block):
    ids = re.findall(rf'OpName (%\w+) "{block}"', spirv)
    return {
        decoration
        for target, decoration in re.findall(
            r"Op(?:Member)?Decorate (%\w+)(?: \d+)? (\w+)", spirv
        )
        if target in ids
    }


def test_metal_output_qualifies_buffer_pointers(tmp_path):
    metal = _translate(tmp_path, POINTERS, ".cgl", "metal")
    assert re.search(r"device float\s*\*\s*__restrict fin\s+\[\[buffer\(0\)\]\]", metal)
    assert re.search(r"volatile device float\s*\*\s*outp\s+\[\[buffer\(1\)\]\]", metal)
    assert re.search(r"coherent device float\s*\*\s*co\s+\[\[buffer\(2\)\]\]", metal)
    # The helper that reads the restricted buffer takes it restricted as well.
    assert re.search(
        r"float load\(uint i,\s*device float\s*\*\s*__restrict fin\)", metal
    )
    air = _air(tmp_path, metal)
    if air is not None:
        assert re.search(r"^define [^\n]*\bnoalias\b", air, flags=re.M)
        assert re.search(r"\b(?:load|store) volatile\b", air)
        assert re.search(r"@air\.\w+\.device_coherent\.", air)


def test_metal_output_restricts_a_structured_buffer(tmp_path):
    metal = _translate(tmp_path, STRUCTURED, ".cgl", "metal")
    assert re.search(
        r"const device float\s*\*\s*__restrict fin\s+\[\[buffer\(0\)\]\]", metal
    )
    assert not re.search(r"__restrict\s+outp", metal)
    _air(tmp_path, metal)


def test_metal_import_keeps_restrict(tmp_path):
    crossgl = _translate(tmp_path, METAL, ".metal", "cgl")
    assert re.search(r"restrict\s+StructuredBuffer<float>\s+fin\b", crossgl), crossgl
    assert re.search(r"restrict\s+RWStructuredBuffer<float>\s+r2\b", crossgl), crossgl
    metal = _translate(tmp_path, METAL, ".metal", "metal")
    assert re.search(r"\*\s*__restrict fin\b", metal)
    assert re.search(r"\*\s*__restrict r2\b", metal)
    _air(tmp_path, metal)


def test_metal_scoped_coherence_reaches_spirv_and_hlsl(tmp_path):
    spirv = _translate(tmp_path, METAL, ".metal", "vulkan")
    assert "Coherent" in _memory_decorations(spirv, "coBuffer")
    assert {"Restrict", "NonWritable"} <= _memory_decorations(spirv, "finBuffer")
    assert "Restrict" in _memory_decorations(spirv, "r2Buffer")
    hlsl = _translate(tmp_path, METAL, ".metal", "directx")
    assert re.search(r"globallycoherent\s+RWStructuredBuffer<float>\s+co\b", hlsl), hlsl


def test_opengl_output_declares_qualified_buffer_pointers(tmp_path):
    glsl = _translate(tmp_path, POINTERS, ".cgl", "opengl")
    assert re.search(r"\brestrict\s+buffer\s+finBuffer\b", glsl), glsl
    assert re.search(r"\bvolatile\s+buffer\s+outpBuffer\b", glsl), glsl
    assert re.search(r"\bcoherent\s+buffer\s+coBuffer\b", glsl), glsl
    validator = shutil.which("glslangValidator")
    if validator is not None:
        source = tmp_path / "out.comp"
        source.write_text(glsl)
        result = subprocess.run(
            [validator, str(source)], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stdout + glsl


@pytest.mark.parametrize(
    "qualifier, name",
    [
        ("coherent", "coherent"),
        ("coherent(device)", "coherent"),
        ("Coherent(threadgroup)", "coherent"),
        # Neither SPIR-V nor HLSL has a coherence wider than the device.
        ("coherent(system)", "coherent(system)"),
        ("volatile", "volatile"),
    ],
)
def test_memory_qualifier_name(qualifier, name):
    assert memory_qualifier_name(qualifier) == name
