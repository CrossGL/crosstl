"""HLSL output keeps the initializers of shader-scope variables and assigns brace lists to structures.

A CrossGL program-scope variable with an initializer starts from that value, as SPIR-V ``Private`` and WGSL
``var<private>`` output declare it. HLSL output used to write it as a ``$Globals`` member without the initializer,
so ``W G = {float[2](1.0, 2.0), 3.0};`` read ``G.m[1]`` as whatever the host bound. A stage constant written as a
brace list became an assignment at the start of the entry point, which passed ``{1.0, 2.0}`` to the structure's
constructor helper, and DXC rejected it (#3714).
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.DirectX.DirectxCrossGLCodeGen import HLSLToCrossGLConverter
from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
from crosstl.backend.DirectX.DirectxParser import HLSLParser

PROGRAM = """shader Probe {
    buffer float* outp;
    struct W { float m[2]; float s; };
    float g = 2.0;
    float A[2] = float[2](1.0, 2.0);
    W G = {float[2](1.0, 2.0), 3.0};
    W H = W(float[2](4.0, 5.0), 6.0);
    compute {
        const W S = {float[2](1.0, 2.0), 3.0};
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = g + A[1] + G.m[1] + H.m[0] + S.m[1] + S.s;
        }
    }
}
"""

# t = {{x, 2.0}, x}, then t = {{x, x}, 2.0}: the second list reads t before it is overwritten.
ASSIGNED = """shader Probe {
    buffer float* outp;
    struct W { float m[2]; float s; };
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            float x = outp[i];
            W t;
            t = {{x, 2.0}, x};
            t = {{t.s, t.m[0]}, t.m[1]};
            outp[i] = t.m[0] + t.m[1] * 10.0 + t.s * 100.0;
        }
    }
}
"""

UNINITIALIZED = """shader Probe {
    buffer float* outp;
    float t;
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            outp[gl_GlobalInvocationID.x] = t;
        }
    }
}
"""

EXTERNAL_GLOBAL = """RWStructuredBuffer<float> outp : register(u0);
float g = 2.0;
const float2 k[2] = {12.0, 13.0, 14.0, 15.0};
static const float s = 4.0;
[numthreads(1, 1, 1)] void main(uint3 id : SV_DispatchThreadID) { outp[id.x] = g + k[1].y + s; }
"""


def _hlsl(tmp_path, program, name="probe.cgl"):
    path = tmp_path / name
    path.write_text(program)
    return crosstl.translate(str(path), backend="directx")


def _dxil(tmp_path, code):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not on PATH")
    source = tmp_path / "translated.hlsl"
    source.write_text(code)
    listing = tmp_path / "translated.ll"
    result = subprocess.run(
        [dxc, "-T", "cs_6_6", "-E", "CSMain", "-WX", str(source), "-Fc", str(listing)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    return listing.read_text()


def test_initialized_program_scope_variables_are_static_with_their_initializer(
    tmp_path,
):
    code = _hlsl(tmp_path, PROGRAM)

    assert "static float g = 2.0;" in code
    assert "static float A[2] = {1.0, 2.0};" in code
    assert "static W G = {{1.0, 2.0}, 3.0};" in code
    assert re.search(r"static W H = .*4\.0.*5\.0.*6\.0", code)
    assert not re.search(r"^(float g|W G|W H|float A\[2\]);$", code, re.MULTILINE)


def test_a_stage_constant_brace_list_initializes_a_temporary(tmp_path):
    code = _hlsl(tmp_path, PROGRAM)

    assert "__crossgl_construct_W({" not in code
    assert re.search(r"S = __crossgl_struct_list_\d+;", code)


def test_initializers_reach_dxil(tmp_path):
    # g + A[1] + G.m[1] + H.m[0] + S.m[1] + S.s = 2 + 2 + 2 + 4 + 2 + 3.
    listing = _dxil(tmp_path, _hlsl(tmp_path, PROGRAM))

    assert re.search(r"rawBufferStore\.f32\(.*float 1\.500000e\+01", listing)
    assert "$Globals" not in listing


def test_a_brace_list_assignment_reads_the_old_structure(tmp_path):
    code = _hlsl(tmp_path, ASSIGNED)
    listing = _dxil(tmp_path, code)

    assert "__crossgl_construct_W({" not in code
    # 11 x + 200: the second list swaps the members of the first.
    assert re.search(r"fmul fast float %\d+, 1\.100000e\+01", listing)
    assert re.search(r"fadd fast float %\d+, 2\.000000e\+02", listing)


def test_an_uninitialized_program_scope_variable_stays_a_globals_member(tmp_path):
    code = _hlsl(tmp_path, UNINITIALIZED)

    assert re.search(r"^float t;$", code, re.MULTILINE)


def test_hlsl_input_reads_an_initialized_external_global_from_globals(tmp_path):
    # DXC: "Initializer of external global will be ignored".
    tokens = HLSLLexer(EXTERNAL_GLOBAL).tokenize()
    crossgl = HLSLToCrossGLConverter().generate(HLSLParser(tokens).parse())

    assert re.search(r"@ hlsl_program_constant\s+float g;", crossgl)
    assert "2.0" not in crossgl
    # A ``const`` global without ``static`` is a $Globals member too, and DXC ignores its initializer as well.
    assert re.search(r"@ hlsl_program_constant\s+vec2 k\[2\];", crossgl)
    assert "15.0" not in crossgl
    assert re.search(r"const float s = 4\.0;", crossgl)

    hlsl = _hlsl(tmp_path, EXTERNAL_GLOBAL, "external.hlsl")
    assert re.search(r"^float g;$", hlsl, re.MULTILINE)
    assert re.search(r"^float2 k\[2\];$", hlsl, re.MULTILINE)
    assert "static float g" not in hlsl
