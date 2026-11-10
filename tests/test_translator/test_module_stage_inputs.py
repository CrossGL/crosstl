"""Module-scope ``in`` declarations reach HLSL and MSL entry points as stage inputs.

CrossGL written the GLSL way declares stage inputs at module scope (``in vec3 pos @location(0);``). The HLSL output
declared such an input as a global, which HLSL makes a uniform, and the MSL output as ``constant float3 pos =
float3(0);``, so the stage read a constant instead of its input. The inputs now move into the program's one vertex or
fragment stage with the outputs (module_stage_outputs.py), and the stage-interface lowering passes them in the
entry point's input structure.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics.stage_interface_structs import (
    StageInterfaceUnsupportedError,
)

FRAGMENT = """
shader main {
    in vec3 n @location(1) @flat;
    in vec2 uv @location(0);
    out vec4 a @location(0);
    vec4 shade() {
        return vec4(n * uv.x, 1.0);
    }
    fragment {
        void main() {
            a = shade();
        }
    }
}
"""

VERTEX = """
shader main {
    in vec3 pos @location(0);
    out vec2 uv @location(1);
    vertex {
        void main() {
            uv = pos.xy;
            gl_Position = vec4(pos, 1.0);
        }
    }
}
"""

TWO_STAGES = """
shader main {
    in vec3 pos @location(0);
    vertex {
        void main() {
            gl_Position = vec4(pos, 1.0);
        }
    }
    fragment {
        vec4 main() @location(0) {
            return vec4(1.0);
        }
    }
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _flat(code):
    return re.sub(r"\s+", " ", code)


def test_hlsl_fragment_reads_module_inputs_from_its_input_structure(tmp_path):
    code = _flat(_translate(tmp_path, FRAGMENT, "directx"))
    assert re.search(r"nointerpolation float3 n\s*:\s*TEXCOORD1;", code), code
    assert re.search(r"float2 uv\s*:\s*TEXCOORD0;", code), code
    assert "float3 n;" not in code and "float2 uv;" not in code, code
    # A helper that reads an input takes the input structure.
    assert "float4 shade(FragmentInput input)" in code, code
    assert "input.n * input.uv.x" in code, code
    assert re.search(r"FragmentOutput PSMain\(FragmentInput input\)", code), code


def test_msl_fragment_reads_module_inputs_from_stage_in(tmp_path):
    code = _flat(_translate(tmp_path, FRAGMENT, "metal"))
    assert "float3 n [[user(locn1)]] [[flat]];" in code, code
    assert "constant float3 n" not in code and "constant float2 uv" not in code, code
    assert "fragment_main(FragmentInput input [[stage_in]])" in code, code


@pytest.mark.parametrize(
    "backend, attribute",
    [
        ("directx", r"float3 pos\s*:\s*TEXCOORD0;"),
        ("metal", r"float3 pos \[\[attribute\(0\)\]\];"),
    ],
)
def test_vertex_module_inputs_are_vertex_attributes(tmp_path, backend, attribute):
    code = _flat(_translate(tmp_path, VERTEX, backend))
    assert re.search(attribute, code), code
    assert "input.pos.xy" in code, code


def test_module_inputs_of_several_stages_are_refused(tmp_path):
    with pytest.raises(
        StageInterfaceUnsupportedError, match="module-scope input 'pos'"
    ):
        _translate(tmp_path, TWO_STAGES, "directx")


def test_hlsl_output_compiles_with_dxc(tmp_path):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")
    source = tmp_path / "fragment.hlsl"
    source.write_text(_translate(tmp_path, FRAGMENT, "directx"), encoding="utf-8")
    completed = subprocess.run(
        [
            dxc,
            "-T",
            "ps_6_0",
            "-E",
            "PSMain",
            str(source),
            "-Fo",
            str(tmp_path / "f.dxo"),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
