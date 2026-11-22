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
from crosstl.translator import ast as A
from crosstl.translator.semantics.module_stage_outputs import (
    adopt_module_stage_outputs,
)
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


# The SPIR-V and GLSL importers write a module-scope stage input with the attribute instead of the qualifier.
ATTRIBUTE_VERTEX = """
shader main {
    float3 position @input @location(0);
    float2 uv @output @location(1);
    vertex {
        void main() {
            uv = position.xy;
            gl_Position = float4(position, 1.0);
        }
    }
}
"""

ATTRIBUTE_FRAGMENT = """
shader main {
    in float2 uv @input @location(0);
    float4 color @output @location(0);
    fragment {
        void main() {
            color = float4(uv, 0.0, 1.0);
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


@pytest.mark.parametrize(
    "backend, attribute, output",
    [
        ("directx", r"float3 position\s*:\s*TEXCOORD0;", r"float2 uv\s*:\s*TEXCOORD1;"),
        (
            "metal",
            r"float3 position \[\[attribute\(0\)\]\];",
            r"float2 uv \[\[user\(locn1\)\]\];",
        ),
        ("slang", r"float3 position\s*:\s*TEXCOORD0;", r"float2 uv\s*:\s*TEXCOORD1;"),
    ],
)
def test_attribute_module_interface_is_the_entry_interface(
    tmp_path, backend, attribute, output
):
    # Before, HLSL and Slang declared 'float3 position;' (a uniform) and MSL 'constant float3 position =
    # float3(0);', so the vertex read zero instead of its attribute.
    code = _flat(_translate(tmp_path, ATTRIBUTE_VERTEX, backend))
    assert re.search(attribute, code), code
    assert re.search(output, code), code
    assert "float4(input.position, 1.0)" in code, code
    assert "float3 position;" not in code, code
    assert "constant float3 position" not in code, code


def _module_variable(name, qualifiers=(), attributes=()):
    return A.VariableNode(
        name,
        "float3",
        qualifiers=list(qualifiers),
        attributes=[A.AttributeNode(attribute) for attribute in attributes],
    )


def _vertex_program(*variables):
    return A.ShaderNode(
        name="main",
        execution_model=A.ExecutionModel.GRAPHICS_PIPELINE,
        global_variables=list(variables),
        stages={A.ShaderStage.VERTEX: A.StageNode(A.ShaderStage.VERTEX, None)},
    )


def test_qualifier_and_attribute_of_one_direction_move_once():
    variable = _module_variable("uv", qualifiers=["in"], attributes=["input"])
    program = _vertex_program(variable)
    assert adopt_module_stage_outputs(program, "directx") == 1
    stage = next(iter(program.stages.values()))
    assert stage.local_variables == [variable]
    assert [attribute.name for attribute in variable.attributes] == ["input"]
    assert variable.qualifiers == []


@pytest.mark.parametrize(
    "qualifiers, attributes",
    [
        (["in"], ["output"]),
        ([], ["input", "output"]),
        (["uniform"], ["input"]),
    ],
)
def test_contradictory_or_resource_declarations_stay_at_module_scope(
    qualifiers, attributes
):
    variable = _module_variable("value", qualifiers=qualifiers, attributes=attributes)
    program = _vertex_program(variable)
    assert adopt_module_stage_outputs(program, "directx") == 0
    assert program.global_variables == [variable]


def test_attribute_fragment_output_compiles_with_dxc(tmp_path):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")
    source = tmp_path / "attribute_fragment.hlsl"
    code = _translate(tmp_path, ATTRIBUTE_FRAGMENT, "directx")
    assert "float2 uv;" not in _flat(code), code
    source.write_text(code, encoding="utf-8")
    completed = subprocess.run(
        [
            dxc,
            "-T",
            "ps_6_0",
            "-E",
            "PSMain",
            str(source),
            "-Fo",
            str(tmp_path / "a.dxo"),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]


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
