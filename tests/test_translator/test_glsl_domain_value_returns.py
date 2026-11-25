"""HLSL domain shaders that return a value instead of a structure lower to GLSL and SPIR-V tessellation stages.

``float4 main(...) : SV_Position`` is a domain shader whose result is the vertex position; a value with a user
semantic is one stage output. The lowered evaluation entry point returns nothing and writes the value to
``gl_Position`` or to one located ``out`` variable, as it writes the members of a returned structure.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics import TessellationPatchUnsupportedError

HULL_HLSL = """
struct ControlPoint { float4 position : POSITION; float3 normal : NORMAL0; };
struct PatchConstants { float edges[3] : SV_TessFactor; float inside : SV_InsideTessFactor; };

PatchConstants Constants(InputPatch<ControlPoint, 3> patch)
{
    PatchConstants c;
    c.edges[0] = 2.0; c.edges[1] = 2.0; c.edges[2] = 2.0; c.inside = 3.0;
    return c;
}

[domain("tri")]
[partitioning("fractional_even")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("Constants")]
ControlPoint Hull(InputPatch<ControlPoint, 3> patch, uint id : SV_OutputControlPointID)
{
    return patch[id];
}
"""

DOMAIN_HLSL = """
[domain("tri")]
float4 Domain(PatchConstants constants, float3 uvw : SV_DomainLocation,
              const OutputPatch<ControlPoint, 3> patch) : RESULT_SEMANTIC
{
    return (patch[0].position * uvw.x + patch[1].position * uvw.y + patch[2].position * uvw.z)
        * (constants.inside / constants.edges[1]);
}
"""

FRAGMENT_HLSL = """
float4 main(float4 value : TEXCOORD2) : SV_Target0 { return value; }
"""


def _source(result_semantic, hull=True):
    domain = DOMAIN_HLSL.replace("RESULT_SEMANTIC", result_semantic)
    return (
        HULL_HLSL if hull else HULL_HLSL.split("PatchConstants Constants")[0]
    ) + domain


def _translate(tmp_path, source, backend, entry_point=None, name="t.hlsl"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(
        str(path),
        backend=backend,
        format_output=False,
        source_backend="directx",
        entry_point=entry_point,
    )


def _glslang(tmp_path, code, stage):
    if shutil.which("glslangValidator") is None:
        return
    path = tmp_path / f"out.{stage}"
    path.write_text(code)
    result = subprocess.run(
        ["glslangValidator", "-V", str(path), "-o", str(tmp_path / f"{stage}.spv")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _spirv_val(tmp_path, code):
    if shutil.which("spirv-as") is None or shutil.which("spirv-val") is None:
        return
    asm = tmp_path / "out.spvasm"
    asm.write_text(code)
    binary = tmp_path / "out.spv"
    for command in (
        ["spirv-as", "--target-env", "vulkan1.1", str(asm), "-o", str(binary)],
        ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stdout + result.stderr


def test_domain_position_value_is_written_to_gl_position(tmp_path):
    source = _source("SV_Position")
    domain = _translate(tmp_path, source, "opengl", entry_point="Domain")

    assert "layout(triangles, fractional_even_spacing, cw) in;" in domain
    assert "vec4 Domain_domain_point(PatchConstants constants, vec3 uvw)" in domain
    assert re.search(r"\bvoid main\(\)", domain)
    assert "vec4 ds_vertex = Domain_domain_point(ds_patch, gl_TessCoord);" in domain
    assert "gl_Position = ds_vertex;" in domain
    assert "ds_patch.inside = gl_TessLevelInner[0];" in domain
    # The value's semantic moved to the output: no user varying is declared for it.
    assert not re.search(r"\bout\s+vec4\b", domain)
    assert "OutputPatch" not in domain
    _glslang(tmp_path, domain, "tese")

    hull = _translate(tmp_path, source, "opengl", entry_point="Hull")
    _glslang(tmp_path, hull, "tesc")


def test_domain_value_with_a_user_semantic_is_one_located_output(tmp_path):
    source = _source("TEXCOORD2")
    domain = _translate(tmp_path, source, "opengl", entry_point="Domain")
    fragment = _translate(tmp_path, FRAGMENT_HLSL, "opengl", name="f.frag.hlsl")

    output = re.search(r"layout\(location = (\d+)\) out vec4 (te_out_\w+);", domain)
    assert output is not None, domain
    assert f"{output.group(2)} = ds_vertex;" in domain
    assert "gl_Position" not in domain
    # The fragment stage reads TEXCOORD2 at the location the domain stage writes it to.
    fragment_input = re.search(r"layout\(location = (\d+)\) in vec4 \w+;", fragment)
    assert fragment_input is not None, fragment
    assert output.group(1) == fragment_input.group(1)
    _glslang(tmp_path, domain, "tese")


def test_domain_value_lowers_to_spirv(tmp_path):
    spirv = _translate(tmp_path, _source("SV_Position"), "vulkan")
    entry = re.search(
        r'OpEntryPoint TessellationEvaluation %\w+ "Domain"([^\n]*)', spirv
    )
    assert entry is not None, spirv
    positions = re.findall(r"OpDecorate %(\w+) BuiltIn Position", spirv)
    assert set(positions) & set(re.findall(r"%(\w+)", entry.group(1))), spirv
    _spirv_val(tmp_path, spirv)


def test_domain_value_without_its_hull_is_still_refused_for_glsl(tmp_path):
    with pytest.raises(TessellationPatchUnsupportedError) as error:
        _translate(tmp_path, _source("SV_Position", hull=False), "opengl")
    assert error.value.reason_code == "domain-without-hull"


def test_domain_value_with_a_system_value_semantic_is_refused(tmp_path):
    with pytest.raises(TessellationPatchUnsupportedError) as error:
        _translate(
            tmp_path, _source("SV_ClipDistance0"), "opengl", entry_point="Domain"
        )
    assert error.value.reason_code == "system-value"
