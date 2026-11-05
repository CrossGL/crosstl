"""Instance indices across Metal's and Slang's base-inclusive and base-exclusive system values.

OpenGL's ``gl_InstanceID`` and Direct3D's ``SV_InstanceID`` count from zero; MSL ``[[instance_id]]`` and Vulkan's
``gl_InstanceIndex`` count from the draw's base instance. The Metal generator bound ``gl_InstanceID`` to
``[[instance_id]]`` and the Metal importer read ``[[instance_id]]`` as ``gl_InstanceID``, so a draw with a non-zero
base instance read shifted instance numbers. Metal output now subtracts ``[[base_instance]]``, and the importer reads
``[[instance_id]]`` as ``gl_InstanceIndex``.

Slang has no system value for ``gl_InstanceIndex`` or ``gl_VertexIndex``: its output copied their reads verbatim
(slangc: undeclared identifier) and refused parameters carrying them. Slang output now spells ``gl_InstanceIndex`` as
``SV_InstanceID + SV_StartInstanceLocation`` and ``gl_VertexIndex`` as ``gl_VertexID`` is spelled.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
import crosstl.translator
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.semantics.slang_vertex_indices import spell_slang_vertex_indices

GLSL_INSTANCE_ID = """#version 460
layout(location = 0) in vec4 pos;
layout(location = 0) out flat int instanceOut;
void main() {
    instanceOut = gl_InstanceID;
    gl_Position = pos + vec4(float(gl_InstanceID), 0.0, 0.0, 0.0);
}
"""

GLSL_BOTH = """#version 460
layout(location = 0) in vec4 pos;
layout(location = 0) out flat int a;
layout(location = 1) out flat int b;
void main() {
    a = gl_InstanceID;
    b = gl_InstanceIndex + gl_BaseInstance;
    gl_Position = pos;
}
"""

HLSL_MEMBER = """struct VSIn { float4 pos : POSITION; uint iid : SV_InstanceID; };
float4 main(VSIn i, uint start : SV_StartInstanceLocation) : SV_Position {
    return i.pos + float4(float(i.iid), float(start), 0.0, 0.0);
}
"""

METAL = """#include <metal_stdlib>
using namespace metal;
struct VOut { float4 pos [[position]]; float k; };
vertex VOut vmain(uint iid [[instance_id]], uint bi [[base_instance]]) {
    VOut o;
    o.pos = float4(float(iid), float(bi), 0.0, 1.0);
    o.k = float(iid - bi);
    return o;
}
"""

VULKAN_GLSL = """#version 460
layout(location = 0) in vec4 pos;
float shift() { return float(gl_InstanceIndex); }
void main() {
    gl_Position = pos + vec4(shift(), float(gl_VertexIndex), float(gl_InstanceIndex), 0.0);
}
"""


def _translate(tmp_path, source, suffix, backend):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def _signature(code, keyword):
    # Formatting differs with and without clang-format: any whitespace may separate the tokens.
    match = re.search(rf"{keyword}\s+\w+\s+\w+\((.*?)\)\s*\{{", code, flags=re.S)
    assert match is not None, code
    return " ".join(match.group(1).split())


def _compile_metal(tmp_path, code):
    if not shutil.which("xcrun"):
        return
    found = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True
    )
    if found.returncode != 0:
        return
    source = tmp_path / "out.metal"
    source.write_text(code)
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-std=metal3.0", "-c", str(source)]
        + ["-o", str(tmp_path / "out.air")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + code


def _compile_slang(tmp_path, code, entry):
    slangc = shutil.which("slangc")
    if slangc is None:
        return None
    source = tmp_path / "out.slang"
    source.write_text(code)
    output = tmp_path / "out.spvasm"
    result = subprocess.run(
        [slangc, str(source), "-entry", entry, "-stage", "vertex"]
        + ["-target", "spirv-asm", "-o", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + code
    return output.read_text()


def test_metal_gl_instance_id_subtracts_the_base_instance(tmp_path):
    metal = _translate(tmp_path, GLSL_INSTANCE_ID, ".vert", "metal")
    signature = _signature(metal, "vertex")
    instance = re.search(r"uint (\w+) \[\[instance_id\]\]", signature)
    base = re.search(r"uint (\w+) \[\[base_instance\]\]", signature)
    assert instance and base, signature
    assert re.search(
        rf"uint (\w+) = uint\({instance.group(1)} - {base.group(1)}\);", metal
    )
    assert metal.count("[[instance_id]]") == 1
    _compile_metal(tmp_path, metal)


def test_metal_instance_index_and_instance_id_share_one_attribute(tmp_path):
    metal = _translate(tmp_path, GLSL_BOTH, ".vert", "metal")
    signature = _signature(metal, "vertex")
    assert signature.count("[[instance_id]]") == 1
    assert signature.count("[[base_instance]]") == 1
    instance = re.search(r"uint (\w+) \[\[instance_id\]\]", signature).group(1)
    base = re.search(r"uint (\w+) \[\[base_instance\]\]", signature).group(1)
    # gl_InstanceID is the difference; gl_InstanceIndex is the attribute itself.
    assert re.search(rf"uint \w+ =\s*uint\({instance} - {base}\);", metal)
    assert re.search(rf"as_type<uint>\({instance}\)", metal)
    _compile_metal(tmp_path, metal)


def test_metal_sv_instance_id_member_reuses_the_start_instance_parameter(tmp_path):
    metal = _translate(tmp_path, HLSL_MEMBER, ".hlsl", "metal")
    signature = _signature(metal, "vertex")
    assert "uint start [[base_instance]]" in signature
    assert signature.count("[[base_instance]]") == 1
    instance = re.search(r"uint (\w+) \[\[instance_id\]\]", signature).group(1)
    assert f"uint i_iid = uint({instance} - start);" in metal
    assert "i.iid = i_iid;" in metal
    _compile_metal(tmp_path, metal)


@pytest.mark.parametrize("semantic", ["gl_InstanceID", "SV_InstanceID"])
def test_metal_codegen_rebases_a_crossgl_instance_parameter(semantic):
    ast = crosstl.translator.parse(f"""
    shader S {{
        vertex {{
            vec4 main(vec4 pos @ POSITION, int inst @ {semantic}) @ gl_Position {{
                return pos + vec4(float(inst), 0.0, 0.0, 0.0);
            }}
        }}
    }}
    """)
    code = MetalCodeGen().generate(ast)
    assert "[[instance_id]]" in code and "[[base_instance]]" in code
    assert re.search(
        r"int inst = int\(_crossglInstanceIndex - _crossglBaseInstance\);", code
    )


def test_metal_instance_index_parameter_is_not_rebased():
    ast = crosstl.translator.parse("""
    shader S {
        vertex {
            vec4 main(vec4 pos @ POSITION, uint inst @ gl_InstanceIndex) @ gl_Position {
                return pos + vec4(float(inst), 0.0, 0.0, 0.0);
            }
        }
    }
    """)
    code = MetalCodeGen().generate(ast)
    assert "uint inst [[instance_id]]" in code
    assert "base_instance" not in code


def test_metal_float_instance_id_is_still_refused():
    ast = crosstl.translator.parse("""
    shader S {
        vertex {
            vec4 main(vec4 pos @ POSITION, float inst @ gl_InstanceID) @ gl_Position {
                return pos + vec4(inst, 0.0, 0.0, 0.0);
            }
        }
    }
    """)
    with pytest.raises(Exception, match="instance_id|gl_InstanceID"):
        MetalCodeGen().generate(ast)


def test_metal_import_reads_instance_id_as_instance_index(tmp_path):
    cgl = _translate(tmp_path, METAL, ".metal", "cgl")
    assert "uint iid @gl_InstanceIndex" in cgl
    assert "@gl_InstanceID" not in cgl

    metal = _translate(tmp_path, METAL, ".metal", "metal")
    signature = _signature(metal, "vertex")
    assert "uint iid [[instance_id]]" in signature
    assert "uint bi [[base_instance]]" in signature
    assert "_crossgl" not in metal
    _compile_metal(tmp_path, metal)

    glsl = _translate(tmp_path, METAL, ".metal", "opengl")
    assert re.search(
        r"uint iid = uint\(\(gl_InstanceID \+ gl_BaseInstance(ARB)?\)\);", glsl
    )


def test_slang_spells_vulkan_indices_through_system_values(tmp_path):
    slang = _translate(tmp_path, VULKAN_GLSL, ".vert", "slang")
    assert not re.search(r"\bgl_(InstanceIndex|VertexIndex)\b", slang)
    assert re.search(r"\w+\s*:\s*SV_InstanceID\b", slang)
    assert re.search(r"\w+\s*:\s*SV_StartInstanceLocation\b", slang)
    assert re.search(r"\w+\s*:\s*SV_VertexID\b", slang)
    # The helper reads the index through the parameters it is passed.
    assert re.search(r"float shift\(int \w+,\s*int \w+\)", slang)
    spirv = _compile_slang(tmp_path, slang, "main")
    if spirv is not None:
        assert "BuiltIn InstanceIndex" in spirv
        assert "BuiltIn BaseInstance" in spirv


def test_slang_instance_index_parameter_becomes_a_sum(tmp_path):
    slang = _translate(tmp_path, METAL, ".metal", "slang")
    signature = _signature(slang, r"\[shader\(\"vertex\"\)\]")
    assert "uint bi : SV_StartInstanceLocation" in signature
    instance = re.search(r"uint (\w+) : SV_InstanceID", signature)
    assert instance is not None, signature
    assert f"uint iid = uint({instance.group(1)} + bi);" in slang
    _compile_slang(tmp_path, slang, "vmain")


def test_slang_vertex_index_parameter_is_spelled_as_vertex_id():
    ast = crosstl.translator.parse("""
    shader S {
        vertex {
            vec4 main(vec4 pos @ POSITION, uint vid @ gl_VertexIndex) @ gl_Position {
                return pos + vec4(float(vid), 0.0, 0.0, 0.0);
            }
        }
    }
    """)
    assert spell_slang_vertex_indices(ast, "slang") == 1
    (stage,) = ast.stages.values()
    attributes = [a.name for p in stage.entry_point.parameters for a in p.attributes]
    assert "gl_VertexID" in attributes and "gl_VertexIndex" not in attributes
    assert spell_slang_vertex_indices(ast, "metal") == 0
