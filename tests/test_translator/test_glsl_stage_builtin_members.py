"""GLSL vertex output and fragment input members that name built-ins beyond the core gl_PerVertex set.

A vertex output ``@gl_Layer`` used to become ``flat out uint layer;``, a user varying that never selects a layer, and
a fragment input member ``@gl_HelperInvocation`` a varying no stage writes. The built-ins are now written and read,
with the extensions GLSL declares them under, and the vertex outputs GLSL cannot express are refused.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.glsl_builtin_extensions import (
    require_builtin_extensions,
)

GLSLANG = shutil.which("glslangValidator")
LAYER_EXTENSION = "#extension GL_ARB_shader_viewport_layer_array : require"


def translate(tmp_path, source, backend="opengl", name="shader.cgl"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def assert_compiles(tmp_path, code, stage):
    if GLSLANG is None:
        return
    result = subprocess.run(
        [GLSLANG, "--stdin", "-S", stage],
        input=code,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def user_varyings(code, direction):
    return re.findall(
        rf"^\s*(?:layout\([^)]*\)\s*)?(?:flat\s+)?{direction}\s+\w+\s+(\w+)", code, re.M
    )


VERTEX_LAYER = """
shader LayeredVertex {
    struct VertexOutput {
        vec4 position @gl_Position;
        uint layer @gl_Layer;
        int viewport @gl_ViewportIndex;
        vec2 uv;
    }
    vertex {
        VertexOutput main(uint id @gl_VertexID) {
            VertexOutput output;
            output.position = vec4(0.0, 0.0, 0.0, 1.0);
            output.layer = id % 4u;
            output.viewport = 1;
            output.uv = vec2(0.0);
            return output;
        }
    }
}
"""


def test_vertex_layer_and_viewport_outputs_write_the_builtins(tmp_path):
    glsl = translate(tmp_path, VERTEX_LAYER)

    assert glsl.splitlines()[1] == LAYER_EXTENSION
    assert re.search(r"gl_Layer = int\(", glsl)
    assert "gl_ViewportIndex = 1;" in glsl
    assert user_varyings(glsl, "out") == ["uv"]
    assert_compiles(tmp_path, glsl, "vert")


def test_hlsl_render_target_array_index_reaches_the_glsl_layer(tmp_path):
    source = """
struct VSOut {
    float4 pos : SV_Position;
    uint layer : SV_RenderTargetArrayIndex;
    uint viewport : SV_ViewportArrayIndex;
};
VSOut main(uint id : SV_VertexID) {
    VSOut o;
    o.pos = float4(0, 0, 0, 1);
    o.layer = id % 4;
    o.viewport = 1;
    return o;
}
"""
    glsl = translate(tmp_path, source, name="shader.vert.hlsl")

    assert LAYER_EXTENSION in glsl
    assert "gl_Layer = int((id % 4));" in glsl
    assert "gl_ViewportIndex = 1;" in glsl
    assert user_varyings(glsl, "out") == []
    assert_compiles(tmp_path, glsl, "vert")


def test_layer_written_by_the_vertex_stage_is_read_by_the_fragment_stage(tmp_path):
    source = """
shader Layered {
    struct VertexOutput { vec4 position @gl_Position; int layer @gl_Layer; vec2 uv; }
    vertex {
        VertexOutput main() {
            VertexOutput output;
            output.position = vec4(0.0);
            output.layer = 2;
            output.uv = vec2(0.0);
            return output;
        }
    }
    fragment {
        vec4 main(VertexOutput input) @gl_FragColor {
            return vec4(input.uv, float(input.layer), 1.0);
        }
    }
}
"""
    glsl = translate(tmp_path, source)

    assert glsl.count(LAYER_EXTENSION) == 1
    assert "gl_Layer = 2;" in glsl
    assert "float(gl_Layer)" in glsl
    assert "layer;" not in glsl
    assert_compiles(tmp_path, glsl, "vert")
    assert_compiles(tmp_path, glsl, "frag")


def test_webgl_refuses_a_vertex_layer_output(tmp_path):
    with pytest.raises(Exception, match="no layered or multi-viewport rendering"):
        translate(tmp_path, VERTEX_LAYER, backend="webgl")


@pytest.mark.parametrize(
    "builtin, value_type",
    [
        ("gl_PrimitiveShadingRateEXT", "int"),
        ("gl_ViewportMaskNV", "int"),
        ("gl_SecondaryPositionNV", "vec4"),
        ("gl_SecondaryViewportMaskNV", "int"),
        ("gl_PositionPerViewNV", "vec4"),
        ("gl_LayerPerViewNV", "int"),
    ],
)
def test_vertex_outputs_glsl_cannot_express_are_refused(tmp_path, builtin, value_type):
    source = f"""
shader P {{
    struct VertexOutput {{ vec4 position @gl_Position; {value_type} value @{builtin}; }}
    vertex {{
        VertexOutput main() {{
            VertexOutput output;
            output.position = vec4(0.0);
            output.value = {value_type}(0);
            return output;
        }}
    }}
}}
"""
    with pytest.raises(Exception, match=f"no vertex output built-in for @{builtin}"):
        translate(tmp_path, source)


@pytest.mark.parametrize(
    "builtin, value_type, read, glsl_name",
    [
        ("gl_HelperInvocation", "bool", "float(input.value)", "gl_HelperInvocation"),
        ("gl_Layer", "int", "float(input.value)", "gl_Layer"),
        ("gl_ViewportIndex", "int", "float(input.value)", "gl_ViewportIndex"),
        ("gl_NumSamples", "int", "float(input.value)", "gl_NumSamples"),
        ("gl_SubgroupSize", "uint", "float(input.value)", "gl_SubgroupSize"),
        (
            "gl_SubgroupInvocationID",
            "uint",
            "float(input.value)",
            "gl_SubgroupInvocationID",
        ),
        ("gl_SubgroupEqMask", "uvec4", "float(input.value.x)", "gl_SubgroupEqMask"),
        ("gl_BaryCoordEXT", "vec3", "input.value.x", "gl_BaryCoordEXT"),
        ("gl_BaryCoordSmoothAMD", "vec2", "input.value.x", "gl_BaryCoordSmoothAMD"),
        (
            "gl_BaryCoordPullModelAMD",
            "vec3",
            "input.value.x",
            "gl_BaryCoordPullModelAMD",
        ),
        ("gl_SMIDNV", "uint", "float(input.value)", "gl_SMIDNV"),
        ("gl_FullyCoveredEXT", "bool", "float(input.value)", "gl_FragFullyCoveredNV"),
        ("gl_FragSizeEXT", "ivec2", "float(input.value.x)", "gl_FragSizeEXT"),
        ("gl_ShadingRateEXT", "int", "float(input.value)", "gl_ShadingRateEXT"),
    ],
)
def test_fragment_input_members_read_the_builtins(
    tmp_path, builtin, value_type, read, glsl_name
):
    source = f"""
shader P {{
    struct FragmentInput {{ vec4 colour; {value_type} value @{builtin}; }}
    fragment {{
        vec4 main(FragmentInput input) @gl_FragColor {{
            return input.colour + vec4({read});
        }}
    }}
}}
"""
    glsl = translate(tmp_path, source)

    assert re.search(rf"\b{glsl_name}\b", glsl)
    assert user_varyings(glsl, "in") == ["colour"]
    assert_compiles(tmp_path, glsl, "frag")


def test_fragment_distance_parameters_read_the_first_elements(tmp_path):
    source = """
shader P {
    fragment {
        vec4 main(vec2 clip @gl_ClipDistance, float cull @gl_CullDistance) @gl_FragColor {
            return vec4(clip, cull, 1.0);
        }
    }
}
"""
    glsl = translate(tmp_path, source)

    assert "vec2 clip = vec2(gl_ClipDistance[0], gl_ClipDistance[1]);" in glsl
    assert "float cull = gl_CullDistance[0];" in glsl
    assert_compiles(tmp_path, glsl, "frag")


@pytest.mark.parametrize(
    "parameter, message",
    [
        ("float size @gl_PointSize", "no fragment input gl_PointSize"),
        ("uint quad @gl_QuadGroupID", "no quad-group index"),
        ("uint lane @gl_QuadGroupInvocationID", "no quad-group index"),
    ],
)
def test_fragment_parameters_glsl_does_not_declare_are_refused(
    tmp_path, parameter, message
):
    name = parameter.split()[1]
    source = f"""
shader P {{
    fragment {{
        vec4 main({parameter}) @gl_FragColor {{ return vec4(float({name})); }}
    }}
}}
"""
    with pytest.raises(Exception, match=message):
        translate(tmp_path, source)


def test_compute_quad_group_indices_still_derive_from_the_local_index(tmp_path):
    source = """
shader P {
    compute {
        buffer uint values[];
        @workgroup_size(8, 1, 1)
        void main(uint quad @gl_QuadGroupID, uint lane @gl_QuadGroupInvocationID) {
            values[quad] = lane;
        }
    }
}
"""
    glsl = translate(tmp_path, source)

    assert "gl_QuadGroup" not in glsl
    assert "gl_LocalInvocationIndex / 4u" in glsl
    assert_compiles(tmp_path, glsl, "comp")


@pytest.mark.parametrize(
    "builtin, extension",
    [
        ("gl_BaryCoordSmoothAMD", "GL_AMD_shader_explicit_vertex_parameter"),
        ("gl_SMIDNV", "GL_NV_shader_sm_builtins"),
        ("gl_FragFullyCoveredNV", "GL_NV_conservative_raster_underestimation"),
    ],
)
def test_webgl_refuses_desktop_vendor_fragment_builtins(tmp_path, builtin, extension):
    semantic = "gl_FullyCoveredEXT" if builtin == "gl_FragFullyCoveredNV" else builtin
    value_type = "vec2" if "Bary" in builtin else "uint"
    if builtin == "gl_FragFullyCoveredNV":
        value_type = "bool"
    read = "value.x" if value_type == "vec2" else "float(value)"
    source = f"""
shader P {{
    fragment {{
        vec4 main({value_type} value @{semantic}) @gl_FragColor {{ return vec4({read}); }}
    }}
}}
"""
    with pytest.raises(Exception, match=re.escape(extension)):
        translate(tmp_path, source, backend="webgl")


def test_stage_extensions_are_required_first_and_once():
    code = "#version 450 core\nvoid main() { gl_Layer = int(gl_SubgroupSize); }\n"

    lines = require_builtin_extensions(
        code,
        [
            "GL_ARB_shader_viewport_layer_array",
            "GL_KHR_shader_subgroup_basic",
        ],
    ).splitlines()

    assert lines[:4] == [
        "#version 450 core",
        "#extension GL_ARB_shader_viewport_layer_array : require",
        "#extension GL_KHR_shader_subgroup_basic : require",
        "void main() { gl_Layer = int(gl_SubgroupSize); }",
    ]


def test_vertex_functions_storing_the_layer_directly_require_the_extension(tmp_path):
    source = """
shader P {
    vertex {
        void pick(uint layer) { gl_Layer = layer; }
        void main() { gl_Position = vec4(0.0); pick(2u); gl_ViewportIndex = 1; }
    }
}
"""
    glsl = translate(tmp_path, source)

    assert glsl.count(LAYER_EXTENSION) == 1
    assert "gl_Layer = int(layer);" in glsl
    assert_compiles(tmp_path, glsl, "vert")


def test_fragment_layer_reads_need_no_extension(tmp_path):
    source = """
shader P {
    fragment {
        vec4 main(int layer @gl_Layer) @gl_FragColor { return vec4(float(layer)); }
    }
}
"""
    glsl = translate(tmp_path, source)

    assert "#extension" not in glsl
    assert "float(gl_Layer)" in glsl
    assert_compiles(tmp_path, glsl, "frag")
