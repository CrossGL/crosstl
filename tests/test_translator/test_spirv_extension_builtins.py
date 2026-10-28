"""SPIR-V built-ins of GLSL extensions, and the refusal of ``gl_`` built-ins SPIR-V has no built-in for."""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.spirv_extension_builtins import unmapped_builtin_reason
from crosstl.translator.codegen.spirv_refusals import SpirvLoweringRefused

FRAGMENT_INPUT = """shader Probe {
    fragment {
        vec4 main(TYPE probe @NAME) @gl_FragColor {
            return vec4(float(probeREAD));
        }
    }
}
"""

COMPUTE_INPUT = """shader Probe {
    compute {
        buffer float values[];
        @workgroup_size(1, 1, 1)
        void main(TYPE probe @NAME) {
            values[0] = float(probeREAD);
        }
    }
}
"""

# (name, CrossGL type, component read, BuiltIn, capability or None, extension or None)
FRAGMENT_BUILTINS = [
    ("gl_HelperInvocation", "bool", "", "HelperInvocation", None, None),
    (
        "gl_FullyCoveredEXT",
        "bool",
        "",
        "FullyCoveredEXT",
        "FragmentFullyCoveredEXT",
        "SPV_EXT_fragment_fully_covered",
    ),
    (
        "gl_BaryCoordSmoothAMD",
        "vec2",
        ".x",
        "BaryCoordSmoothAMD",
        None,
        "SPV_AMD_shader_explicit_vertex_parameter",
    ),
    (
        "gl_BaryCoordNoPerspSampleAMD",
        "vec2",
        ".y",
        "BaryCoordNoPerspSampleAMD",
        None,
        "SPV_AMD_shader_explicit_vertex_parameter",
    ),
    (
        "gl_BaryCoordPullModelAMD",
        "vec3",
        ".z",
        "BaryCoordPullModelAMD",
        None,
        "SPV_AMD_shader_explicit_vertex_parameter",
    ),
    (
        "gl_BaryCoordNV",
        "vec3",
        ".x",
        "BaryCoordKHR",
        "FragmentBarycentricKHR",
        "SPV_KHR_fragment_shader_barycentric",
    ),
    (
        "gl_BaryCoordNoPerspNV",
        "vec3",
        ".y",
        "BaryCoordNoPerspKHR",
        "FragmentBarycentricKHR",
        "SPV_KHR_fragment_shader_barycentric",
    ),
]

ANY_STAGE_BUILTINS = [
    ("gl_DeviceIndex", "int", "DeviceIndex", "DeviceGroup", "SPV_KHR_device_group"),
    (
        "gl_SMCountNV",
        "uint",
        "SMCountNV",
        "ShaderSMBuiltinsNV",
        "SPV_NV_shader_sm_builtins",
    ),
    ("gl_SMIDNV", "uint", "SMIDNV", "ShaderSMBuiltinsNV", "SPV_NV_shader_sm_builtins"),
    (
        "gl_WarpIDNV",
        "uint",
        "WarpIDNV",
        "ShaderSMBuiltinsNV",
        "SPV_NV_shader_sm_builtins",
    ),
    (
        "gl_WarpsPerSMNV",
        "uint",
        "WarpsPerSMNV",
        "ShaderSMBuiltinsNV",
        "SPV_NV_shader_sm_builtins",
    ),
]

# Built-ins CrossGL defines that SPIR-V has no built-in for in these positions.
UNMAPPED = [
    (FRAGMENT_INPUT, "gl_NumSamples", "uint", ""),
    (COMPUTE_INPUT, "gl_NumSamples", "uint", ""),
    (FRAGMENT_INPUT, "gl_QuadGroupID", "uint", ""),
    (COMPUTE_INPUT, "gl_QuadGroupInvocationID", "uint", ""),
    # SPIR-V reads the fragment's clip distances only through stage structure members (spirv_stage_distances).
    (FRAGMENT_INPUT, "gl_ClipDistance", "float", ""),
    (FRAGMENT_INPUT, "gl_CullDistance", "float", ""),
]

VERTEX_OUTPUT = """shader Probe {
    struct VertexOutput {
        vec4 position @gl_Position;
        TYPE probe @NAME;
    }
    vertex {
        VertexOutput main() {
            VertexOutput output;
            output.position = vec4(0.0);
            output.probe = VALUE;
            return output;
        }
    }
}
"""

UNMAPPED_VERTEX_OUTPUTS = [
    ("gl_ViewportMaskNV", "int", "1"),
    ("gl_SecondaryPositionNV", "vec4", "vec4(0.0)"),
    ("gl_PositionPerViewNV", "vec4", "vec4(0.0)"),
    ("gl_LayerPerViewNV", "int", "0"),
]


def _translate(tmp_path, source):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend="vulkan", format_output=False)


def _source(template, name, type_name, read):
    return (
        template.replace("TYPE", type_name).replace("NAME", name).replace("READ", read)
    )


def _validate(tmp_path, code):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        return
    asm, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
    asm.write_text(code)
    for command in (
        ["spirv-as", "--target-env", "vulkan1.1", str(asm), "-o", str(binary)],
        ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr + result.stdout


def _builtin_variable_has_no_location(code, builtin):
    match = re.search(rf"OpDecorate %(\w+) BuiltIn {builtin}\b", code)
    assert match, f"no BuiltIn {builtin}"
    assert not re.search(rf"OpDecorate %{match.group(1)} Location\b", code)
    return match.group(1)


@pytest.mark.parametrize(
    "name, type_name, read, builtin, capability, extension", FRAGMENT_BUILTINS
)
def test_fragment_extension_builtins(
    tmp_path, name, type_name, read, builtin, capability, extension
):
    code = _translate(tmp_path, _source(FRAGMENT_INPUT, name, type_name, read))
    _builtin_variable_has_no_location(code, builtin)
    if capability is not None:
        assert f"OpCapability {capability}" in code
    if extension is not None:
        assert f'OpExtension "{extension}"' in code
    _validate(tmp_path, code)


@pytest.mark.parametrize("template", [FRAGMENT_INPUT, COMPUTE_INPUT])
@pytest.mark.parametrize(
    "name, type_name, builtin, capability, extension", ANY_STAGE_BUILTINS
)
def test_any_stage_extension_builtins(
    tmp_path, template, name, type_name, builtin, capability, extension
):
    code = _translate(tmp_path, _source(template, name, type_name, ""))
    _builtin_variable_has_no_location(code, builtin)
    assert f"OpCapability {capability}" in code
    assert f'OpExtension "{extension}"' in code
    _validate(tmp_path, code)


@pytest.mark.parametrize("template, name, type_name, read", UNMAPPED)
def test_unmapped_inputs_are_refused(tmp_path, template, name, type_name, read):
    with pytest.raises(SpirvLoweringRefused, match=rf"no input built-in for @{name}"):
        _translate(tmp_path, _source(template, name, type_name, read))


@pytest.mark.parametrize("name, type_name, value", UNMAPPED_VERTEX_OUTPUTS)
def test_unmapped_vertex_outputs_are_refused(tmp_path, name, type_name, value):
    source = (
        VERTEX_OUTPUT.replace("TYPE", type_name)
        .replace("NAME", name)
        .replace("VALUE", value)
    )
    with pytest.raises(
        SpirvLoweringRefused, match=rf"no output built-in for @{name} in a Vertex stage"
    ):
        _translate(tmp_path, source)


@pytest.mark.parametrize(
    "semantic", ["gl_FragColor", "gl_FragColor1", "gl_FragData[2]", "TEXCOORD0", None]
)
def test_colour_outputs_and_user_semantics_keep_locations(semantic):
    assert unmapped_builtin_reason(semantic, "Output", "Fragment") is None


def test_unknown_gl_names_are_left_to_the_name_check():
    # resolve_builtin_names refuses names CrossGL does not define before generation.
    assert unmapped_builtin_reason("gl_FragBogus", "Input", "Fragment") is None
