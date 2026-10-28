"""``gl_`` built-in value names: unknown names are refused, aliases take the name every generator reads."""

import re

import pytest

import crosstl
from crosstl.translator.semantics.builtin_names import (
    CROSSGL_GL_BUILTIN_NAMES,
    is_crossgl_gl_builtin,
)
from crosstl.translator.semantics.builtin_values import BuiltinValueUnsupportedError

GRAPHICS_TARGETS = ["vulkan", "metal", "directx", "opengl", "webgl"]
ALL_TARGETS = GRAPHICS_TARGETS + ["slang", "cuda", "hip", "rust", "mojo", "wgsl"]

VERTEX_MEMBER = """shader Probe {
    struct VertexOutput {
        vec4 position @gl_Position;
        float value SEMANTIC;
    }
    vertex {
        VertexOutput main() {
            VertexOutput output;
            output.position = vec4(0.0);
            output.value = 1.0;
            return output;
        }
    }
}
"""

FRAGMENT_PARAMETER = """shader Probe {
    fragment {
        vec4 main(float value SEMANTIC) @gl_FragColor {
            return vec4(value);
        }
    }
}
"""

COMPUTE_VARIABLE = """shader Probe {
    compute {
        uint index @input SEMANTIC;
        buffer uint values[];
        @workgroup_size(1, 1, 1)
        void main() {
            values[0] = index;
        }
    }
}
"""

FRONT_FACING = """shader Probe {
    struct FragmentInput {
        bool front @gl_IsFrontFace;
    }
    struct FragmentOutput {
        vec4 colour @gl_FragColor;
    }
    fragment {
        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            output.colour = input.front ? vec4(1.0) : vec4(0.0);
            return output;
        }
    }
}
"""

STENCIL = """shader Probe {
    struct FragmentOutput {
        vec4 colour @gl_FragColor;
        uint stencil @gl_FragStencilRefARB;
    }
    fragment {
        FragmentOutput main() {
            FragmentOutput output;
            output.colour = vec4(1.0);
            output.stencil = 3u;
            return output;
        }
    }
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize("backend", ALL_TARGETS)
@pytest.mark.parametrize(
    "template, context",
    [
        (VERTEX_MEMBER, "member 'value' of 'VertexOutput'"),
        (FRAGMENT_PARAMETER, "parameter 'value' of 'main'"),
        (COMPUTE_VARIABLE, "variable 'index'"),
    ],
    ids=["member", "parameter", "variable"],
)
def test_unknown_gl_names_are_refused(tmp_path, backend, template, context):
    source = template.replace("SEMANTIC", "@gl_FragBogus")
    with pytest.raises(BuiltinValueUnsupportedError) as raised:
        _translate(tmp_path, source, backend)
    message = str(raised.value)
    assert "Unknown built-in value @gl_FragBogus" in message
    assert context in message
    assert raised.value.code == "project.translate.builtin-value-unsupported"


@pytest.mark.parametrize("backend", GRAPHICS_TARGETS)
def test_unknown_gl_name_on_return_value_is_refused(tmp_path, backend):
    source = FRAGMENT_PARAMETER.replace("SEMANTIC", "@location(0)").replace(
        "@gl_FragColor", "@gl_FragColour"
    )
    with pytest.raises(
        BuiltinValueUnsupportedError, match="the return value of 'main'"
    ):
        _translate(tmp_path, source, backend)


@pytest.mark.parametrize(
    "name",
    [
        "gl_Position",
        "gl_position",
        "GL_FRAGCOORD",
        "gl_FragColor",
        "gl_FragColor3",
        "gl_FragData",
        "gl_FragData[2]",
        "gl_ClipDistance1",
        "gl_CullDistance2",
        "gl_SubgroupLtMask",
        "gl_PrimitiveTriangleIndicesEXT",
        "gl_HitTriangleVertexPositionsEXT",
        "gl_QuadGroupInvocationID",
        "gl_FragStencilRefARB",
    ],
)
def test_known_gl_names(name):
    assert is_crossgl_gl_builtin(name)


@pytest.mark.parametrize(
    "name",
    [
        "gl_FragBogus",
        "gl_Position1",
        "gl_FragDepth0",
        "gl_FragData[x]",
        "gl_",
        "gl_PerVertex",
    ],
)
def test_unknown_gl_names(name):
    assert not is_crossgl_gl_builtin(name)


def test_registry_is_glsl_reserved_names_only():
    assert all(name.startswith("gl_") for name in CROSSGL_GL_BUILTIN_NAMES)
    assert len({name.lower() for name in CROSSGL_GL_BUILTIN_NAMES}) == len(
        CROSSGL_GL_BUILTIN_NAMES
    )


def _spirv_builtin_variable(code, builtin):
    """The id of the one interface variable decorated ``BuiltIn builtin``; it has no ``Location``."""
    variables = re.findall(rf"OpDecorate %(\w+) BuiltIn {builtin}\b", code)
    assert len(variables) == 1, code
    assert not re.search(rf"OpDecorate %{re.escape(variables[0])} Location\b", code)
    return variables[0]


def test_spirv_reads_the_front_facing_alias_as_the_builtin(tmp_path):
    code = _translate(tmp_path, FRONT_FACING, "vulkan")
    _spirv_builtin_variable(code, "FrontFacing")
    assert len(re.findall(r"OpDecorate %\w+ Location\b", code)) == 1


@pytest.mark.parametrize(
    "backend, expected",
    [
        ("metal", "[[front_facing]]"),
        ("directx", ": SV_IsFrontFace"),
        ("opengl", "gl_FrontFacing"),
    ],
)
def test_front_facing_alias_keeps_its_builtin(tmp_path, backend, expected):
    assert expected in _translate(tmp_path, FRONT_FACING, backend)


def test_spirv_reads_the_arb_stencil_reference_as_the_builtin(tmp_path):
    code = _translate(tmp_path, STENCIL, "vulkan")
    _spirv_builtin_variable(code, "FragStencilRefEXT")
    assert "OpCapability StencilExportEXT" in code


@pytest.mark.parametrize(
    "backend, expected",
    [
        ("directx", r"uint stencil\s*:\s*SV_StencilRef;"),
        ("opengl", r"gl_FragStencilRefARB = "),
    ],
)
def test_arb_stencil_reference_keeps_its_builtin(tmp_path, backend, expected):
    code = _translate(tmp_path, STENCIL, backend)
    assert re.search(expected, code), code
    assert "gl_FragStencilRefARB;" not in code
