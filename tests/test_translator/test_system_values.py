"""HLSL system-value attributes reach every target as the system value, or are refused.

CrossGL accepts HLSL system-value semantics as built-in value attributes. Outside the fourteen spellings
:mod:`crosstl.translator.semantics.builtin_values` listed, GLSL and SPIR-V declared them as user varyings and Metal as
vertex attributes or user varyings: a vertex stage writing ``@SV_RenderTargetArrayIndex`` selected no layer, a fragment
``@SV_StencilRef`` wrote no stencil reference, and ``@SV_Target3`` wrote attachment 0 in GLSL. Spellings Direct3D does
not define, which DXC rejects, became user varyings the same way.

Each program below is checked for the built-in in every target's output and compiled with the target's native
checker when it is installed: ``glslangValidator``, ``spirv-as`` and ``spirv-val``, ``xcrun metal`` and ``dxc``.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics.builtin_values import BuiltinValueUnsupportedError
from crosstl.translator.semantics.system_values import (
    SystemValueError,
    lookup_system_value,
)

VERTEX_OUTPUTS = """shader S {
struct VOut {
    vec4 pos @ SV_Position;
    float clip @ SV_ClipDistance0;
    uint layer @ SV_RenderTargetArrayIndex;
    uint viewport @ SV_ViewportArrayIndex;
    vec2 uv @ TEXCOORD0;
};
struct FOut { vec4 color @ SV_Target; };
vertex {
    VOut main(vec3 p @ POSITION) {
        VOut o;
        o.pos = vec4(p, 1.0);
        o.clip = p.x;
        o.layer = 2u;
        o.viewport = 1u;
        o.uv = p.xy;
        return o;
    }
}
fragment {
    FOut main(VOut i) {
        FOut o;
        o.color = vec4(i.uv, float(i.layer), float(i.viewport));
        return o;
    }
}
}
"""

DISTANCES = """shader S {
struct VOut {
    vec4 pos @ SV_Position;
    float clip0 @ SV_ClipDistance0;
    float clip1 @ SV_ClipDistance1;
    float cull0 @ SV_CullDistance0;
    vec2 cull1 @ SV_CullDistance1;
};
vertex {
    VOut main(vec3 p @ POSITION) {
        VOut o;
        o.pos = vec4(p, 1.0);
        o.clip0 = p.x;
        o.clip1 = p.y;
        o.cull0 = p.z;
        o.cull1 = p.xy;
        return o;
    }
}
}
"""

FRAGMENT_OUTPUTS = """shader S {
struct FIn { vec2 uv @ TEXCOORD0; };
struct FOut {
    vec4 a @ SV_Target3;
    vec4 b @ SV_Target1;
    float depth @ SV_DepthGreaterEqual;
    uint stencil @ SV_StencilRef;
};
fragment {
    FOut main(FIn i) {
        FOut o;
        o.a = vec4(i.uv, 0.0, 1.0);
        o.b = vec4(1.0);
        o.depth = i.uv.x + 0.25;
        o.stencil = 7u;
        return o;
    }
}
}
"""

FRAGMENT_INPUTS = """shader S {
struct FIn {
    vec4 pos @ SV_Position;
    uint layer @ SV_RenderTargetArrayIndex;
    uint viewport @ SV_ViewportArrayIndex;
};
struct FOut { vec4 color @ SV_Target; };
fragment {
    FOut main(FIn i, vec3 bary @ SV_Barycentrics, vec3 linear @ SV_Barycentrics1 @ noperspective,
              uint inner @ SV_InnerCoverage) {
        FOut o;
        o.color = vec4(bary + linear, float(i.layer + i.viewport + inner));
        return o;
    }
}
}
"""

VERTEX_INPUTS = """shader S {
struct VOut { vec4 pos @ SV_Position; };
vertex {
    VOut main(vec3 p @ POSITION, uint base_vertex @ SV_StartVertexLocation,
              uint base_instance @ SV_StartInstanceLocation) {
        VOut o;
        o.pos = vec4(p, float(base_vertex + base_instance));
        return o;
    }
}
}
"""

VIEW_INDEX = """shader S {
struct VOut { vec4 pos @ SV_Position; };
vertex {
    VOut main(vec3 p @ POSITION, uint view @ SV_ViewID) {
        VOut o;
        o.pos = vec4(p, float(view));
        return o;
    }
}
}
"""

# Patterns each output must contain. A user varying would carry a location or a user attribute instead.
EXPECTED = {
    "vertex_outputs": {
        "opengl": [
            r"gl_ClipDistance\[0\] = ",
            r"\bgl_Layer = ",
            r"\bgl_ViewportIndex = ",
            r"#extension GL_ARB_shader_viewport_layer_array : require",
        ],
        "vulkan": [
            r"BuiltIn ClipDistance",
            r"%gl_Layer BuiltIn Layer",
            r"BuiltIn ViewportIndex",
            r"OpCapability ShaderViewportIndexLayerEXT",
            r'OpExtension "SPV_EXT_shader_viewport_index_layer"',
        ],
        "metal": [
            r"\[\[clip_distance\]\]",
            r"\[\[render_target_array_index\]\]",
            r"\[\[viewport_array_index\]\]",
        ],
        "directx": [
            r": SV_ClipDistance0",
            r": SV_RenderTargetArrayIndex",
            r": SV_ViewportArrayIndex",
        ],
    },
    "distances": {
        "opengl": [r"gl_ClipDistance\[1\] = ", r"gl_CullDistance\[2\] = "],
        "vulkan": [r"BuiltIn ClipDistance", r"BuiltIn CullDistance"],
        "directx": [r": SV_ClipDistance1", r": SV_CullDistance1"],
    },
    "fragment_outputs": {
        "opengl": [
            r"layout\(location = 3\) out vec4 a;",
            r"layout\(location = 1\) out vec4 b;",
            r"gl_FragDepth = ",
            r"gl_FragStencilRefARB = int\(7u\)",
        ],
        "vulkan": [
            r"%CrossGL_fragment_output_main_a Location 3",
            r"%CrossGL_fragment_output_main_b Location 1",
            r"BuiltIn FragDepth",
            r"BuiltIn FragStencilRefEXT",
        ],
        "metal": [
            r"\[\[color\(3\)\]\]",
            r"\[\[color\(1\)\]\]",
            r"\[\[depth\(any\)\]\]",
            r"\[\[stencil\]\]",
        ],
        "directx": [r": SV_Target3", r": SV_DepthGreaterEqual", r": SV_StencilRef"],
    },
    "fragment_inputs": {
        "opengl": [
            r"\bgl_Layer\b",
            r"\bgl_ViewportIndex\b",
            r"\bgl_BaryCoordEXT\b",
            r"\bgl_BaryCoordNoPerspEXT\b",
            r"uint inner = uint\(gl_FragFullyCoveredNV\);",
        ],
        "vulkan": [
            r"BuiltIn Layer",
            r"BuiltIn ViewportIndex",
            r"BuiltIn BaryCoordKHR",
            r"BuiltIn BaryCoordNoPerspKHR",
            r"BuiltIn FullyCoveredEXT",
        ],
        "directx": [r": SV_Barycentrics1", r": SV_InnerCoverage"],
    },
    "vertex_inputs": {
        "opengl": [r"\bgl_BaseVertexARB\b", r"\bgl_BaseInstanceARB\b"],
        "vulkan": [r"BuiltIn BaseVertex", r"BuiltIn BaseInstance"],
        "metal": [r"\[\[base_vertex\]\]", r"\[\[base_instance\]\]"],
        "directx": [r": SV_StartVertexLocation", r": SV_StartInstanceLocation"],
    },
    "view_index": {
        "vulkan": [r"BuiltIn ViewIndex"],
        "metal": [r"\[\[amplification_id\]\]"],
        "directx": [r": SV_ViewID"],
    },
}

PROGRAMS = {
    "vertex_outputs": (VERTEX_OUTPUTS, "graphics"),
    "distances": (DISTANCES, "vertex"),
    "fragment_outputs": (FRAGMENT_OUTPUTS, "fragment"),
    "fragment_inputs": (FRAGMENT_INPUTS, "fragment"),
    "vertex_inputs": (VERTEX_INPUTS, "vertex"),
    "view_index": (VIEW_INDEX, "vertex"),
}

# Values the target has no built-in for, refused with the generator's typed diagnostic.
REFUSED = {
    ("distances", "metal"): "clip distance semantic index 1",
    ("fragment_inputs", "metal"): "gl_FullyCoveredEXT",
    ("view_index", "opengl"): "multiview view index",
}

# Native outputs that must not declare these values as user interface variables.
USER_VARYINGS = {
    "opengl": re.compile(
        r"^\s*(?:layout\([^)]*\)\s*)?(?:flat\s+)?(?:in|out)\s+\w+\s+"
        r"(?:clip|layer|viewport|depth|stencil|inner|base_vertex|base_instance|view|cull\d)\b",
        re.M,
    ),
    "vulkan": re.compile(
        r"OpDecorate %CrossGL_\w+_(?:clip|layer|viewport|depth|stencil|inner|base_vertex|base_instance|view|cull\d)"
        r"\w* Location"
    ),
    "metal": re.compile(
        r"\b(?:clip|layer|viewport|depth|stencil|inner|base_vertex|base_instance|view)\s*\[\[(?:user|attribute)\("
    ),
}

_HLSL_PROFILES = {
    "graphics": [("vs_6_8", "VSMain"), ("ps_6_8", "PSMain")],
    "vertex": [("vs_6_8", "VSMain")],
    "fragment": [("ps_6_8", "PSMain")],
}
_GLSL_STAGES = {
    "graphics": ("vert", "frag"),
    "vertex": ("vert",),
    "fragment": ("frag",),
}


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _view(backend, code):
    """``code`` as the assertions read it: SPIR-V ids by their ``OpName``, HLSL semantics as ``name : SEMANTIC``."""
    if backend == "vulkan":
        names = dict(re.findall(r'OpName (%\d+) "(\w+)"', code))
        return re.sub(
            r"%\d+\b", lambda m: "%" + names.get(m.group(0), m.group(0)[1:]), code
        )
    if backend == "directx":
        return re.sub(r"(\w)\s*:\s*(SV_\w+)", r"\1 : \2", code)
    return code


def _run(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]


def _metal_available():
    if shutil.which("xcrun") is None:
        return False
    found = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True
    )
    return found.returncode == 0


def _validate(tmp_path, backend, code, stages):
    """Compile ``code`` with the target's native checker, when it is installed."""
    if backend == "opengl" and shutil.which("glslangValidator"):
        for stage in _GLSL_STAGES[stages]:
            _run(["glslangValidator", "--stdin", "-S", stage], input=code)
    elif backend == "vulkan" and shutil.which("spirv-as") and shutil.which("spirv-val"):
        asm, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
        asm.write_text(code)
        _run(["spirv-as", "--target-env", "vulkan1.1", str(asm), "-o", str(binary)])
        _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])
    elif backend == "metal" and _metal_available():
        source = tmp_path / "out.metal"
        source.write_text(code)
        _run(
            ["xcrun", "-sdk", "macosx", "metal", "-c", str(source)]
            + ["-o", str(tmp_path / "out.air")]
        )
    elif backend == "directx" and shutil.which("dxc"):
        source = tmp_path / "out.hlsl"
        source.write_text(code)
        for profile, entry in _HLSL_PROFILES[stages]:
            _run(["dxc", "-T", profile, "-E", entry, str(source)])


@pytest.mark.parametrize("backend", ["opengl", "vulkan", "metal", "directx"])
@pytest.mark.parametrize("program", sorted(PROGRAMS))
def test_system_values_reach_each_target_as_built_ins(tmp_path, program, backend):
    source, stages = PROGRAMS[program]
    refusal = REFUSED.get((program, backend))
    if refusal is not None:
        with pytest.raises(Exception, match=refusal) as caught:
            _translate(tmp_path, source, backend)
        assert getattr(caught.value, "code", "").startswith("project.translate.")
        return
    code = _translate(tmp_path, source, backend)
    code_view = _view(backend, code)
    for pattern in EXPECTED[program].get(backend, []):
        assert re.search(pattern, code_view), (pattern, code[-3000:])
    user = USER_VARYINGS.get(backend)
    if user is not None:
        assert not user.search(code_view), user.search(code_view).group(0)
    _validate(tmp_path, backend, code, stages)


@pytest.mark.parametrize(
    "spelling, reason",
    [
        ("SV_Fancy", "is not a Direct3D system value"),
        ("SV_Position1", "SV_Position takes no semantic index"),
        ("SV_Target8", "SV_Target takes semantic indices 0 to 7"),
        ("SV_StencilRef1", "SV_StencilRef takes no semantic index"),
        ("SV_Barycentrics2", "SV_Barycentrics takes semantic indices 0 to 1"),
    ],
)
@pytest.mark.parametrize("backend", ["opengl", "vulkan", "metal", "directx"])
def test_undefined_system_values_are_refused_on_every_target(
    tmp_path, spelling, reason, backend
):
    source = f"""shader S {{
struct FOut {{ vec4 color @ SV_Target0; vec4 extra @ {spelling}; }};
fragment {{
    FOut main(vec2 uv @ TEXCOORD0) {{
        FOut o;
        o.color = vec4(uv, 0.0, 1.0);
        o.extra = vec4(1.0);
        return o;
    }}
}}
}}
"""
    with pytest.raises(BuiltinValueUnsupportedError, match=reason):
        _translate(tmp_path, source, backend)


SHARED_SHADING_RATE = """shader S {
struct VOut { vec4 pos @ SV_Position; uint rate @ SV_ShadingRate; };
struct FOut { vec4 color @ SV_Target; };
vertex {
    VOut main(vec3 p @ POSITION) {
        VOut o;
        o.pos = vec4(p, 1.0);
        o.rate = 4u;
        return o;
    }
}
fragment {
    FOut main(VOut i) {
        FOut o;
        o.color = vec4(float(i.rate));
        return o;
    }
}
}
"""


def test_a_shading_rate_shared_by_two_stages_resolves_per_stage_in_spirv(tmp_path):
    code = _translate(tmp_path, SHARED_SHADING_RATE, "vulkan")
    assert "BuiltIn PrimitiveShadingRateKHR" in code
    assert "BuiltIn ShadingRateKHR" in code
    _validate(tmp_path, "vulkan", code, "graphics")


def test_a_shading_rate_shared_by_two_stages_is_refused_for_glsl(tmp_path):
    with pytest.raises(
        BuiltinValueUnsupportedError, match="gl_PrimitiveShadingRateEXT"
    ):
        _translate(tmp_path, SHARED_SHADING_RATE, "opengl")


def test_a_fragment_shading_rate_input_reads_the_built_in(tmp_path):
    source = """shader S {
fragment {
    vec4 main(uint rate @ SV_ShadingRate) @ SV_Target {
        return vec4(float(rate + 1u));
    }
}
}
"""
    code = _translate(tmp_path, source, "opengl")
    assert "uint rate = uint(gl_ShadingRateEXT);" in code
    assert not USER_VARYINGS["opengl"].search(code)
    spirv = _translate(tmp_path, source, "vulkan")
    assert "BuiltIn ShadingRateKHR" in spirv
    _validate(tmp_path, "vulkan", spirv, "fragment")


def test_hlsl_system_values_without_crossgl_names_import_as_built_ins(tmp_path):
    """The HLSL importer keeps the semantics it has no CrossGL name for; GLSL and SPIR-V still read the built-ins."""
    path = tmp_path / "shader.hlsl"
    path.write_text(
        "float4 main(float4 pos : SV_Position, uint rate : SV_ShadingRate,\n"
        "            uint inner : SV_InnerCoverage) : SV_Target {\n"
        "    return float4(pos.xy, float(rate), float(inner + 1));\n"
        "}\n"
    )
    glsl = crosstl.translate(str(path), backend="opengl", format_output=False)
    assert "flat in uint" not in glsl
    assert "uint(gl_FragFullyCoveredNV)" in glsl
    assert "uint(gl_ShadingRateEXT)" in glsl
    _validate(tmp_path, "opengl", glsl, "fragment")
    spirv = crosstl.translate(str(path), backend="vulkan", format_output=False)
    assert "BuiltIn FullyCoveredEXT" in spirv and "BuiltIn ShadingRateKHR" in spirv
    assert not re.search(
        r"OpDecorate %CrossGL_fragment_input_\w+ Location", _view("vulkan", spirv)
    )
    _validate(tmp_path, "vulkan", spirv, "fragment")


@pytest.mark.parametrize(
    "spelling, name, index",
    [
        ("SV_Target", "SV_Target", None),
        ("sv_target7", "SV_Target", 7),
        ("SV_ClipDistance12", "SV_ClipDistance", 12),
        ("sv_group_thread_id", "SV_GroupThreadID", None),
        ("SV_StartInstanceLocation", "SV_StartInstanceLocation", None),
    ],
)
def test_lookup_reads_case_index_and_snake_case(spelling, name, index):
    value, found_index = lookup_system_value(spelling)
    assert (value.name, found_index) == (name, index)


def test_lookup_ignores_other_attributes():
    assert lookup_system_value("TEXCOORD0") is None
    assert lookup_system_value("gl_Position") is None
    with pytest.raises(SystemValueError):
        lookup_system_value("SV_")
