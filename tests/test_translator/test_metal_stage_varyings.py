"""Metal links a vertex output to a fragment input by the key the source language links them by.

Metal matches fragment ``[[stage_in]]`` members to vertex outputs by ``[[user(KEY)]]`` and otherwise by member name;
``[[attribute(n)]]`` is ignored there, and a loose fragment parameter cannot take it.  GLSL links by location and
HLSL by semantic, so emitting ``[[attribute(n)]]`` on varyings silently fell back to name matching, which fails at
pipeline creation whenever the two stages name a value differently, and loose HLSL pixel-shader inputs crashed the
Metal compiler.  See :mod:`crosstl.translator.codegen.metal_varyings`.
"""

import platform
import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.metal_varyings import (
    MetalVaryingLinkError,
    varying_user_key,
)

GLSL_VERTEX = """#version 450
layout(location = 0) in vec3 aPos;
layout(location = 0) out vec3 vColor;
layout(location = 1) out vec2 vUv;
void main() { vColor = aPos; vUv = aPos.xy; gl_Position = vec4(aPos, 1.0); }
"""

GLSL_FRAGMENT = """#version 450
layout(location = 1) in vec2 uvIn;
layout(location = 0) in vec3 colorIn;
layout(location = 0) out vec4 fragColor;
void main() { fragColor = vec4(colorIn * uvIn.x, 1.0); }
"""

HLSL_VERTEX = """struct VSOut { float4 pos : SV_Position; float3 color : COLOR; float2 uv : TEXCOORD2; };
VSOut VSMain(float3 pos : POSITION) {
    VSOut o;
    o.pos = float4(pos, 1.0);
    o.color = pos;
    o.uv = pos.xy;
    return o;
}
"""

HLSL_PIXEL = """float4 PSMain(float2 t : TEXCOORD2, float3 c : COLOR0) : SV_Target {
    return float4(c * t.x, 1.0);
}
"""

HLSL_SYSTEM_VALUES = """struct VSOut {
    float4 pos : SV_Position;
    uint layer : SV_RenderTargetArrayIndex;
    nointerpolation uint id : BLENDINDICES;
};
VSOut VSMain(uint vid : SV_VertexID) {
    VSOut o;
    o.pos = float4(0.0, 0.0, 0.0, 1.0);
    o.layer = 0;
    o.id = vid;
    return o;
}
float4 PSMain(VSOut i, bool front : SV_IsFrontFace) : SV_Target {
    return front ? float4(float(i.id), 0.0, 0.0, 1.0) : float4(0.0, 0.0, 0.0, 1.0);
}
"""

HLSL_CULL_DISTANCE = """struct VSOut { float4 pos : SV_Position; float cull : SV_CullDistance0; };
VSOut VSMain(float3 p : POSITION) { VSOut o; o.pos = float4(p, 1.0); o.cull = 1.0; return o; }
"""

CROSSGL_LOOSE_FRAGMENT = """shader S {
    fragment {
        vec4 main(vec2 uv, vec3 c @location(2), vec3 d @ TEXCOORD1, vec4 fc @ gl_FragCoord) @ gl_FragColor {
            return vec4(uv, c.x + d.y + fc.x, 1.0);
        }
    }
}
"""

CROSSGL_MULTI_SLOT = """shader S {
    struct VOut { vec4 pos @ gl_Position; mat3 m @location(2); float w[2] @location(5); vec2 n; };
    struct FIn { mat3 m @location(2); float w[2] @location(5); vec2 n; };
    vertex {
        VOut main(vec3 p @location(0)) {
            VOut o;
            o.pos = vec4(p, 1.0);
            o.m = mat3(1.0);
            o.w[0] = 1.0;
            o.w[1] = 2.0;
            o.n = p.xy;
            return o;
        }
    }
    fragment {
        vec4 main(FIn i) @ gl_FragColor { return vec4(i.m[1] * i.w[1], i.n.x); }
    }
}
"""


def translate(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend="metal")


def struct_body(code, name):
    match = re.search(rf"struct {re.escape(name)} \{{(.*?)\}};", code, re.S)
    assert match, f"struct {name} not emitted"
    return match.group(1)


def entry_parameters(code, qualifier, name):
    match = re.search(rf"\b{qualifier}\s+\w+\s+{re.escape(name)}\((.*?)\)\s*\{{", code, re.S)
    assert match, f"{qualifier} {name} not emitted"
    return match.group(1)


@pytest.mark.parametrize(
    "semantic, kwargs, key",
    [
        ("TexCoord", {}, "TEXCOORD0"),
        ("TexCoord3", {}, "TEXCOORD3"),
        ("Color", {}, "COLOR0"),
        ("MyValue7", {}, "MYVALUE7"),
        ("TEXCOORD1", {"offset": 2}, "TEXCOORD3"),
        ("TexCoord", {"location": 4}, "locn4"),
        (None, {"location": 4, "offset": 1}, "locn5"),
        (None, {"user_name": "locn0"}, "locn0"),
        (None, {"user_name": "tint", "offset": 1}, "tint_1"),
        (None, {}, None),
    ],
)
def test_user_keys_follow_the_source_linkage(semantic, kwargs, key):
    assert varying_user_key(semantic, **kwargs) == key


@pytest.mark.parametrize("semantic", ["SV_CullDistance0", "gl_CullDistance", "a(b)"])
def test_system_values_and_malformed_semantics_are_refused(semantic):
    with pytest.raises(MetalVaryingLinkError):
        varying_user_key(semantic)


def test_glsl_varyings_link_by_location(tmp_path):
    vertex = translate(tmp_path, "a.vert", GLSL_VERTEX)
    fragment = translate(tmp_path, "a.frag", GLSL_FRAGMENT)
    assert "float3 vColor [[user(locn0)]];" in vertex
    assert "float2 vUv [[user(locn1)]];" in vertex
    assert "float2 uvIn [[user(locn1)]];" in fragment
    assert "float3 colorIn [[user(locn0)]];" in fragment
    assert "float3 aPos [[attribute(0)]];" in vertex


def test_hlsl_varyings_link_by_semantic_and_pixel_inputs_use_stage_in(tmp_path):
    vertex = translate(tmp_path, "b.vs.hlsl", HLSL_VERTEX)
    fragment = translate(tmp_path, "b.ps.hlsl", HLSL_PIXEL)
    outputs = struct_body(vertex, "VSOut")
    assert "float3 color [[user(COLOR0)]];" in outputs
    assert "float2 uv [[user(TEXCOORD2)]];" in outputs
    parameters = entry_parameters(fragment, "fragment", "PSMain")
    assert parameters.count("[[stage_in]]") == 1
    assert "[[attribute(" not in parameters
    inputs = struct_body(fragment, "PSMain_Input")
    assert "float2 t [[user(TEXCOORD2)]];" in inputs
    assert "float3 c [[user(COLOR0)]];" in inputs
    assert "[[attribute(" not in inputs


def test_loose_fragment_inputs_are_gathered_into_one_stage_in_struct(tmp_path):
    code = translate(tmp_path, "loose.cgl", CROSSGL_LOOSE_FRAGMENT)
    parameters = entry_parameters(code, "fragment", "fragment_main")
    assert parameters.count("[[stage_in]]") == 1
    assert "float4 fc [[position]]" in parameters
    inputs = struct_body(code, "fragment_main_Input")
    assert re.search(r"float2 uv;", inputs), "an unkeyed input links by name"
    assert "float3 c [[user(locn2)]];" in inputs
    assert "float3 d [[user(TEXCOORD1)]];" in inputs


def test_system_values_keep_their_metal_builtins(tmp_path):
    code = translate(tmp_path, "sv.hlsl", HLSL_SYSTEM_VALUES)
    outputs = struct_body(code, "VSOut")
    assert "uint layer [[render_target_array_index]];" in outputs
    assert "uint id [[user(BLENDINDICES0)]] [[flat]];" in outputs
    assert "uint vid [[vertex_id]]" in entry_parameters(code, "vertex", "VSMain")
    assert "VSMain_Input" not in code, "[[vertex_id]] must stay a parameter"
    assert "bool front [[front_facing]]" in entry_parameters(code, "fragment", "PSMain")
    assert "is_front_facing" not in code


def test_system_value_without_a_metal_builtin_is_refused(tmp_path):
    with pytest.raises(MetalVaryingLinkError, match="CullDistance"):
        translate(tmp_path, "cull.hlsl", HLSL_CULL_DISTANCE)


def test_multi_slot_varyings_take_consecutive_keys(tmp_path):
    code = translate(tmp_path, "slots.cgl", CROSSGL_MULTI_SLOT)
    for struct in ("VOut", "FIn"):
        body = struct_body(code, struct)
        for index, key in enumerate(("locn2", "locn3", "locn4")):
            assert f"float3 m_{index} [[user({key})]];" in body
        assert "float w_0 [[user(locn5)]];" in body
        assert "float w_1 [[user(locn6)]];" in body
        assert re.search(r"float2 n;", body)


def metal_compiler():
    if platform.system() != "Darwin" or shutil.which("xcrun") is None:
        return None
    probe = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.stdout.strip() if probe.returncode == 0 else None


NATIVE_SOURCES = {
    "a.vert": GLSL_VERTEX,
    "a.frag": GLSL_FRAGMENT,
    "b.vs.hlsl": HLSL_VERTEX,
    "b.ps.hlsl": HLSL_PIXEL,
    "sv.hlsl": HLSL_SYSTEM_VALUES,
    "loose.cgl": CROSSGL_LOOSE_FRAGMENT,
    "slots.cgl": CROSSGL_MULTI_SLOT,
}


@pytest.mark.parametrize("name", sorted(NATIVE_SOURCES))
def test_translations_compile_with_the_metal_compiler(tmp_path, name):
    compiler = metal_compiler()
    if compiler is None:
        pytest.skip("the Metal compiler is only available on macOS with Xcode")
    code = translate(tmp_path, name, NATIVE_SOURCES[name])
    source = tmp_path / "out.metal"
    source.write_text(code)
    result = subprocess.run(
        [compiler, "-c", str(source), "-o", str(tmp_path / "out.air")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]


def link_render_pipeline(source, vertex_name, fragment_name):
    """Create a render pipeline from two functions in ``source``; returns the driver's error or None."""
    Metal = pytest.importorskip("Metal")
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")
    library, error = device.newLibraryWithSource_options_error_(source, None, None)
    assert library is not None, str(error)
    descriptor = Metal.MTLRenderPipelineDescriptor.new()
    vertex = library.newFunctionWithName_(vertex_name)
    fragment = library.newFunctionWithName_(fragment_name)
    assert vertex is not None and fragment is not None
    descriptor.setVertexFunction_(vertex)
    descriptor.setFragmentFunction_(fragment)
    descriptor.colorAttachments().objectAtIndexedSubscript_(0).setPixelFormat_(
        Metal.MTLPixelFormatRGBA8Unorm
    )
    formats = {
        Metal.MTLDataTypeFloat: Metal.MTLVertexFormatFloat,
        Metal.MTLDataTypeFloat2: Metal.MTLVertexFormatFloat2,
        Metal.MTLDataTypeFloat3: Metal.MTLVertexFormatFloat3,
        Metal.MTLDataTypeFloat4: Metal.MTLVertexFormatFloat4,
    }
    attributes = [a for a in (vertex.vertexAttributes() or []) if a.isActive()]
    if attributes:
        layout = Metal.MTLVertexDescriptor.vertexDescriptor()
        for slot, attribute in enumerate(attributes):
            entry = layout.attributes().objectAtIndexedSubscript_(
                attribute.attributeIndex()
            )
            entry.setFormat_(formats[attribute.attributeType()])
            entry.setOffset_(16 * slot)
            entry.setBufferIndex_(30)
        layout.layouts().objectAtIndexedSubscript_(30).setStride_(16 * len(attributes))
        descriptor.setVertexDescriptor_(layout)
    state, error = device.newRenderPipelineStateWithDescriptor_error_(descriptor, None)
    return None if state is not None else str(error)


def one_library(*codes):
    head, *rest = codes
    lines = [
        line
        for code in rest
        for line in code.splitlines()
        if not line.startswith(("#include", "using namespace"))
    ]
    return head + "\n" + "\n".join(lines)


@pytest.mark.parametrize(
    "vertex, fragment, vertex_name, fragment_name",
    [
        ("a.vert", "a.frag", "vertex_main", "fragment_main"),
        ("b.vs.hlsl", "b.ps.hlsl", "VSMain", "PSMain"),
    ],
)
def test_separately_translated_stages_link_into_a_pipeline(
    tmp_path, vertex, fragment, vertex_name, fragment_name
):
    if platform.system() != "Darwin":
        pytest.skip("Metal pipelines are only available on macOS")
    source = one_library(
        translate(tmp_path, vertex, NATIVE_SOURCES[vertex]),
        translate(tmp_path, fragment, NATIVE_SOURCES[fragment]),
    )
    assert link_render_pipeline(source, vertex_name, fragment_name) is None
