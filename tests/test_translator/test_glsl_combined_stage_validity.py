"""Combined vertex and fragment GLSL from MSL that every stage of the file compiles.

The GLSL emitters put all stages in one file under ``#ifdef GL_<STAGE>_SHADER`` guards, and the file is compiled
once per stage. These cases each produced a file that glslang rejected for one of the stages.
"""

import shutil
import subprocess

import pytest

import crosstl

HEADER = "#include <metal_stdlib>\nusing namespace metal;\n"


def translate(tmp_path, source, backend="opengl"):
    path = tmp_path / "shader.metal"
    path.write_text(HEADER + source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def compile_stages(tmp_path, glsl, prelude=""):
    if shutil.which("glslangValidator") is None:
        return
    if prelude:
        version, _, rest = glsl.partition("\n")
        glsl = f"{version}\n{prelude}{rest}"
    path = tmp_path / "combined.glsl"
    path.write_text(glsl)
    for stage in ("vert", "frag"):
        result = subprocess.run(
            ["glslangValidator", "-S", stage, str(path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, stage + "\n" + result.stdout + glsl


def test_written_half_stage_input_members_are_copied(tmp_path):
    # f16vec3 is a plain value, so a fragment that writes in.tangent writes a local copy, not the `in` variable.
    glsl = translate(
        tmp_path,
        """
struct Vertex { float3 position [[attribute(0)]]; half3 tangent [[attribute(1)]]; };
struct ColorInOut { float4 position [[position]]; half3 tangent; };
vertex ColorInOut vmain(Vertex in [[stage_in]]) {
    ColorInOut out; out.position = float4(in.position, 1.0); out.tangent = in.tangent; return out;
}
fragment float4 fmain(ColorInOut in [[stage_in]]) {
    in.tangent = normalize(in.tangent);
    return float4(float3(in.tangent), 1.0);
}
""",
    )
    fragment = glsl.split("GL_FRAGMENT_SHADER")[-1]
    assert "in_copy.tangent = normalize(in_copy.tangent);" in fragment
    assert "out_tangent = normalize" not in fragment
    compile_stages(tmp_path, glsl)


def test_varyings_avoid_the_names_of_entry_buffers(tmp_path):
    # The buffer becomes the unnamed-block member `color`, so the varying of the same name is renamed.
    glsl = translate(
        tmp_path,
        """
struct ColorInOut { float4 position [[position]]; float4 color; };
vertex ColorInOut vmain(uint vid [[vertex_id]], const device float4* position [[buffer(0)]],
                        const device float4* color [[buffer(1)]]) {
    ColorInOut out; out.position = position[vid]; out.color = color[vid]; return out;
}
fragment float4 fmain(ColorInOut in [[stage_in]]) { return in.color; }
""",
    )
    assert "out vec4 out_color;" in glsl
    assert "in vec4 out_color;" in glsl
    assert "fragColor = out_color;" in glsl
    compile_stages(tmp_path, glsl)


def test_fragment_only_helpers_are_guarded_to_the_fragment_stage(tmp_path):
    # fwidth and a clamped implicit-LOD sample exist in fragment shaders only.
    glsl = translate(
        tmp_path,
        """
struct V { float4 position [[position]]; float2 uv; };
float edge(float2 uv) { return fwidth(uv.x); }
float shared_scale(float x) { return x * 2.0; }
float4 clamped(texture2d<float> t, sampler s, float2 uv, float lod) {
    return t.sample(s, uv, min_lod_clamp(lod));
}
vertex V vmain(uint vid [[vertex_id]]) {
    V o; o.position = float4(shared_scale(1.0)); o.uv = float2(0.0); return o;
}
fragment float4 fmain(V in [[stage_in]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]]) {
    return clamped(t, s, in.uv, 1.0) + edge(in.uv) + shared_scale(in.uv.y);
}
""",
    )
    edge = glsl.index("float edge(vec2 uv) {")
    assert glsl.rfind("#ifdef GL_FRAGMENT_SHADER", 0, edge) > glsl.rfind(
        "#endif", 0, edge
    )
    shared = glsl.index("float shared_scale(float x) {")
    assert glsl.rfind("#ifdef", 0, shared) < glsl.rfind("#endif", 0, shared)
    compile_stages(tmp_path, glsl)


def test_helper_guards_are_absent_from_single_stage_output(tmp_path):
    glsl = translate(
        tmp_path,
        """
float edge(float2 uv) { return fwidth(uv.x); }
fragment float4 fmain(float4 p [[position]]) { return float4(edge(p.xy)); }
""",
    )
    assert "#ifdef GL_FRAGMENT_SHADER" not in glsl


FUNCTION_CONSTANTS = """
constant bool has_map [[function_constant(0)]];
constant int count_c [[function_constant(1)]];
struct V { float4 position [[position]]; };
vertex V vmain(uint vid [[vertex_id]]) { V o; o.position = float4(0.0); return o; }
fragment float4 fmain(V in [[stage_in]]) {
    if (has_map) { return float4(float(count_c)); }
    return float4(0.0);
}
"""


def test_required_function_constants_read_a_host_macro_from_source(tmp_path):
    glsl = translate(tmp_path, FUNCTION_CONSTANTS)
    assert "layout(constant_id = 0) const bool has_map = false;" in glsl
    assert "#elif defined(CROSSGL_SPECIALIZE_has_map)" in glsl
    assert "const bool has_map = bool(CROSSGL_SPECIALIZE_has_map);" in glsl
    assert "const int count_c = int(CROSSGL_SPECIALIZE_count_c);" in glsl
    assert "source_macro=CROSSGL_SPECIALIZE_count_c" in glsl
    assert '#error "specialization constant has_map has no default' in glsl
    compile_stages(
        tmp_path,
        glsl,
        "#define CROSSGL_SPECIALIZE_has_map 1\n#define CROSSGL_SPECIALIZE_count_c 3\n",
    )


def test_required_function_constants_fail_without_a_value(tmp_path):
    if shutil.which("glslangValidator") is None:
        pytest.skip("glslangValidator is not installed")
    glsl = translate(tmp_path, FUNCTION_CONSTANTS)
    path = tmp_path / "unspecialized.glsl"
    path.write_text(glsl)
    result = subprocess.run(
        ["glslangValidator", "-S", "frag", str(path)], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "define CROSSGL_SPECIALIZE_has_map" in result.stdout
