"""MSL texture member functions have the result types MSL 3.2 section 6.12 gives them.

Two importer gaps lost those types:

- ``expression_metal_type`` had no rule for ``tex.sample(...)`` or ``tex.read(...)``/``gather``/``sample_compare``,
  so a math builtin that took one, as in ``smoothstep(0.5, 0.7, tex.sample(s, uv).x) * 0.2``, could not pick its
  overload and the import was refused.
- A depth texture's ``sample`` and ``read`` return the scalar depth, but the CrossGL ``texture``/``texelFetch``
  family returns ``vec4(depth, 0, 0, 1)``. ``float d = depthMap.sample(s, uv);`` imported as a ``vec4`` assigned to
  a ``float``, which every generator rejected or emitted as invalid code. The importer now reads component ``x``.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.backend.Metal.MetalCrossGLCodeGen import (
    MetalBuiltinResultTypeResolutionError,
)
from crosstl.backend.Metal.texture_method_results import (
    is_depth_texture_name,
    texture_method_result_shape,
)

BUILTIN_ARGUMENTS = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;

    fragment float4 shade(float4 p [[position]],
                          texture2d<float> color [[texture(0)]],
                          depth2d<float> depth [[texture(1)]],
                          sampler s [[sampler(0)]]) {
        float a = smoothstep(0.5, 0.7, color.sample(s, p.xy, level(0)).x) * 0.2;
        float b = abs(color.read(uint2(p.xy)).y) * 0.5;
        float c = saturate(color.gather(s, p.xy).z) * 0.25;
        float e = max(depth.sample_compare(s, p.xy, 0.5), 0.25) * 2.0;
        return float4(a, b, c, e);
    }
    """)

DEPTH_READS = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;

    fragment float4 shade(float4 p [[position]],
                          texture2d<float> color [[texture(0)]],
                          depth2d<float> depth [[texture(1)]],
                          depth2d_array<float> cascades [[texture(2)]],
                          depthcube<float> cube [[texture(3)]],
                          sampler s [[sampler(0)]]) {
        float sampled = depth.sample(s, p.xy);
        float loaded = depth.read(uint2(p.xy));
        float level0 = depth.sample(s, p.xy, level(1.0));
        float layered = cascades.sample(s, p.xy, 1);
        float faced = abs(cube.sample(s, p.xyz)) * 0.5;
        float4 tint = color.sample(s, p.xy);
        return tint * (sampled + loaded + level0 + layered + faced);
    }
    """)


def _translate(tmp_path, source, backend):
    path = tmp_path / "textures.metal"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(tmp_path, *command):
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=120, cwd=tmp_path
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]


@pytest.mark.parametrize(
    ("base_name", "generic_args", "method", "expected"),
    [
        ("texture2d", ["float"], "sample", ("float", 4)),
        ("texture2d", ["half", "access::read"], "read", ("half", 4)),
        ("texturecube_array", ["metal::float"], "gather", ("float", 4)),
        ("texture2d_ms", ["uint"], "read", ("uint", 4)),
        ("texture_buffer", ["int", "access::read"], "read", ("int", 4)),
        ("depth2d", ["float"], "sample", ("float", 1)),
        ("depth2d_array", ["float"], "sample_compare", ("float", 1)),
        ("depth2d_ms", ["float"], "read", ("float", 1)),
        ("depthcube", ["float"], "gather_compare", ("float", 4)),
        ("depth2d", ["float"], "gather", ("float", 4)),
        ("texture3d", ["float"], "get_depth", ("uint", 1)),
        ("depth2d", ["float"], "get_num_mip_levels", ("uint", 1)),
        ("texture2d", ["float"], "calculate_clamped_lod", ("float", 1)),
        ("texture2d", ["float"], "sample_compare", None),
        ("texture2d", ["float"], "write", None),
        ("texture2d", [], "sample", None),
        ("SomeStruct", ["float"], "sample", None),
        ("SomeStruct", ["float"], "get_width", None),
    ],
)
def test_texture_method_result_shapes(base_name, generic_args, method, expected):
    assert texture_method_result_shape(base_name, generic_args, method) == expected


def test_depth_texture_names():
    assert is_depth_texture_name("depthcube_array")
    assert is_depth_texture_name("depth2d_ms")
    assert not is_depth_texture_name("texture2d")
    assert not is_depth_texture_name(None)


def test_builtins_resolve_overloads_of_texture_results(tmp_path):
    crossgl = _translate(tmp_path, BUILTIN_ARGUMENTS, "cgl")

    assert "smoothstep(0.5, 0.7, textureLod(color, s, p.xy, 0).x) * 0.2" in crossgl
    assert "abs(texelFetch(color, uvec2(p.xy), 0).y) * 0.5" in crossgl
    assert "textureCompare(depth, s, p.xy, 0.5)" in crossgl


def test_ambiguous_builtin_on_half_texel_stays_refused(tmp_path):
    # Metal rejects smoothstep(double, double, half) as ambiguous between the half and float overloads.
    source = BUILTIN_ARGUMENTS.replace(
        "texture2d<float> color", "texture2d<half> color"
    )
    with pytest.raises(MetalBuiltinResultTypeResolutionError, match="smoothstep"):
        _translate(tmp_path, source, "cgl")


def test_depth_reads_import_as_the_depth_component(tmp_path):
    crossgl = _translate(tmp_path, DEPTH_READS, "cgl")

    assert "float sampled = texture(depth, s, p.xy).x;" in crossgl
    assert "float loaded = texelFetch(depth, uvec2(p.xy), 0).x;" in crossgl
    assert "float level0 = textureLod(depth, s, p.xy, 1.0).x;" in crossgl
    assert "float layered = texture(cascades, s, vec3(p.xy, 1)).x;" in crossgl
    assert "abs(texture(cube, s, p.xyz).x) * 0.5" in crossgl
    assert "vec4 tint = texture(color, s, p.xy);" in crossgl


def test_depth_reads_compile_as_metal(tmp_path):
    metal = _translate(tmp_path, DEPTH_READS, "metal")

    assert "float sampled = float4(depth.sample(s, p.xy), 0.0, 0.0, 1.0).x;" in metal
    if not shutil.which("xcrun"):
        pytest.skip("xcrun is not installed")
    (tmp_path / "a.metal").write_text(metal, encoding="utf-8")
    _run(tmp_path, "xcrun", "-sdk", "macosx", "metal", "-c", "a.metal", "-o", "a.air")


def test_depth_reads_compile_as_glsl(tmp_path):
    glsl = _translate(tmp_path, DEPTH_READS, "opengl")

    assert ").x;" in glsl
    if not shutil.which("glslangValidator"):
        pytest.skip("glslangValidator is not installed")
    (tmp_path / "a.frag").write_text(glsl, encoding="utf-8")
    _run(tmp_path, "glslangValidator", "a.frag")


def test_depth_reads_select_the_depth_in_hlsl(tmp_path):
    hlsl = _translate(tmp_path, DEPTH_READS, "directx")

    assert "float sampled = depth.Sample(s, p.xy).x;" in hlsl
    assert "float loaded = depth.Load(int3(uint2(p.xy), 0)).x;" in hlsl


def test_depth_reads_validate_as_spirv(tmp_path):
    spirv = _translate(tmp_path, DEPTH_READS, "vulkan")

    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools is not installed")
    (tmp_path / "a.spvasm").write_text(spirv, encoding="utf-8")
    _run(tmp_path, "spirv-as", "a.spvasm", "-o", "a.spv")
    _run(tmp_path, "spirv-val", "a.spv")
