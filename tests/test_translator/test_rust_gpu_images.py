"""CrossGL images and samplers in the rust-gpu output (:mod:`crosstl.translator.codegen.rust_gpu_images`).

An image is a ``spirv_std::Image!`` entry-point binding and a sampler a ``Sampler``; each CrossGL image built-in is
the spirv-std method that issues the same SPIR-V instruction. A combined sampler is split into an image and a sampler
bound beside it, and a storage image's format becomes part of its type, so helpers that receive it name the same
type. Each program is translated as ``crosstl.translate`` translates it (the CrossGL semantics passes included), and
each generated crate compiles with rust-gpu when it is built.
"""

from __future__ import annotations

import pytest

import crosstl
from crosstl.translator.codegen.rust_refusals import RustLoweringRefused

from .test_codegen.test_rust_codegen import assert_generated_rust_smoke_compiles


def _kernel(declarations, body, helpers=""):
    return f"""
shader Probe {{
    compute {{
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        {declarations}
        {helpers}
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            {body}
        }}
    }}
}}
"""


def _translate(source, tmp_path, format_output=False):
    path = tmp_path / "probe.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="rust", format_output=format_output)


def _flat(code):
    return " ".join(code.split())


LOWERINGS = [
    pytest.param(
        "uniform sampler2D tex;",
        "outp[i] = textureLod(tex, vec2(0.5, 0.5), 0.0).x;",
        "",
        [
            "#[spirv(descriptor_set = 0, binding = 1)] tex: &Image!(2D, type=f32, sampled)",
            "#[spirv(descriptor_set = 0, binding = 2)] tex_sampler: &Sampler",
            "tex.sample_by_lod(*tex_sampler, Vec2::new(0.5, 0.5), 0.0).x",
        ],
        id="combined-sampler-split",
    ),
    pytest.param(
        "uniform sampler2D tex;",
        "outp[i] = lookup(0.25) * 2.0;",
        "float lookup(float x) { return textureLod(tex, vec2(x, 0.5), 0.0).x; }",
        [
            "lookup(0.25, tex, tex_sampler)",
            "pub fn lookup(x: f32, tex: &Image!(2D, type=f32, sampled), tex_sampler: &Sampler) -> f32",
        ],
        id="helper-captures-texture-and-sampler",
    ),
    pytest.param(
        "uniform sampler2D texs[2];",
        "outp[i] = look(texs[1]);",
        "float look(sampler2D t) { return textureLod(t, vec2(0.5, 0.5), 0.0).x; }",
        [
            "texs: &[Image!(2D, type=f32, sampled); 2]",
            "texs_sampler: &Sampler",
            "pub fn look(t: &Image!(2D, type=f32, sampled), t_sampler: &Sampler) -> f32",
            "t.sample_by_lod(*t_sampler, Vec2::new(0.5, 0.5), 0.0).x",
        ],
        id="texture-array-element-argument",
    ),
    pytest.param(
        "uniform texture2D texa; uniform texture2D texb; uniform sampler smp;",
        "outp[i] = textureLod(texa, smp, vec2(0.5, 0.5), 0.0).x + textureLod(texb, smp, vec2(0.5, 0.5), 0.0).y;",
        "",
        [
            "smp: &Sampler",
            "texa.sample_by_lod(*smp, Vec2::new(0.5, 0.5), 0.0).x",
            "texb.sample_by_lod(*smp, Vec2::new(0.5, 0.5), 0.0).y",
        ],
        id="shared-separate-sampler",
    ),
    pytest.param(
        "uniform texture2D tex; uniform sampler smps[2];",
        "outp[i] = look(tex, smps[1]);",
        "float look(texture2D t, sampler s) { return textureLod(t, s, vec2(0.5, 0.5), 0.0).x; }",
        ["smps: &[Sampler; 2]", "pub fn look(t: &Image!(2D, type=f32, sampled), s: &Sampler) -> f32"],
        id="sampler-array-element-argument",
    ),
    pytest.param(
        "uniform samplerCube tex;",
        "outp[i] = textureLod(tex, vec3(0.5, 0.5, 1.0), 0.0).x;",
        "",
        ["tex: &Image!(cube, type=f32, sampled)", "tex.sample_by_lod(*tex_sampler, Vec3::new(0.5, 0.5, 1.0), 0.0)"],
        id="cube",
    ),
    pytest.param(
        # GLSL 4.60 section 8.9: outside a fragment shader an implicit-LOD sample reads the base level, and SPIR-V
        # allows ImplicitLod instructions only in the Fragment execution model.
        "uniform sampler2D tex;",
        "outp[i] = texture(tex, vec2(0.5, 0.5)).x;",
        "",
        ["tex.sample_by_lod(*tex_sampler, Vec2::new(0.5, 0.5), 0.0).x"],
        id="compute-sample-reads-base-level",
    ),
    pytest.param(
        "uniform sampler2DShadow tex;",
        "outp[i] = texture(tex, vec3(0.5, 0.5, 0.25));",
        "",
        [
            "tex: &Image!(2D, type=f32, sampled, depth)",
            "tex.sample_depth_reference_by_lod(*tex_sampler, Vec2::new(0.5, 0.5), 0.25, 0.0)",
        ],
        id="shadow-depth-reference",
    ),
    pytest.param(
        "uniform sampler2D tex;",
        "outp[i] = textureGather(tex, vec2(0.5, 0.5), 1).y;",
        "",
        ["tex.gather(*tex_sampler, Vec2::new(0.5, 0.5), 1).y"],
        id="gather-constant-component",
    ),
    pytest.param(
        "uniform sampler2D tex;",
        "ivec2 s = textureSize(tex, 0); outp[i] = texelFetch(tex, s - ivec2(1, 1), 0).x;",
        "",
        [
            "tex.query_size_lod::<UVec2>((0) as u32).as_ivec2()",
            "tex.fetch_with_lod(",
        ],
        id="fetch-and-size-without-sampler",
    ),
    pytest.param(
        "layout(rgba32f) uniform image2D img;",
        "imageStore(img, ivec2(0, 0), vec4(1.0, 2.0, 3.0, 4.0)); ivec2 s = imageSize(img); outp[i] = float(s.y);",
        "",
        [
            "img: &Image!(2D, format=rgba32f, sampled=false)",
            "unsafe { img.write(IVec2::new(0, 0), Vec4::new(1.0, 2.0, 3.0, 4.0)) }",
            "img.query_size::<UVec2>().as_ivec2()",
        ],
        id="storage-image-store-and-size",
    ),
    pytest.param(
        # GLSL 4.60 section 8.12: imageLoad fills the components a format lacks with 0, and alpha with 1.
        "layout(r32f) uniform image2D img;",
        "outp[i] = imageLoad(img, ivec2(0, 0)).w;",
        "",
        ["{ let __t: f32 = img.read(IVec2::new(0, 0)); Vec4::new(__t, 0.0, 0.0, 1.0) }.w"],
        id="one-component-load-widens",
    ),
    pytest.param(
        "layout(rg32f) uniform image2D img;",
        "imageStore(img, ivec2(0, 0), vec4(1.0, 2.0, 3.0, 4.0)); outp[i] = 0.0;",
        "",
        ["unsafe { img.write(IVec2::new(0, 0), (Vec4::new(1.0, 2.0, 3.0, 4.0)).xy()) }"],
        id="two-component-store-narrows",
    ),
    pytest.param(
        "layout(r32ui) uniform uimage2D img;",
        "imageStore(img, ivec2(0, 0), uvec4(i)); outp[i] = float(imageLoad(img, ivec2(0, 0)).x);",
        "",
        [
            "img: &Image!(2D, format=r32ui, sampled=false)",
            "{ let __t: u32 = img.read(IVec2::new(0, 0)); UVec4::new(__t, 0, 0, 1) }.x",
        ],
        id="unsigned-storage-image",
    ),
    pytest.param(
        "layout(rgba32f) uniform image2DArray img;",
        "imageStore(img, ivec3(0, 0, 1), vec4(1.0)); outp[i] = 0.0;",
        "",
        ["img: &Image!(2D, format=rgba32f, sampled=false, arrayed)", "img.write(IVec3::new(0, 0, 1),"],
        id="arrayed-storage-image",
    ),
    pytest.param(
        "layout(rgba32f) uniform image2D img;",
        "put(img); outp[i] = 0.0;",
        "void put(image2D dst) { imageStore(dst, ivec2(0, 0), vec4(1.0)); }",
        ["pub fn put(dst: &Image!(2D, format=rgba32f, sampled=false)) -> ()"],
        id="helper-parameter-takes-the-argument-format",
    ),
    pytest.param(
        "layout(r32f) uniform image2D img;",
        "outer(img); outp[i] = 0.0;",
        "void inner(image2D dst) { imageStore(dst, ivec2(0, 0), vec4(1.0)); } void outer(image2D d) { inner(d); }",
        [
            "pub fn inner(dst: &Image!(2D, format=r32f, sampled=false)) -> ()",
            "pub fn outer(d: &Image!(2D, format=r32f, sampled=false)) -> ()",
        ],
        id="format-passes-through-helpers",
    ),
    pytest.param(
        "layout(rgba8) uniform image2D imgs[2];",
        "put(imgs[1]); outp[i] = imageLoad(imgs[0], ivec2(0, 0)).x;",
        "void put(image2D dst) { imageStore(dst, ivec2(0, 0), vec4(1.0)); }",
        [
            "imgs: &[Image!(2D, format=rgba8, sampled=false); 2]",
            "put(&imgs[",
            "pub fn put(dst: &Image!(2D, format=rgba8, sampled=false)) -> ()",
        ],
        id="image-array-element-argument",
    ),
]


@pytest.mark.parametrize("declarations, body, helpers, expected", LOWERINGS)
def test_images_lower_to_spirv_std_bindings_and_methods(
    declarations, body, helpers, expected, tmp_path
):
    source = _kernel(declarations, body, helpers)
    flat = _flat(_translate(source, tmp_path))
    assert "use spirv_std::Image;" in flat
    for text in expected:
        assert _flat(text) in flat, text
    assert_generated_rust_smoke_compiles(
        _translate(source, tmp_path, format_output=True), tmp_path
    )


def test_fragment_samples_keep_their_implicit_level_of_detail(tmp_path):
    source = """
shader Probe {
    struct FragmentOutput {
        vec4 color;
    }
    fragment {
        uniform sampler2D tex;
        uniform sampler2DShadow depth;
        FragmentOutput main() {
            FragmentOutput result;
            float lit = texture(depth, vec3(0.5, 0.5, 0.25));
            result.color = texture(tex, vec2(0.5, 0.5)) * lit;
            return result;
        }
    }
}
"""
    flat = _flat(_translate(source, tmp_path))
    assert "tex.sample(*tex_sampler, Vec2::new(0.5, 0.5))" in flat
    assert "depth.sample_depth_reference(*depth_sampler, Vec2::new(0.5, 0.5), 0.25)" in flat
    assert_generated_rust_smoke_compiles(
        _translate(source, tmp_path, format_output=True), tmp_path
    )


@pytest.mark.parametrize(
    "declarations, body, helpers, message",
    [
        pytest.param(
            "uniform sampler2D tex;",
            "outp[i] = textureLodOffset(tex, vec2(0.5, 0.5), 0.0, ivec2(1, 0)).x;",
            "",
            "'textureLodOffset' reads with a texel offset, and spirv-std's image operands have no Offset",
            id="texel-offset",
        ),
        pytest.param(
            "uniform image2D img;",
            "imageStore(img, ivec2(0, 0), vec4(1.0)); outp[i] = 0.0;",
            "",
            r"storage image 'img' \(image2D\) has no format layout qualifier",
            id="storage-image-without-format",
        ),
        pytest.param(
            "",
            "outp[i] = 0.0;",
            "void put(image2D dst) { imageStore(dst, ivec2(0, 0), vec4(1.0)); }",
            "storage image parameter 'dst' of 'put' receives no image with a format layout qualifier",
            id="helper-parameter-without-format",
        ),
        pytest.param(
            "layout(r32f) uniform image2D a; layout(rgba32f) uniform image2D b;",
            "put(a); put(b); outp[i] = 0.0;",
            "void put(image2D dst) { imageStore(dst, ivec2(0, 0), vec4(1.0)); }",
            "storage image parameter 'dst' of 'put' receives images of the formats r32f, rgba32f",
            id="helper-parameter-with-two-formats",
        ),
        pytest.param(
            "uniform sampler2D texs[2][2];",
            "outp[i] = textureLod(texs[1][0], vec2(0.5, 0.5), 0.0).x;",
            "",
            "'texs' is an array of resource arrays, which rust-gpu does not bind",
            id="array-of-arrays",
        ),
        pytest.param(
            "uniform sampler2D tex;",
            "int c = int(i); outp[i] = textureGather(tex, vec2(0.5, 0.5), c).y;",
            "",
            "'textureGather' has no spirv-std method for these operands",
            id="gather-variable-component",
        ),
        pytest.param(
            "uniform sampler2DShadow tex;",
            "outp[i] = textureGatherCompare(tex, vec2(0.5, 0.5), 0.5).x;",
            "",
            r"'textureGatherCompare' has no spirv-std method \(OpImageDrefGather\)",
            id="gather-compare",
        ),
        pytest.param(
            "layout(rgba32f) uniform image2DMS img;",
            "outp[i] = imageLoad(img, ivec2(0, 0), 0).x;",
            "",
            "'imageLoad' has no spirv-std method for these operands",
            id="multisampled-storage-load",
        ),
    ],
)
def test_images_rust_gpu_cannot_express_are_refused(
    declarations, body, helpers, message, tmp_path
):
    with pytest.raises(RustLoweringRefused, match=message):
        _translate(_kernel(declarations, body, helpers), tmp_path)
