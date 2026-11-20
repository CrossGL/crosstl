"""spirv_std's Image, SampledImage and ByteAddressableBuffer methods in the Rust feature list, and their probes."""

from pathlib import Path

import pytest

from tools.coverage import objects, probes, rustgpu, validators
from tools.coverage.features import spirv_std
from tools.coverage.model import Type
from tools.coverage.rust import Rust

FIXTURES = Path(__file__).parent / "fixtures"

# An excerpt of spirv-std 0.11's image.rs, with the shapes the parser reads: impl blocks that fix some of Image's
# constant parameters, asm! strings whose braces need not balance, doc comments that quote across lines, a method
# defined twice, trait bounds and the trait implementations they name.
IMAGE = (FIXTURES / "spirv_std_image.rs").read_text(encoding="utf-8")

SAMPLED = "Image!(2D, type=f32, sampled)"
FLOAT2, FLOAT3, FLOAT4 = (Type("float", (n,)) for n in (2, 3, 4))
INT2 = Type("int", (2,))


@pytest.fixture(scope="module")
def features():
    return {feature.name: feature for feature in spirv_std.images(IMAGE)}


def test_image_methods_are_named_by_their_type_and_counted_once(features):
    assert sorted(features) == [
        "Image::fetch",
        "Image::gather",
        "Image::query_lod",
        "Image::query_size",
        "Image::read",
        "Image::read_subpass",
        "Image::sample",
        "Image::sample_depth_reference_by_lod",
        "Image::sample_depth_reference_with_project_coordinate_by_lod",
        "Image::sample_with_project_coordinate",
        "Image::write",
        "SampledImage::sample_by_lod",
    ]
    assert {feature.group for feature in features.values()} == {"image"}


def receiver(feature):
    return feature.signature.params[0].type.opaque


def test_image_methods_are_probed_on_the_2d_image_their_block_and_bounds_allow(
    features,
):
    assert receiver(features["Image::fetch"]) == SAMPLED
    # HasGather's 2D implementation is single-sampled, as the block that defines gather is.
    assert receiver(features["Image::gather"]) == SAMPLED
    # Storage images, from the block that fixes Sampled::No before the one of unknown sampling.
    assert (
        receiver(features["Image::read"]) == "Image!(2D, format=rgba32f, sampled=false)"
    )
    assert (
        receiver(features["Image::write"])
        == "Image!(2D, format=rgba32f, sampled=false)"
    )
    assert (
        receiver(features["Image::read_subpass"])
        == "Image!(subpass, type=f32, sampled=false)"
    )
    # HasQuerySize's first 2D implementation is the multisampled one.
    assert (
        receiver(features["Image::query_size"])
        == "Image!(2D, type=f32, sampled, multisampled)"
    )
    assert (
        receiver(features["Image::sample_depth_reference_by_lod"])
        == "Image!(2D, type=f32, sampled, depth)"
    )
    assert (
        receiver(features["SampledImage::sample_by_lod"]) == f"SampledImage<{SAMPLED}>"
    )
    for feature in features.values():
        assert receiver(feature) in objects.UNIFORMS["rust"], feature.name


def test_image_methods_with_an_implicit_level_of_detail_run_in_a_fragment_shader(
    features,
):
    fragment = {
        name for name, feature in features.items() if feature.stage == "fragment"
    }
    assert fragment == {
        "Image::sample",
        "Image::sample_with_project_coordinate",
        "Image::query_lod",
        "Image::read_subpass",
    }


def test_image_method_arguments_have_the_types_of_a_2d_f32_image(features):
    def params(name):
        return [(p.type, p.direction) for p in features[name].signature.params[1:]]

    sampler = (Type.other("Sampler"), "in")
    assert params("Image::fetch") == [(INT2, "in")]
    assert params("Image::sample") == [sampler, (FLOAT2, "in")]
    assert features["Image::sample"].signature.result == FLOAT4
    # The gathered component is a constant (VUID-StandaloneSpirv-OpImageGather-04664), and the first.
    assert params("Image::gather") == [sampler, (FLOAT2, "in"), (Type("int"), "const")]
    assert params("Image::sample_with_project_coordinate") == [sampler, (FLOAT3, "in")]
    assert params("Image::sample_depth_reference_by_lod") == [
        sampler,
        (FLOAT2, "in"),
        (Type("float"), "in"),
        (Type("float"), "in"),
    ]
    assert features["Image::sample_depth_reference_by_lod"].signature.result == Type(
        "float"
    )
    assert params("Image::read_subpass") == [(INT2, "const")]
    assert params("Image::write") == [(INT2, "in"), (FLOAT4, "in")]
    assert features["Image::query_size"].signature.result == Type("uint", (2,))
    assert features["Image::write"].spelling == "unsafe .write"


def test_a_projective_depth_comparison_reads_its_inputs_in_glsl_order(features):
    feature = features["Image::sample_depth_reference_with_project_coordinate_by_lod"]
    assert [p.type for p in feature.signature.params[2:]] == [
        Type.other(spirv_std.PROJECTIVE_DEPTH),
        Type("float"),
    ]
    assert Rust().making(spirv_std.PROJECTIVE_DEPTH) == (FLOAT4, "{0}.xyw(), {0}.z")
    native = probes.build(feature, "rust").source
    assert (
        "let a2: Vec4 = Vec4::new(fin[i + 0], fin[i + 1], fin[i + 2], fin[i + 3]);"
        in native
    )
    assert "(*u1, a2.xyw(), a2.z, a3)" in native
    crossgl = probes.build(feature, "crossgl").source
    assert (
        "vec4 a1 = vec4(fin[i + 0u], fin[i + 1u], fin[i + 2u], fin[i + 3u]);" in crossgl
    )
    assert "textureProjLod(u0, a1, a2)" in crossgl


def test_image_probes_declare_resources_as_entry_point_parameters(features):
    source = probes.build(features["Image::sample"], "rust").source
    assert "use spirv_std::Image;\nuse spirv_std::Sampler;\n" in source
    assert (
        "    #[spirv(descriptor_set = 0, binding = 8)] u0: &Image!(2D, type=f32, sampled),\n"
        "    #[spirv(descriptor_set = 0, binding = 9)] u1: &Sampler,\n"
        "    color: &mut Vec4,\n"
    ) in source
    # A method takes the sampler by value.
    assert "u0.sample(*u1, a2)" in source
    subpass = probes.build(features["Image::read_subpass"], "rust").source
    assert (
        "#[spirv(input_attachment_index = 0, descriptor_set = 0, binding = 8)] u0: "
        "&Image!(subpass, type=f32, sampled=false)"
    ) in subpass
    assert "u0.read_subpass(IVec2::ZERO)" in subpass
    sampled = probes.build(features["SampledImage::sample_by_lod"], "rust").source
    assert "use spirv_std::image::SampledImage;" in sampled
    assert "u0: &SampledImage<Image!(2D, type=f32, sampled)>" in sampled


def test_image_methods_are_glsl_texture_functions_in_crossgl(features):
    assert set(spirv_std.COUNTERPARTS) >= set(features)
    assert features["Image::sample"].crossgl == "texture"
    assert features["Image::read"].crossgl == "imageLoad"
    crossgl = probes.build(features["Image::gather"], "crossgl").source
    assert "uniform sampler2D u0;" in crossgl
    assert "textureGather(u0, a1)" in crossgl
    native = probes.build(features["Image::gather"], "rust").source
    assert "u0.gather(*u1, a2, 0)" in native


def test_rust_gpu_enables_the_image_query_and_input_attachment_capabilities():
    assert {"ImageQuery", "InputAttachment"} <= set(rustgpu.CAPABILITIES)


@pytest.mark.skipif(
    not validators.available("rust"), reason="rust-gpu is not installed"
)
def test_image_probes_compile_with_rust_gpu(features):
    for feature in features.values():
        probe = probes.build(feature, "rust")
        verdict = validators.validate(
            "rust", probe.source, feature.stage, entry=probe.entry
        )
        assert verdict.accepted, (feature.name, verdict.log)


# An excerpt of spirv-std 0.11's byte_addressable_buffer.rs: constructors, and a load that the mutable buffer defines
# a second time.
BYTE_BUFFER = (FIXTURES / "spirv_std_byte_buffer.rs").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def byte_features():
    return {feature.name: feature for feature in spirv_std.byte_buffers(BYTE_BUFFER)}


def test_byte_buffer_methods_are_counted_once_and_constructors_have_no_crossgl_spelling(
    byte_features,
):
    assert sorted(byte_features) == [
        "ByteAddressableBuffer::as_ref",
        "ByteAddressableBuffer::from_mut_slice",
        "ByteAddressableBuffer::from_slice",
        "ByteAddressableBuffer::load",
        "ByteAddressableBuffer::store",
    ]
    # load is the read-only view's, the first block that defines it.
    load = byte_features["ByteAddressableBuffer::load"]
    assert load.signature.params[0].type.opaque == spirv_std.BYTE_BUFFER
    for name in ("from_slice", "from_mut_slice", "as_ref"):
        feature = byte_features[f"ByteAddressableBuffer::{name}"]
        assert feature.crossgl == "" and "byte-address buffer" in feature.note
        assert feature.signature is not None
        with pytest.raises(probes.NoSpelling):
            probes.build(feature, "crossgl")


def test_byte_buffer_probes_load_and_store_a_word_at_four_times_a_word_index(
    byte_features,
):
    native = probes.build(byte_features["ByteAddressableBuffer::load"], "rust").source
    assert (
        "#[spirv(storage_buffer, descriptor_set = 0, binding = 8)] u0: &[u32],"
        in native
    )
    assert (
        "let r: u32 = unsafe { spirv_std::ByteAddressableBuffer::from_slice(u0).load(a1 * 4) };"
        in native
    )
    crossgl = probes.build(
        byte_features["ByteAddressableBuffer::load"], "crossgl"
    ).source
    assert "uniform ByteAddressBuffer u0;" in crossgl
    assert "uint r = u0.Load(a1 * 4u);" in crossgl
    store = byte_features["ByteAddressableBuffer::store"]
    native = probes.build(store, "rust").source
    assert "u0: &mut [u32]," in native
    assert (
        "spirv_std::ByteAddressableBuffer::from_mut_slice(u0).store(a1 * 4, a2)"
        in native
    )
    crossgl = probes.build(store, "crossgl").source
    assert "uniform RWByteAddressBuffer u0;" in crossgl
    assert "u0.Store(a1 * 4u, a2);" in crossgl


@pytest.mark.skipif(
    not validators.available("rust"), reason="rust-gpu is not installed"
)
def test_byte_buffer_probes_compile_with_rust_gpu(byte_features):
    for feature in byte_features.values():
        probe = probes.build(feature, "rust")
        verdict = validators.validate("rust", probe.source, entry=probe.entry)
        assert verdict.accepted, (feature.name, verdict.log)
