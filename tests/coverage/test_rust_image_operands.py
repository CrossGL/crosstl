"""The ``*_with`` forms of spirv_std's image methods in the Rust feature list: one feature per image operand that
SPIR-V lets each form's instruction take, and their probes."""

import pytest

from tools.coverage import probes, validators
from tools.coverage.features import spirv_std
from tools.coverage.model import Type

# An excerpt of spirv-std 0.11's image.rs: the ImageWithMethods implementation that the sample-parameter macro
# expands, with a fetch, a gather, a sample and a projective depth comparison, whose asm! strings end in the $LOD and
# $PARAMS placeholders the macro fills.
IMAGE = r"""
#[crate::macros::gen_sample_param_permutations]
impl<
    SampledType: SampleType<FORMAT, COMPONENTS>,
    const DIM: u32,
    const DEPTH: u32,
    const ARRAYED: u32,
    const MULTISAMPLED: u32,
    const SAMPLED: u32,
    const FORMAT: u32,
    const COMPONENTS: u32,
>
    ImageWithMethods<
        SampledType,
        DIM,
        DEPTH,
        ARRAYED,
        MULTISAMPLED,
        SAMPLED,
        FORMAT,
        COMPONENTS,
        SampleParams,
    > for Image<SampledType, DIM, DEPTH, ARRAYED, MULTISAMPLED, SAMPLED, FORMAT, COMPONENTS>
{
    #[crate::macros::gpu_only]
    #[doc(alias = "OpImageFetch")]
    fn fetch_with<I>(
        &self,
        coordinate: impl ImageCoordinate<I, DIM, ARRAYED>,
        params: SampleParams,
    ) -> SampledType::SampleResult
    where
        I: Integer,
    {
        let mut result = SampledType::Vec4::default();
        unsafe {
            asm! {
                "%image = OpLoad _ {this}",
                "%coordinate = OpLoad _ {coordinate}",
                "%result = OpImageFetch typeof*{result} %image %coordinate $PARAMS",
                "OpStore {result} %result",
                result = in(reg) &mut result,
                this = in(reg) self,
                coordinate = in(reg) &coordinate,
            }
        }
        result.truncate_into()
    }

    /// Gathers the requested component from four texels.
    #[crate::macros::gpu_only]
    #[doc(alias = "OpImageGather")]
    fn gather_with<F>(
        &self,
        sampler: Sampler,
        coordinate: impl ImageCoordinate<F, DIM, ARRAYED>,
        component: u32,
        params: SampleParams,
    ) -> SampledType::Vec4
    where
        Self: HasGather,
        F: Float,
    {
        let mut result = SampledType::Vec4::default();
        unsafe {
            asm! {
                "%result = OpImageGather typeof*{result} %sampledImage %coordinate {component} $PARAMS",
                "OpStore {result} %result",
            }
        }
        result
    }

    /// Sample texels at `coord` from the image using `sampler`.
    #[crate::macros::gpu_only]
    fn sample_with<F>(
        &self,
        sampler: Sampler,
        coord: impl ImageCoordinate<F, DIM, ARRAYED>,
        params: SampleParams,
    ) -> SampledType::SampleResult
    where
        F: Float,
    {
        unsafe {
            let mut result = SampledType::Vec4::default();
            asm!(
                "%result = OpImageSample$LOD typeof*{result} %sampledImage %coord $PARAMS",
                "OpStore {result} %result",
            );
            result.truncate_into()
        }
    }

    #[crate::macros::gpu_only]
    #[doc(alias = "OpImageSampleProjDrefImplicitLod")]
    fn sample_depth_reference_with_project_coordinate_with<F>(
        &self,
        sampler: Sampler,
        project_coordinate: impl ImageCoordinate<F, DIM, { Arrayed::True as u32 }>,
        depth_reference: f32,
        params: SampleParams,
    ) -> SampledType
    where
        F: Float,
    {
        let mut result = Default::default();
        unsafe {
            asm!(
                "%result = OpImageSampleProjDref$LOD _ %sampledImage %project_coordinate %depth_reference $PARAMS",
                "OpStore {result} %result",
            );
        }
        result
    }
}
"""


@pytest.fixture(scope="module")
def features():
    return {feature.name: feature for feature in spirv_std.images(IMAGE)}


def test_each_form_is_a_feature_per_operand_its_instruction_takes(features):
    assert sorted(features) == [
        "Image::fetch_with(lod)",
        "Image::fetch_with(sample_index)",
        "Image::sample_depth_reference_with_project_coordinate_with(bias)",
        "Image::sample_depth_reference_with_project_coordinate_with(grad)",
        "Image::sample_depth_reference_with_project_coordinate_with(lod)",
        "Image::sample_with(bias)",
        "Image::sample_with(grad)",
        "Image::sample_with(lod)",
    ]


def test_operands_are_made_from_probe_values_and_a_sample_index_reads_a_multisampled_image(
    features,
):
    fetch = features["Image::fetch_with(sample_index)"].signature.params
    assert fetch[0].type.opaque == "Image!(2D, type=f32, sampled, multisampled)"
    assert fetch[1].type == Type("int", (2,))
    assert fetch[2].type.opaque == "SampleParams(sample_index)"
    # A fetch takes an integer level of detail, a sample a float one.
    lod = features["Image::fetch_with(lod)"].signature.params
    assert lod[0].type.opaque == "Image!(2D, type=f32, sampled)"
    assert lod[2].type.opaque == "SampleParams(integer lod)"
    assert features["Image::sample_with(lod)"].signature.params[-1].type.opaque == (
        "SampleParams(lod)"
    )


def test_only_a_bias_needs_a_fragment_shader(features):
    stages = {name: feature.stage for name, feature in features.items()}
    assert {name for name, stage in stages.items() if stage == "fragment"} == {
        "Image::sample_with(bias)",
        "Image::sample_depth_reference_with_project_coordinate_with(bias)",
    }


def test_forms_are_the_glsl_texture_functions_with_the_same_operand(features):
    spelled = {name: feature.crossgl for name, feature in features.items()}
    assert spelled == {
        "Image::fetch_with(lod)": "texelFetch",
        "Image::fetch_with(sample_index)": "texelFetch",
        "Image::sample_depth_reference_with_project_coordinate_with(bias)": (
            "textureProj"
        ),
        "Image::sample_depth_reference_with_project_coordinate_with(grad)": (
            "textureProjGrad"
        ),
        "Image::sample_depth_reference_with_project_coordinate_with(lod)": (
            "textureProjLod"
        ),
        "Image::sample_with(bias)": "texture",
        "Image::sample_with(grad)": "textureGrad",
        "Image::sample_with(lod)": "textureLod",
    }
    multisampled = features["Image::fetch_with(sample_index)"].counterpart
    assert multisampled.params[0].type.opaque == "sampler2DMS"


def test_probes_import_the_trait_and_pass_the_operand(features):
    native = probes.build(features["Image::sample_with(grad)"], "rust").source
    assert "use spirv_std::image::ImageWithMethods;" in native
    assert "spirv_std::image::sample_with::grad(a3.xy(), a3.zw())" in native
    native = probes.build(features["Image::fetch_with(sample_index)"], "rust").source
    assert (
        "u0.fetch_with(a1, spirv_std::image::sample_with::sample_index(a2))" in native
    )
    crossgl = probes.build(
        features["Image::fetch_with(sample_index)"], "crossgl"
    ).source
    assert "uniform sampler2DMS u0;" in crossgl
    assert "texelFetch(u0, a1, a2)" in crossgl


@pytest.mark.skipif(
    not validators.available("rust"), reason="rust-gpu is not installed"
)
def test_image_operand_probes_compile_with_rust_gpu(features):
    for feature in features.values():
        probe = probes.build(feature, "rust")
        verdict = validators.validate(
            "rust", probe.source, feature.stage, entry=probe.entry
        )
        assert verdict.accepted, (feature.name, verdict.log)
