"""Sampler feedback textures: declared for Direct3D, refused with a typed diagnostic for every other target."""

import pytest

import crosstl
from crosstl.translator.semantics import FeedbackTextureUnsupportedError

HLSL_FEEDBACK = """RWStructuredBuffer<uint> outp : register(u0);
FeedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP> feedback : register(u1);
Texture2D<float4> tex : register(t0);
SamplerState s : register(s0);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    feedback.WriteSamplerFeedbackLevel(tex, s, float2(0.5, 0.5), 0.0);
    outp[id.x] = 0u;
}
"""

CROSSGL_FEEDBACK = """
shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer uint* outp;
        void main() {
            %s obj;
            outp[gl_GlobalInvocationID.x] = 0u;
        }
    }
}
"""

TARGETS = ["metal", "opengl", "vulkan", "slang", "wgsl", "cuda", "hip", "rust", "mojo"]


def _write(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(source)
    return str(path)


def test_hlsl_feedback_textures_round_trip_to_directx(tmp_path):
    path = _write(tmp_path, "feedback.hlsl", HLSL_FEEDBACK)
    hlsl = crosstl.translate(path, backend="directx")
    assert "FeedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP> feedback" in hlsl
    assert "feedback.WriteSamplerFeedbackLevel(tex, s" in hlsl


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "name, source",
    [
        ("feedback.hlsl", HLSL_FEEDBACK),
        (
            "feedback.cgl",
            CROSSGL_FEEDBACK
            % "FeedbackTexture2DArray<SAMPLER_FEEDBACK_MIP_REGION_USED>",
        ),
    ],
)
def test_feedback_textures_are_refused_for_targets_without_them(
    tmp_path, target, name, source
):
    path = _write(tmp_path, name, source)
    with pytest.raises(FeedbackTextureUnsupportedError) as raised:
        crosstl.translate(path, backend=target)
    assert (
        raised.value.code == f"project.translate.{target}-sampler-feedback-unsupported"
    )


def test_a_feedback_texture_argument_that_is_not_a_kind_is_refused(tmp_path):
    path = _write(
        tmp_path, "feedback.cgl", CROSSGL_FEEDBACK % "FeedbackTexture2D<float>"
    )
    with pytest.raises(
        FeedbackTextureUnsupportedError, match="SAMPLER_FEEDBACK_MIN_MIP"
    ):
        crosstl.translate(path, backend="directx")
