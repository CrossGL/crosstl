"""Normalized storage images must keep their texel format in HLSL and Slang."""

import pytest

from tests.test_translator.test_spirv_separate_textures import _translate

SOURCE = """shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* fin;
        layout(rgba8) uniform writeonly image2D img;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            imageStore(img, ivec2(0, 0), vec4(fin[i]));
        }
    }
}
"""


@pytest.mark.parametrize("backend", ["directx", "slang"])
@pytest.mark.parametrize(
    ("image_format", "texel_type"),
    [
        ("rgba8", "unorm float4"),
        ("rgba8_snorm", "snorm float4"),
        ("rgba16", "unorm float4"),
    ],
)
def test_normalized_storage_image_keeps_texel_format(
    tmp_path, backend, image_format, texel_type
):
    code = _translate(tmp_path, SOURCE.replace("rgba8", image_format), ".cgl", backend)
    assert f"RWTexture2D<{texel_type}> img" in code
    if backend == "directx":
        assert (
            f'#ifdef __spirv__\n[[vk::image_format("{image_format}")]]\n#endif\n'
            f"RWTexture2D<{texel_type}> img"
        ) in code
