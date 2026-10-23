"""Metal sparse texture reads are refused instead of reaching the targets verbatim.

``texture.sparse_sample(s, uv)`` returns ``sparse_color<T>``: the texel and whether its tile is resident. CrossGL has
no read that reports residency, and the importer passed the call, the type and ``.resident()`` through unchanged, so
HLSL failed to compile (``invalid format for vector swizzle 'resident'``).
"""

import pytest

import crosstl
from crosstl.backend.import_refusals import MetalImportRefused

SOURCE = """
#include <metal_stdlib>
using namespace metal;

struct Varyings {
    float4 position [[position]];
    float2 uv;
};

fragment half4 shade(Varyings in [[stage_in]],
                     texture2d<half> image [[texture(0)]],
                     sampler linear [[sampler(0)]]) {
    sparse_color<half4> color = image.METHOD(linear, in.uv);
    half4 base = half4(0.0h);
    if (color.resident()) {
        base = color.value();
    }
    return base;
}
"""


@pytest.mark.parametrize("method", ["sparse_sample", "sparse_gather"])
@pytest.mark.parametrize("backend", ["cgl", "directx", "opengl", "vulkan", "metal"])
def test_sparse_texture_reads_are_refused(tmp_path, method, backend):
    path = tmp_path / "shade.metal"
    path.write_text(SOURCE.replace("METHOD", method), encoding="utf-8")
    with pytest.raises(MetalImportRefused) as raised:
        crosstl.translate(str(path), backend=backend)
    assert f"sparse texture {method}" in str(raised.value)
    assert (
        raised.value.project_diagnostic_code
        == "project.translate.metal-import-unsupported"
    )
