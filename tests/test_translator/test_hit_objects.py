"""Shader Model 6.9 hit objects: kept for Direct3D, refused with a typed diagnostic for every other target."""

import pytest

import crosstl
from crosstl.translator.semantics import HitObjectUnsupportedError

HLSL_HIT_OBJECT = """RWStructuredBuffer<uint> outp : register(u0);
[shader("raygeneration")]
void main() {
    uint i = 0u;
    dx::HitObject obj;
    uint r = obj.GetHitKind();
    outp[i] = r;
}
"""

# A static method names the type only in the call.
CROSSGL_MAKE_NOP = """
shader HitObjects {
    ray_generation {
        RWStructuredBuffer<uint> outp;
        void main() {
            outp[0] = dx::HitObject::MakeNop().GetHitKind();
        }
    }
}
"""

TARGETS = ["metal", "opengl", "vulkan", "slang", "cuda", "hip", "rust", "mojo"]


def _write(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(source)
    return str(path)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "name, source",
    [("hit.hlsl", HLSL_HIT_OBJECT), ("make_nop.cgl", CROSSGL_MAKE_NOP)],
)
def test_hit_objects_are_refused_for_targets_without_them(
    tmp_path, target, name, source
):
    path = _write(tmp_path, name, source)
    with pytest.raises(HitObjectUnsupportedError, match="dx::HitObject") as raised:
        crosstl.translate(path, backend=target)
    assert raised.value.code == f"project.translate.{target}-hit-object-unsupported"


def test_directx_keeps_hit_objects(tmp_path):
    hlsl = crosstl.translate(
        _write(tmp_path, "hit.hlsl", HLSL_HIT_OBJECT), backend="directx"
    )
    assert "dx::HitObject obj;" in hlsl
    hlsl = crosstl.translate(
        _write(tmp_path, "make_nop.cgl", CROSSGL_MAKE_NOP), backend="directx"
    )
    assert "dx::HitObject::MakeNop().GetHitKind()" in hlsl
