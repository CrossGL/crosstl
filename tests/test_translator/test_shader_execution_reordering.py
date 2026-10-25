"""HLSL Shader Model 6.9 shader execution reordering: kept for Direct3D, refused for every other target.

``dx::MaybeReorderThread(hint, bits)`` is the reorder overload that takes no hit object, so it names neither
``dx::HitObject`` nor a ``dx::HitObject::`` method, and the hit object refusal does not cover it. The Metal, GLSL,
Slang, CUDA, HIP, Rust and Mojo generators used to copy the call into their output, where the target compiler
rejects it: ``xcrun metal`` reports ``error: use of undeclared identifier 'dx'`` and ``glslangValidator`` reports
``'::' : not supported``. See :mod:`crosstl.translator.semantics.shader_execution_reordering`.
"""

import pytest

import crosstl
from crosstl.translator.semantics.shader_execution_reordering import (
    ShaderExecutionReorderingUnsupportedError,
)

RAYGEN_REORDER = """RWStructuredBuffer<uint> outp : register(u0);
[shader("raygeneration")]
void main() {
    uint hint = 3u;
    dx::MaybeReorderThread(hint, 2u);
    outp[0] = hint;
}
"""

# The same program written in CrossGL, which spells the intrinsic the way HLSL does.
CROSSGL_REORDER = """
shader Reorder {
    ray_generation {
        RWStructuredBuffer<uint> outp;
        void main() {
            uint hint = 3u;
            dx::MaybeReorderThread(hint, 2u);
            outp[0] = hint;
        }
    }
}
"""

# A reorder call reached through a helper function is refused as well.
RAYGEN_REORDER_IN_HELPER = """RWStructuredBuffer<uint> outp : register(u0);
void ReorderThreads(uint hint) {
    dx::MaybeReorderThread(hint, 2u);
}
[shader("raygeneration")]
void main() {
    ReorderThreads(3u);
    outp[0] = 0u;
}
"""

RAYGEN_PLAIN = """RWStructuredBuffer<uint> outp : register(u0);
[shader("raygeneration")]
void main() {
    uint index = 3u;
    outp[0] = index;
}
"""

TARGETS = ["metal", "opengl", "vulkan", "slang", "cuda", "hip", "rust", "mojo"]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    ("name", "source"),
    [("reorder.hlsl", RAYGEN_REORDER), ("reorder.cgl", CROSSGL_REORDER)],
)
def test_reorder_calls_are_refused(tmp_path, target, name, source):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")

    with pytest.raises(
        ShaderExecutionReorderingUnsupportedError, match="dx::MaybeReorderThread"
    ) as raised:
        crosstl.translate(str(path), backend=target)

    assert raised.value.code == (
        f"project.translate.{target}-shader-execution-reordering-unsupported"
    )
    assert raised.value.missing_capabilities == (
        f"{target}.shader-execution-reordering",
    )
    assert raised.value.project_diagnostic_code == raised.value.code


def test_a_reorder_call_in_a_helper_function_is_refused(tmp_path):
    path = tmp_path / "reorder.hlsl"
    path.write_text(RAYGEN_REORDER_IN_HELPER, encoding="utf-8")

    with pytest.raises(ShaderExecutionReorderingUnsupportedError):
        crosstl.translate(str(path), backend="metal")


@pytest.mark.parametrize("backend", ["directx", "hlsl", "dx11", "dx12"])
def test_direct3d_keeps_the_reorder_call(tmp_path, backend):
    path = tmp_path / "reorder.hlsl"
    path.write_text(RAYGEN_REORDER, encoding="utf-8")

    assert "dx::MaybeReorderThread(hint, 2u);" in crosstl.translate(
        str(path), backend=backend
    )


def test_a_ray_stage_without_reordering_still_translates(tmp_path):
    path = tmp_path / "plain.hlsl"
    path.write_text(RAYGEN_PLAIN, encoding="utf-8")

    for target in TARGETS:
        assert crosstl.translate(str(path), backend=target).strip()
