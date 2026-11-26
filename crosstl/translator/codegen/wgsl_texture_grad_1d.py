"""WGSL output of an explicit-gradient read of a 1D texture.

WGSL defines ``textureSampleGrad`` only for 2D, 2D array, 3D, cube and cube array textures (WGSL section
17.7.11), but ``textureSampleLevel`` takes a ``texture_1d<f32>`` (section 17.7.10). For a 1D texture the
level of detail that the gradients select is ``log2(rho)``, where ``rho = max(|du/dx|, |du/dy|)`` and ``u``
is the coordinate scaled by the texture's width (Vulkan specification, "Scale Factor Operation, Level-of-Detail
Operation and Image Level(s) Selection"; a 1D texture has no anisotropy), so HLSL ``Texture1D.SampleGrad``
and GLSL ``textureGrad`` on a ``sampler1D`` become ``textureSampleLevel`` at that level. A zero gradient
gives ``log2(0) = -inf``, which selects the base level and the magnification filter; the level is written
with the smallest normal ``f32`` instead, which selects the same, so no infinity reaches the sampler.

WGSL has no offset overload of ``textureSampleLevel`` for a 1D texture, so an offset read is refused.
"""

from __future__ import annotations

from ..refusals import WGSLRefusal

SMALLEST_NORMAL_F32 = "1.17549435e-38"


def is_wgsl_1d_texture(texture_type):
    """Whether ``texture_type``, a WGSL sampled texture type such as ``texture_1d<f32>``, is 1D."""
    return bool(texture_type) and texture_type.startswith("texture_1d")


def wgsl_texture_1d_grad_call(codegen, function_name, args, *, offset):
    """``textureSampleLevel`` at the gradients' level for a 1D texture, else ``None`` for any other texture."""
    texture_type = codegen.sampled_texture_type(codegen.expression_type(args[0]))
    if not is_wgsl_1d_texture(texture_type):
        return None
    if offset:
        raise WGSLRefusal(
            f"WGSL target cannot lower {function_name}() with an offset on a 1D texture: WGSL's "
            "textureSampleGrad takes no 1D texture and textureSampleLevel has no 1D offset overload"
        )
    implicit = 4
    explicit = 5
    rendered = codegen.generate_texture_call_args(
        args, function_name=function_name, implicit=implicit, explicit=explicit
    )
    texture, sampler, coordinate, ddx, ddy = rendered
    width = f"f32(textureDimensions({texture}))"
    rho = f"max(max(abs({ddx}), abs({ddy})) * {width}, {SMALLEST_NORMAL_F32})"
    return f"textureSampleLevel({texture}, {sampler}, {coordinate}, log2({rho}))"
