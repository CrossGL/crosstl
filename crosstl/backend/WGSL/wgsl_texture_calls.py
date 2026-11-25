"""WGSL texture built-in functions (W3C WGSL, section 17.7) written as CrossGL for the WGSL importer.

CrossGL takes a texture and a sampler as separate arguments where WGSL does, in the forms the HLSL importer writes:
``texture``, ``textureOffset``, ``textureLod``, ``textureLodOffset``, ``textureGrad``, ``textureGradOffset``,
``textureCompare``, ``textureCompareLod``, ``textureGather`` and ``textureGatherCompare``. An array layer, which
WGSL passes as its own argument, is the last coordinate component, as in GLSL. Storage textures are CrossGL images
(``imageLoad``, ``imageStore``, ``imageSize``). WGSL counts dimensions, levels, layers and samples as ``u32`` where
CrossGL's queries return ``int`` values, so the results are converted.
"""

from __future__ import annotations

from .wgsl_types import (
    F32,
    I32,
    U32,
    Scalar,
    Texture,
    Vector,
    WgslImportError,
    concretize,
    is_abstract,
)

__all__ = ["emit_texture_call"]

_COORDINATE_SIZE = {
    "1d": 1,
    "2d": 2,
    "2d_array": 2,
    "3d": 3,
    "cube": 3,
    "cube_array": 3,
}


def _arg(converter, node, index, scalar=None):
    arg = node.args[index]
    type_ = converter.deref(converter.type_of(arg))
    if scalar is not None and is_abstract(type_):
        from .wgsl_types import with_scalar

        return converter.emit(arg, with_scalar(type_, scalar))
    return converter.emit(arg)


def _int_coordinates(converter, node, index, size):
    """Integer texel coordinates as CrossGL's ``int``/``ivecN``; WGSL allows ``u32`` ones."""
    type_ = concretize(converter.deref(converter.type_of(node.args[index])))
    text = _arg(converter, node, index, I32)
    scalar = type_.elem if isinstance(type_, Vector) else type_
    if scalar == U32:
        return f"ivec{size}({text})" if size > 1 else f"int({text})"
    return text


def _int_value(converter, node, index):
    type_ = concretize(converter.deref(converter.type_of(node.args[index])))
    text = _arg(converter, node, index, I32)
    return f"int({text})" if type_ == U32 else text


def _with_layer(converter, coordinates, layer, size, floating):
    if floating:
        return f"vec{size + 1}({coordinates}, float({layer}))"
    return f"ivec{size + 1}({coordinates}, {layer})"


def _sampled_coordinates(converter, node, texture, start):
    """The coordinates of a sampling call from argument ``start``, with the layer folded in, and the index after."""
    size = _COORDINATE_SIZE[texture.dimension]
    coordinates = _arg(converter, node, start, F32)
    if texture.dimension.endswith("_array"):
        layer = _int_value(converter, node, start + 1)
        return _with_layer(converter, coordinates, layer, size, True), start + 2
    return coordinates, start + 1


def _offset(converter, node, index):
    if index < len(node.args):
        return _int_coordinates(converter, node, index, 2), True
    return None, False


def _query(converter, node, texture, name):
    node.line
    target = converter.emit(node.args[0])
    if name == "textureNumLevels":
        return f"uint(textureQueryLevels({target}))"
    if name == "textureNumSamples":
        return f"uint(textureSamples({target}))"
    if texture.is_storage:
        size = f"imageSize({target})"
    else:
        level = _int_value(converter, node, 1) if len(node.args) > 1 else "0"
        size = f"textureSize({target}, {level})"
    if name == "textureNumLayers":
        return f"uint({size}.z)"
    dimension = {"1d": 1, "3d": 3}.get(texture.dimension, 2)
    if texture.dimension.endswith("_array"):
        size = f"{size}.xy"
    if dimension == 1:
        return f"uint({size})"
    return f"uvec{dimension}({size})"


def _load_store(converter, node, texture, name):
    target = converter.emit(node.args[0])
    size = _COORDINATE_SIZE[texture.dimension]
    coordinates = _int_coordinates(converter, node, 1, size)
    index = 2
    if texture.dimension.endswith("_array"):
        coordinates = _with_layer(
            converter, coordinates, _int_value(converter, node, 2), size, False
        )
        index = 3
    if name == "textureStore":
        value = _arg(converter, node, index, texture.sampled)
        return f"imageStore({target}, {coordinates}, {value})"
    if texture.is_storage:
        return f"imageLoad({target}, {coordinates})"
    level = _int_value(converter, node, index) if index < len(node.args) else "0"
    fetched = f"texelFetch({target}, {coordinates}, {level})"
    return f"{fetched}.x" if texture.is_depth else fetched


def _sample(converter, node, texture, name):
    target, sampler = converter.emit(node.args[0]), converter.emit(node.args[1])
    coordinates, index = _sampled_coordinates(converter, node, texture, 2)
    prefix = f"{target}, {sampler}, {coordinates}"
    if name == "textureSample":
        offset, has_offset = _offset(converter, node, index)
        return (
            f"textureOffset({prefix}, {offset})" if has_offset else f"texture({prefix})"
        )
    if name == "textureSampleBias":
        bias = _arg(converter, node, index, F32)
        offset, has_offset = _offset(converter, node, index + 1)
        return (
            f"textureOffset({prefix}, {offset}, {bias})"
            if has_offset
            else f"texture({prefix}, {bias})"
        )
    if name == "textureSampleLevel":
        level = _arg(converter, node, index, I32 if texture.is_depth else F32)
        if texture.is_depth:
            level = f"float({level})"
        offset, has_offset = _offset(converter, node, index + 1)
        call = (
            f"textureLodOffset({prefix}, {level}, {offset})"
            if has_offset
            else f"textureLod({prefix}, {level})"
        )
        return f"{call}.x" if texture.is_depth else call
    if name == "textureSampleGrad":
        ddx, ddy = _arg(converter, node, index, F32), _arg(
            converter, node, index + 1, F32
        )
        offset, has_offset = _offset(converter, node, index + 2)
        if has_offset:
            return f"textureGradOffset({prefix}, {ddx}, {ddy}, {offset})"
        return f"textureGrad({prefix}, {ddx}, {ddy})"
    if name in ("textureSampleCompare", "textureSampleCompareLevel"):
        reference = _arg(converter, node, index, F32)
        if len(node.args) > index + 1:
            raise WgslImportError(f"{name} with an offset is not supported", node.line)
        if name == "textureSampleCompare":
            return f"textureCompare({prefix}, {reference})"
        return f"textureCompareLod({prefix}, {reference}, 0.0)"
    if name == "textureSampleBaseClampToEdge":
        half = f"vec2(0.5) / vec2(textureSize({target}, 0))"
        return f"textureLod({target}, {sampler}, clamp({coordinates}, {half}, vec2(1.0) - {half}), 0.0)"
    raise WgslImportError(f"{name} is not supported", node.line)


def _gather(converter, node, texture_index, name):
    texture = converter.deref(converter.type_of(node.args[texture_index]))
    target = converter.emit(node.args[texture_index])
    sampler = converter.emit(node.args[texture_index + 1])
    coordinates, index = _sampled_coordinates(
        converter, node, texture, texture_index + 2
    )
    prefix = f"{target}, {sampler}, {coordinates}"
    if name == "textureGatherCompare":
        reference = _arg(converter, node, index, F32)
        if len(node.args) > index + 1:
            raise WgslImportError(
                "textureGatherCompare with an offset is not supported", node.line
            )
        return f"textureGatherCompare({prefix}, {reference})"
    if len(node.args) > index:
        raise WgslImportError(
            "textureGather with an offset is not supported", node.line
        )
    if texture_index == 0:
        return f"textureGather({prefix})"
    component = converter.ctx.eval_const(node.args[0])
    if not isinstance(component, int) or not 0 <= component <= 3:
        raise WgslImportError(
            "textureGather needs a constant component from 0 to 3", node.line
        )
    return (
        f"textureGather({prefix})"
        if component == 0
        else f"textureGather({prefix}, {component})"
    )


def emit_texture_call(converter, node, arg_types):
    name = node.callee.name
    if name == "textureGather":
        first = arg_types[0]
        return _gather(converter, node, 0 if isinstance(first, Texture) else 1, name)
    texture = arg_types[0]
    if not isinstance(texture, Texture):
        raise WgslImportError(f"{name} needs a texture", node.line)
    if name == "textureGatherCompare":
        return _gather(converter, node, 0, name)
    if name in (
        "textureDimensions",
        "textureNumLayers",
        "textureNumLevels",
        "textureNumSamples",
    ):
        return _query(converter, node, texture, name)
    if name in ("textureLoad", "textureStore"):
        return _load_store(converter, node, texture, name)
    return _sample(converter, node, texture, name)


__all__ += ["Scalar"]
