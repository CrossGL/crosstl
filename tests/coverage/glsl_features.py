"""Builders for the GLSL function features that the probe tests share."""

from tools.coverage.model import Feature, Param, Signature, Type

VEC3 = Type("float", (3,))


def feature(name, result, *params, stage="compute", crossgl=None):
    signature = Signature(
        result, tuple(p if isinstance(p, Param) else Param(p) for p in params)
    )
    return Feature(
        "glsl",
        "function",
        name,
        signature,
        crossgl=name if crossgl is None else crossgl,
        stage=stage,
    )
