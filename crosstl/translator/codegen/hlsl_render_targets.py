"""HLSL types a pixel shader can write to a render target (``SV_Target``).

A render-target output is a scalar or a vector of up to four components of a 32-bit or 16-bit float or integer type
(D3D11 functional specification 22.12.1; DXC accepts ``float16_t``, ``min16float`` and the 16-bit integers with
native 16-bit types). Booleans and matrices are not.
"""

from __future__ import annotations

import re

__all__ = ["is_render_target_value_type"]

_RENDER_TARGET_TYPE = re.compile(
    r"(?:float|half|float16_t|min16float|min10float|int|uint|int16_t|uint16_t|min16int|min16uint|dword)[1-4]?"
)


def is_render_target_value_type(type_name) -> bool:
    """Whether the HLSL type ``type_name`` can be written to ``SV_Target``."""
    return (
        bool(type_name)
        and _RENDER_TARGET_TYPE.fullmatch(str(type_name).strip()) is not None
    )
