"""Typed refusals for GLSL buffer blocks that a target's byte-offset lowering cannot express.

HLSL and Metal have no interface-block storage type, so CrossTL lowers a GLSL ``buffer`` block whose members mix
fixed-size data with a runtime-sized array to a ``(RW)ByteAddressBuffer`` or a ``device uchar*`` addressed by byte
offsets. When that lowering cannot cover a block, a member access or a store, the translation stops here instead of
omitting the declaration and replacing its uses with zero values, which would change the shader's interface and
its results without a word.
"""

from __future__ import annotations

from ..diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = [
    "BufferBlockUnsupportedError",
    "buffer_block_target",
    "record_unsupported_layout_failures",
]

_OFFSET_LAYOUTS = ("std140", "std430")


def record_unsupported_layout_failures(
    nodes,
    lowered_blocks,
    var_failures,
    struct_failures,
    *,
    is_block,
    node_type,
    layout_of,
    base_type,
    lowering,
):
    """Record why a block in a layout other than std140/std430 was not lowered.

    The shared collector skips such blocks without a reason, so the refusal would not say what failed.
    ``lowering`` names the target lowering for the message (``Metal device pointer offset lowering``).
    """
    for node in nodes or []:
        var_name = getattr(node, "name", getattr(node, "variable_name", None))
        if not var_name or var_name in lowered_blocks or var_name in var_failures:
            continue
        vtype = node_type(node)
        if not is_block(node, vtype):
            continue
        layout = layout_of(node)
        if str(layout).lower() in _OFFSET_LAYOUTS:
            continue
        reason = (
            f"unsupported layout {layout}: {lowering} currently supports std140 and "
            "std430 only"
        )
        var_failures[var_name] = reason
        struct_failures.setdefault(str(base_type(vtype)), reason)


_TARGETS = {"hlsl": "directx", "directx": "directx", "metal": "metal"}


def buffer_block_target(target):
    """The diagnostic target key for an emitter's display name (``HLSL`` -> ``directx``)."""
    return _TARGETS[str(target).lower()]


class BufferBlockUnsupportedError(TranslationDiagnostic):
    """A GLSL buffer block construct the target's byte-offset lowering does not cover.

    ``construct`` names what failed (``declaration``, ``compound store``, ``atomic`` and so on) and becomes part of
    the missing capability, so reports can group refusals by construct.
    """

    def __init__(self, target, construct, reason):
        target = buffer_block_target(target)
        self.construct = construct
        self.missing_capabilities = (
            f"{target}.glsl-buffer-block.{construct.replace(' ', '-')}",
        )
        super().__init__(
            f"GLSL buffer block {construct}: {reason}",
            target=target,
            code=f"{CODE_PREFIX}{target}-buffer-block-unsupported",
        )
