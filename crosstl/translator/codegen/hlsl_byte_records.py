"""Byte-array records in HLSL private and storage views.

HLSL has no 8-bit scalar type, so a CrossGL ``uint8``/``int8`` value lives
in one 32-bit element per byte. A record whose sole instance member is a
fixed byte array therefore keeps one HLSL element per source byte, and a
scalar view of it reads byte ``i`` from element ``i``:

* a private ``(const thread T*)&record`` view passes the member array and
  assembles each ``T`` from ``sizeof(T)`` consecutive elements, masked to
  their low byte;
* a storage ``*((const device Record*)bytes)`` copy reads element ``i`` of
  the record from byte ``i`` of an 8-bit storage view.

Front ends record a record's native storage alignment as one
``@metal_alignas(N)`` attribute. It does not change the member layout of
either view, so it is accepted when ``N`` is a positive power of two and
folded into the record's alignment, which still has to leave no trailing
padding in a private view.
"""

ALIGNMENT_ATTRIBUTE = "metal_alignas"


def record_alignment(struct_node, literal_int_value):
    """``(alignment, ok)`` for a record's layout attributes.

    ``alignment`` is ``None`` without attributes. ``ok`` is false when the
    record carries any other attribute, more than one, or a non-literal or
    non-power-of-two alignment.
    """

    attributes = list(getattr(struct_node, "attributes", None) or ())
    if not attributes:
        return None, True
    if len(attributes) != 1:
        return None, False
    attribute = attributes[0]
    if str(getattr(attribute, "name", "")).lower() != ALIGNMENT_ATTRIBUTE:
        return None, False
    arguments = list(getattr(attribute, "arguments", None) or ())
    if len(arguments) != 1:
        return None, False
    alignment = literal_int_value(arguments[0])
    if (
        not isinstance(alignment, int)
        or isinstance(alignment, bool)
        or alignment <= 0
        or alignment & (alignment - 1)
    ):
        return None, False
    return alignment, True


def is_byte_layout(layout):
    """Whether ``layout`` is an 8-bit integer scalar layout."""

    return (
        layout is not None
        and getattr(layout, "kind", None) == "integer"
        and getattr(layout, "bit_width", None) == 8
    )


def byte_component(layout, mapped_type):
    """A private-view component for one 8-bit integer scalar, else ``None``.

    Its ``word_extent`` is ``None``: bytes never occupy whole words, so only
    a record made of one byte array can back a view.
    """

    if not is_byte_layout(layout):
        return None
    return {
        "kind": "scalar",
        "layout": layout,
        "mapped_type": mapped_type,
        "word_extent": None,
        "size_bytes": 1,
        "alignment": 1,
    }


def byte_segments_are_viewable(segments):
    """Whether byte-array components, if any, form the view's only segment."""

    byte_segments = [
        segment for segment in segments if is_byte_layout(segment.get("layout"))
    ]
    if not byte_segments:
        return True
    return len(segments) == 1 and segments[0]["kind"] == "array_scalar"


def storage_byte_copy_allowed(source_layout, target_layout):
    """Whether a storage record copy may read the record byte by byte.

    Both the storage view and the record's array elements must be 8-bit
    integers, so element ``i`` of the record is byte ``i`` of the view.
    """

    return is_byte_layout(source_layout) and is_byte_layout(target_layout)
