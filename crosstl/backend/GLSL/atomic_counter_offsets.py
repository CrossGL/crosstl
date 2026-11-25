"""The buffer offset of each atomic counter a GLSL source declares.

GLSL 4.60 section 4.4.6.3 (GLSL ES 3.10 section 4.4.6.2): every atomic counter binding point keeps a next offset. A
counter declared without ``offset`` takes it, and each counter moves it past its own end, 4 bytes per counter times
the elements of an array. A type-only declaration such as ``layout(binding = 2, offset = 4) uniform atomic_uint;``
moves the next offset of binding 2 to 4 without declaring a counter.

CrossGL keeps neither those defaults nor the order of counters across stage sections, so the importer writes the
offset of every counter that does not start its binding's buffer as ``@offset`` (:attr:`OFFSET`), and drops the
type-only declarations it has applied (:func:`is_applied_default`). A value that is not an integer literal, such as a
constant's name, leaves the binding's later implicit offsets unknown; those counters keep their source layout.
"""

import re

# The resolved offset in bytes, set on each atomic counter declaration whose offset this module knows.
OFFSET = "atomic_counter_offset"
_COUNTER_BYTES = 4
_INTEGER = re.compile(r"(0[xX][0-9a-fA-F]+|0[0-7]*|[1-9][0-9]*)[uU]?")


def literal(value):
    """The value of a GLSL integer literal (``4``, ``0x10``, ``010``, ``4u``), else None."""
    text = getattr(value, "value", value)
    if isinstance(text, bool):
        return None
    if isinstance(text, int):
        return text
    match = _INTEGER.fullmatch(str(text).strip())
    if match is None:
        return None
    digits = match.group(1)
    if digits.startswith(("0x", "0X")):
        return int(digits, 16)
    if len(digits) > 1 and digits.startswith("0"):
        return int(digits, 8)
    return int(digits)


def _default_layout(layout_entry):
    """``(binding, offset)`` of a type-only atomic counter declaration, None for other layout declarations."""
    if not isinstance(layout_entry, dict) or layout_entry.get("type") != "atomic_uint":
        return None
    layout = layout_entry.get("layout") or {}
    binding = literal(layout.get("binding"))
    if binding is None or "offset" not in layout:
        return None
    return binding, literal(layout.get("offset"))


def is_applied_default(layout_entry):
    """Whether ``layout_entry`` is a type-only atomic counter declaration whose offset the counters now carry."""
    default = _default_layout(layout_entry)
    return default is not None and default[1] is not None


class AtomicCounterOffsets:
    """The next offset of each binding point of one program, in declaration order."""

    def __init__(self):
        # binding -> next offset, or None once a non-literal offset or array size has made it unknown
        self.next = {}

    def default(self, layout_entry):
        """Apply a type-only ``layout(binding = B, offset = O) uniform atomic_uint;`` declaration."""
        default = _default_layout(layout_entry)
        if default is not None:
            binding, offset = default
            self.next[binding] = offset

    def place(self, variable):
        """Record the offset of an ``atomic_uint`` declaration as :attr:`OFFSET`."""
        if str(getattr(variable, "vtype", "")) != "atomic_uint":
            return
        layout = getattr(variable, "layout", None) or {}
        binding = literal(layout.get("binding"))
        if binding is None:
            return
        if "offset" in layout:
            offset = literal(layout.get("offset"))
        else:
            offset = self.next.get(binding, 0)
        if offset is None:
            self.next[binding] = None
            return
        setattr(variable, OFFSET, offset)
        count = 1
        sizes = getattr(variable, "array_sizes", None) or ()
        if not sizes and getattr(variable, "array_size", None) is not None:
            sizes = (variable.array_size,)
        for size in sizes:
            elements = literal(size)
            if elements is None:
                self.next[binding] = None
                return
            count *= elements
        self.next[binding] = offset + _COUNTER_BYTES * count
