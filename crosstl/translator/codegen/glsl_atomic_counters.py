"""The atomic counter buffer binding and offset of each ``atomic_uint`` that GLSL output declares.

Atomic counters have their own binding table (GLSL 4.60 section 4.4.6.3), apart from the texture and image units.
Its guaranteed size, ``gl_MaxAtomicCounterBindings``, is 1 (section 7.3), and glslang checks that minimum, so the
only portable binding is 0. GLSL output therefore puts every counter in binding 0.

Each source binding keeps its own offsets, under GLSL's rule: a counter without an ``offset`` takes the next offset of
its binding, and every counter moves that offset 4 bytes per element past its start. When the source uses more than
one binding, the bindings follow one another in binding 0 in the order the source first declares them, each starting
where the one before ends. The offset of every counter that does not start the buffer is written, so the layout does
not depend on which counters a stage declares, and the resource binding records give each counter's binding and
offset (``atomic-counter-buffer``) for the host.
"""

import re

from ..refusals import OpenGLRefusal

COUNTER_BYTES = 4
BINDING = 0


def is_atomic_counter(codegen, node):
    vtype = codegen.resource_declaration_shape(node)[0]
    return codegen.map_resource_type_with_format(vtype, node) == "atomic_uint"


def _offset_attribute(codegen, node, name):
    for attr in getattr(node, "attributes", None) or []:
        if str(getattr(attr, "name", "") or "").lower() != "offset":
            continue
        arguments = getattr(attr, "arguments", None) or []
        offset = codegen.binding_index_value(arguments[0]) if arguments else None
        if offset is None or offset % COUNTER_BYTES:
            source = (
                codegen.attribute_value_to_string(arguments[0]) if arguments else None
            )
            raise OpenGLRefusal(
                f"GLSL output cannot place atomic counter '{name}' at offset {source}: an atomic counter offset is "
                "an integer literal that is a multiple of 4"
            )
        return offset
    return None


def _elements(codegen, node):
    """The number of counters ``node`` declares, or None when an array size is not an integer literal."""
    array_suffix = codegen.resource_declaration_shape(node)[2] or ""
    count = 1
    for size in re.findall(r"\[([^\]]*)\]", array_suffix):
        size = size.strip()
        if not size.isdigit():
            return None
        count *= int(size)
    return count


def counter_placements(codegen, nodes):
    """``{id(node): (binding, offset)}`` for the atomic counters among ``nodes``, in declaration order."""
    counters = []
    next_offsets = {}
    for node in nodes:
        if not is_atomic_counter(codegen, node):
            continue
        name = codegen.resource_node_name(node, "<unnamed>")
        source_binding = codegen.explicit_resource_binding_index(node)
        source_binding = 0 if source_binding is None else source_binding
        offset = _offset_attribute(codegen, node, name)
        if offset is None:
            offset = next_offsets.get(source_binding, 0)
            if offset is None:
                raise OpenGLRefusal(
                    f"GLSL output cannot place atomic counter '{name}': it follows an atomic counter array in "
                    f"binding {source_binding} whose size is not an integer literal, so its offset is unknown"
                )
        elements = _elements(codegen, node)
        size = None if elements is None else COUNTER_BYTES * elements
        next_offsets[source_binding] = None if size is None else offset + size
        counters.append((node, name, source_binding, offset, size))

    extents = {}
    for node, name, source_binding, offset, size in counters:
        for other, other_name, other_binding, other_offset, other_size in counters:
            if other is node:
                break
            if other_binding != source_binding:
                continue
            if (other_size is None or offset < other_offset + other_size) and (
                size is None or other_offset < offset + size
            ):
                raise OpenGLRefusal(
                    f"GLSL output cannot place atomic counters '{other_name}' and '{name}': they overlap in "
                    f"atomic counter binding {source_binding} at offset {offset}"
                )
        end = None if size is None else offset + size
        if source_binding not in extents:
            extents[source_binding] = end
        elif extents[source_binding] is not None:
            extents[source_binding] = (
                None if end is None else max(extents[source_binding], end)
            )

    bases = {}
    base = 0
    for source_binding, extent in extents.items():
        if base is None:
            raise OpenGLRefusal(
                f"GLSL output cannot place the atomic counters of binding {source_binding} in binding {BINDING}: "
                "they follow an atomic counter array whose size is not an integer literal"
            )
        bases[source_binding] = base
        base = None if extent is None else base + extent

    return {
        id(node): (BINDING, bases[source_binding] + offset)
        for node, _name, source_binding, offset, _size in counters
    }
