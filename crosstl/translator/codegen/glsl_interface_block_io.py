"""Stage inputs and outputs that a GLSL ``in``/``out`` interface block already declares.

The GLSL import keeps a stage interface block twice: as a ``@glsl_interface_block`` structure, which the GLSL
emitter declares as the block itself, and as the entry structure's members, which every other target reads. An
instance block (``out Data {...} data;``) becomes one member of the block's type named after the instance; an
anonymous block becomes one member per block member. The GLSL emitter declares those members once, through the
block, and still reserves their locations.
"""


def _base_type(type_name):
    return str(type_name or "").split("[", 1)[0].strip()


def _blocks(codegen, direction):
    blocks = getattr(codegen, "glsl_interface_block_structs_by_name", None) or {}
    for name in sorted(blocks):
        node = blocks[name]
        qualifiers = {
            str(qualifier).lower()
            for qualifier in codegen.glsl_interface_block_qualifiers(node)
        }
        if direction in qualifiers:
            yield node


def _instance_name(codegen, block):
    instance_args = codegen.glsl_attribute_arguments(block, "glsl_interface_instance")
    if not instance_args:
        return None
    return codegen.glsl_global_identifier_name(
        codegen.attribute_value_to_string(instance_args[0])
    )


def block_declared_stage_io(codegen, direction, member, emitted_name):
    """The block that declares entry-structure ``member`` (emitted as ``emitted_name``), or None.

    ``direction`` is ``"in"`` or ``"out"``.
    """
    member_base = _base_type(codegen.member_type_name(member))
    for block in _blocks(codegen, direction):
        instance = _instance_name(codegen, block)
        if instance is not None:
            if instance == emitted_name and member_base == _base_type(
                codegen.map_type(block.name)
            ):
                return block
            continue
        for block_member in getattr(block, "members", []) or []:
            name = getattr(block_member, "name", None)
            if name and codegen.glsl_global_identifier_name(name) == emitted_name:
                return block
    return None


def block_member_location_type(codegen, member, block):
    """A type that consumes as many locations as ``member``, counting an instance block's members."""
    if _instance_name(codegen, block) is None:
        return codegen.member_type_name(member)
    count = sum(
        codegen.stage_io_location_count(codegen.member_type_name(block_member))
        for block_member in getattr(block, "members", []) or []
    )
    # The location counter reads only the array length of this spelling.
    return f"vec4[{max(count, 1)}]"


def stage_block_member_layout(codegen, block, member):
    """``layout(location = N, component = C) `` for a member of an ``in``/``out`` block that declares them."""
    qualifiers = {
        str(qualifier).lower()
        for qualifier in codegen.glsl_interface_block_qualifiers(block)
    }
    if not qualifiers & {"in", "out"}:
        return ""
    parts = []
    for attr in getattr(member, "attributes", []) or []:
        name = str(getattr(attr, "name", "")).lower()
        arguments = getattr(attr, "arguments", None) or []
        if name in {"location", "component"} and len(arguments) == 1:
            parts.append(f"{name} = {codegen.attribute_value_to_string(arguments[0])}")
    return f"layout({', '.join(parts)}) " if parts else ""
