"""Validate retained source union layouts for word-based target storage."""

import re

from ..ast import StructNode
from .array_utils import evaluate_literal_int_expression, split_array_type_suffix


class UnionLayoutCollector:
    """Collect overlapping value views without changing the source member types."""

    def __init__(self, generator, error, static_member):
        self.generator = generator
        self.error = error
        self.static_member = static_member

    @staticmethod
    def attribute(node, name):
        return next(
            (
                attribute
                for attribute in getattr(node, "attributes", []) or []
                if str(getattr(attribute, "name", "")).lower() == name
            ),
            None,
        )

    def layout_int(self, attribute, index):
        arguments = list(getattr(attribute, "arguments", []) or [])
        if index >= len(arguments):
            return None
        return evaluate_literal_int_expression(
            arguments[index], self.generator.literal_int_constants
        )

    def layout_name(self, attribute, index):
        arguments = list(getattr(attribute, "arguments", []) or [])
        if index >= len(arguments):
            return None
        value = self.generator.attribute_value_to_string(arguments[index])
        return str(value).strip().lower() if value is not None else None

    def collect(self, structs):
        layouts = {}
        for node in structs or []:
            if not isinstance(node, StructNode):
                continue
            attribute = self.attribute(node, "union_layout")
            if attribute is None:
                continue

            arguments = list(getattr(attribute, "arguments", []) or [])
            if len(arguments) != 4:
                raise self.error(
                    node,
                    reason="layout-contract-malformed",
                    detail="@union_layout requires size, alignment, byte order, and ABI",
                )
            size = self.layout_int(attribute, 0)
            alignment = self.layout_int(attribute, 1)
            byte_order = self.layout_name(attribute, 2)
            source_abi = self.layout_name(attribute, 3)
            if not size or not alignment:
                raise self.error(
                    node,
                    reason="source-layout-unresolved",
                    detail="the source ABI did not provide a concrete size and alignment",
                    size=size,
                    alignment=alignment,
                )
            if size not in {4, 8, 16}:
                raise self.error(
                    node,
                    reason="storage-size-unsupported",
                    detail=(
                        f"{size}-byte storage is outside the supported 4/8/16-byte "
                        "value contract"
                    ),
                    size=size,
                    alignment=alignment,
                )
            if alignment != size:
                raise self.error(
                    node,
                    reason="aggregate-alignment-unsupported",
                    detail=(
                        f"{size}-byte storage with {alignment}-byte alignment may "
                        "change enclosing aggregate layout"
                    ),
                    size=size,
                    alignment=alignment,
                )
            if byte_order != "little_endian":
                raise self.error(
                    node,
                    reason="byte-order-unsupported",
                    detail=f"byte order '{byte_order or '<missing>'}' is not supported",
                    size=size,
                    alignment=alignment,
                )
            if not source_abi:
                raise self.error(
                    node,
                    reason="source-abi-missing",
                    detail="the retained union contract has no source ABI",
                    size=size,
                    alignment=alignment,
                )

            word_count = size // 4
            storage_type = self.generator.map_type(
                "uint" if word_count == 1 else f"uvec{word_count}"
            )
            member_names = {
                getattr(member, "name", None)
                for member in getattr(node, "members", []) or []
                if getattr(member, "name", None)
            }
            storage_field = "CrossGLUnionStorage"
            suffix = 2
            while storage_field in member_names:
                storage_field = f"CrossGLUnionStorage{suffix}"
                suffix += 1

            members = {}
            for member in getattr(node, "members", []) or []:
                if self.static_member(member):
                    continue
                member_name = getattr(member, "name", None)
                if not member_name:
                    raise self.error(
                        node,
                        reason="anonymous-member-unsupported",
                        detail="anonymous instance members are not representable",
                        member=member,
                        size=size,
                        alignment=alignment,
                    )
                member_attribute = self.attribute(member, "union_member_layout")
                if (
                    member_attribute is None
                    or len(list(getattr(member_attribute, "arguments", []) or [])) != 3
                ):
                    raise self.error(
                        node,
                        reason="member-layout-missing",
                        detail="the retained member layout is missing or malformed",
                        member=member,
                        size=size,
                        alignment=alignment,
                    )
                offset = self.layout_int(member_attribute, 0)
                member_size = self.layout_int(member_attribute, 1)
                member_alignment = self.layout_int(member_attribute, 2)
                member_type = self.generator.struct_member_types.get(node.name, {}).get(
                    member_name
                )
                if offset != 0:
                    raise self.error(
                        node,
                        reason="member-offset-unsupported",
                        detail=f"member offset {offset!r} does not alias byte zero",
                        member=member,
                        member_type=member_type,
                        size=size,
                        alignment=alignment,
                    )
                if member_size != size:
                    raise self.error(
                        node,
                        reason="member-size-mismatch",
                        detail=(
                            f"member size {member_size!r} does not cover the full "
                            f"{size}-byte canonical storage"
                        ),
                        member=member,
                        member_type=member_type,
                        size=size,
                        alignment=alignment,
                    )
                if (
                    not member_alignment
                    or member_alignment > alignment
                    or alignment % member_alignment != 0
                ):
                    raise self.error(
                        node,
                        reason="member-alignment-unsupported",
                        detail=(
                            f"member alignment {member_alignment!r} is incompatible "
                            f"with aggregate alignment {alignment}"
                        ),
                        member=member,
                        member_type=member_type,
                        size=size,
                        alignment=alignment,
                    )
                member_storage = self.member_storage_kind(
                    node,
                    member,
                    member_type,
                    word_count,
                    size,
                    alignment,
                )
                member_storage["node"] = member
                members[member_name] = member_storage

            if not members:
                raise self.error(
                    node,
                    reason="empty-union-unsupported",
                    detail="the union has no instance members",
                    size=size,
                    alignment=alignment,
                )
            layouts[node.name] = {
                "name": node.name,
                "size": size,
                "alignment": alignment,
                "byte_order": byte_order,
                "source_abi": source_abi,
                "word_count": word_count,
                "storage_type": storage_type,
                "storage_field": storage_field,
                "members": members,
                "node": node,
            }
        return layouts

    def member_storage_kind(
        self,
        node,
        member,
        member_type,
        word_count,
        size,
        alignment,
    ):
        type_name = str(member_type or "").strip()
        base_type, array_suffix = split_array_type_suffix(type_name)
        base_type = str(base_type).strip()
        extent = None
        if array_suffix:
            match = re.fullmatch(r"\[([1-9][0-9]*)\]", array_suffix)
            if match is None:
                raise self.error(
                    node,
                    reason="member-array-shape-unsupported",
                    detail=(
                        f"member type '{type_name}' is not a one-dimensional "
                        "literal fixed array"
                    ),
                    member=member,
                    member_type=type_name,
                    size=size,
                    alignment=alignment,
                )
            extent = int(match.group(1))
            if word_count == 1:
                raise self.error(
                    node,
                    reason="single-word-array-member-unsupported",
                    detail=(
                        "single-word canonical storage cannot preserve dynamic "
                        f"indexing of '{type_name}'"
                    ),
                    member=member,
                    member_type=type_name,
                    size=size,
                    alignment=alignment,
                )

        if base_type == "u8vec4":
            if extent is None and word_count == 1:
                return {"kind": "u8x4", "type": type_name}
            if extent == word_count:
                return {
                    "kind": "u8x4_array",
                    "type": type_name,
                    "extent": extent,
                }
            raise self.error(
                node,
                reason="member-storage-shape-mismatch",
                detail=(
                    f"byte-vector view '{type_name}' does not provide one u8vec4 "
                    f"per canonical word ({word_count} required)"
                ),
                member=member,
                member_type=type_name,
                size=size,
                alignment=alignment,
            )

        mapped_base = self.generator.map_type(base_type)
        vector = re.fullmatch(r"([uib]?)vec([234])", mapped_base)
        if vector:
            family = {"": "float", "u": "uint", "i": "int", "b": "bool"}[vector[1]]
            mapped_base = f"{family}{vector[2]}"
        if mapped_base == "bool4":
            if extent is None and word_count == 1:
                return {"kind": "bool8x4", "type": type_name}
            if extent == word_count:
                return {
                    "kind": "bool8x4_array",
                    "type": type_name,
                    "extent": extent,
                }
            raise self.error(
                node,
                reason="member-storage-shape-mismatch",
                detail=(
                    f"Boolean-vector view '{type_name}' does not provide one "
                    f"four-byte bool4 per canonical word ({word_count} required)"
                ),
                member=member,
                member_type=type_name,
                size=size,
                alignment=alignment,
            )

        scalar_match = re.fullmatch(r"(uint|int|float)([234])?", mapped_base)
        if scalar_match is None:
            raise self.error(
                node,
                reason="member-shape-unsupported",
                detail=(
                    f"member type '{type_name}' is not a supported 32-bit "
                    "scalar/vector or u8vec4 word view"
                ),
                member=member,
                member_type=type_name,
                size=size,
                alignment=alignment,
            )
        component_kind, width_text = scalar_match.groups()
        width = int(width_text) if width_text else 1
        if extent is not None:
            if width != 1 or extent != word_count:
                raise self.error(
                    node,
                    reason="member-storage-shape-mismatch",
                    detail=(
                        f"word-array view '{type_name}' does not provide exactly "
                        f"{word_count} scalar words"
                    ),
                    member=member,
                    member_type=type_name,
                    size=size,
                    alignment=alignment,
                )
            return {
                "kind": f"{component_kind}32_array",
                "type": type_name,
                "extent": extent,
            }
        if width != word_count:
            raise self.error(
                node,
                reason="member-storage-shape-mismatch",
                detail=(
                    f"member type '{type_name}' exposes {width} words but the "
                    f"canonical storage contains {word_count}"
                ),
                member=member,
                member_type=type_name,
                size=size,
                alignment=alignment,
            )
        return {"kind": f"{component_kind}32", "type": type_name}
