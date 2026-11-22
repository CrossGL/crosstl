"""GLSL buffer-block access in the Slang generator, lowered to byte-address buffer loads, stores and atomics."""

from hashlib import sha1

from ..ast import MemberAccessNode
from .glsl_buffer_array_values import (
    glsl_buffer_array_store_lines,
    glsl_buffer_array_value,
    glsl_buffer_array_value_type,
    glsl_buffer_subscripts_are_repeatable,
)
from .glsl_buffer_block_arrays import (
    glsl_buffer_array_elements,
    glsl_buffer_block_indexed_access,
    glsl_buffer_matrix_column_access,
    glsl_buffer_whole_array_reason,
)
from .glsl_buffer_layout import (
    byte_offset_add,
    glsl_buffer_compound_binary_operator,
    matrix_column_offsets,
    vector_component_offsets,
)
from .hlsl_buffer_component_stores import slang_buffer_component_store
from .hlsl_byteaddress_typed_access import (
    hlsl_byteaddress_is_typed_access,
    hlsl_byteaddress_typed_load,
    hlsl_byteaddress_typed_store,
)


class SlangGlslBufferAccessMixin:
    """GLSL buffer-access members of ``SlangCodeGen``."""

    def glsl_buffer_block_lowering_failure_detail(self, type_name, var_name=None):
        if var_name:
            reason = self.glsl_buffer_block_lowering_failures.get(var_name)
            if reason:
                return reason
        type_name = str(self.resource_base_type(type_name))
        return self.glsl_buffer_block_struct_lowering_failures.get(type_name)

    def glsl_buffer_block_diagnostic(
        self, target, type_name, var_name=None, node=None, declaration_kind=None
    ):
        declaration = str(self.resource_base_type(type_name))
        if declaration_kind:
            declaration = f"{declaration_kind} {declaration}"
        if var_name:
            declaration += f" {var_name}"
        details = ""
        if node is not None:
            layout = self.glsl_buffer_block_layout(node)
            binding, binding_expr = self.explicit_slang_resource_binding(node)
            (
                _register_prefix,
                register_binding,
                register_binding_expr,
                _register_space,
                _register_space_expr,
            ) = self.explicit_slang_register(node)
            if binding is None and binding_expr is None:
                binding = register_binding
                binding_expr = register_binding_expr
            details = f" ({layout}"
            binding_label = binding_expr if binding_expr is not None else binding
            if binding_label is not None:
                details += f", binding = {binding_label}"
            details += ")"
        failure_detail = self.glsl_buffer_block_lowering_failure_detail(
            type_name, var_name
        )
        reason = f"; {failure_detail}" if failure_detail else ""
        return (
            f"// unsupported {target} GLSL buffer block {declaration}{details}: "
            "requires ByteAddressBuffer offset lowering"
            f"{reason}\n"
        )

    def glsl_buffer_block_member_access(self, expr):
        if not isinstance(expr, MemberAccessNode):
            return None
        object_expr = getattr(expr, "object_expr", getattr(expr, "object", None))
        var_name = self.get_expression_name(object_expr)
        member_name = getattr(expr, "member", None)
        if var_name:
            block = self.lowered_glsl_buffer_blocks.get(var_name)
            if block is not None:
                buffer_expr = self.generate_expression(object_expr)
                member = block["members"].get(member_name)
                if member is None:
                    return None
                return {
                    "buffer": buffer_expr,
                    "member": member_name,
                    "readonly": block.get("readonly", False),
                    "writeonly": block.get("writeonly", False),
                    **member,
                }

        parent = self.glsl_buffer_block_array_access(
            object_expr
        ) or self.glsl_buffer_block_member_access(object_expr)
        if parent is None or not parent.get("members") or parent.get("subarray"):
            return None
        member = parent["members"].get(member_name)
        if member is None:
            return None
        return {
            "buffer": parent["buffer"],
            "member": f"{parent['member']}.{member_name}",
            "readonly": parent.get("readonly", False),
            "writeonly": parent.get("writeonly", False),
            **member,
            "offset": byte_offset_add(parent["offset"], member["offset"]),
        }

    def glsl_buffer_block_array_access(self, expr):
        return glsl_buffer_block_indexed_access(self, expr)

    def slang_byteaddress_load_method(self, components):
        return "Load" if components == 1 else f"Load{components}"

    def slang_byteaddress_store_method(self, components):
        return "Store" if components == 1 else f"Store{components}"

    def slang_byteaddress_word_load(self, buffer_name, components, offset):
        """``components`` 32-bit words at byte ``offset``; three go through a helper (slang_byte_address_words)."""
        if components == 3:
            return self.slang_three_component_access.load(buffer_name, offset)
        method = self.slang_byteaddress_load_method(components)
        return f"{buffer_name}.{method}({offset})"

    def slang_byteaddress_word_store(self, buffer_name, components, offset, value):
        """Store ``components`` 32-bit words at byte ``offset``; three go through a helper."""
        if components == 3:
            return self.slang_three_component_access.store(buffer_name, offset, value)
        method = self.slang_byteaddress_store_method(components)
        return f"{buffer_name}.{method}({offset}, {value})"

    def slang_byteaddress_load(self, buffer_name, offset, access):
        if access.get("writeonly"):
            return self.unsupported_glsl_buffer_block_call(
                "load", "requires readable buffer block resource", access["type"]
            )
        if access.get("matrix_columns"):
            columns = []
            for _, column_offset in matrix_column_offsets(
                offset, access["matrix_columns"], access["column_stride"]
            ):
                load = self.slang_byteaddress_word_load(
                    buffer_name, access["matrix_rows"], column_offset
                )
                columns.append(f"asfloat({load})")
            # The buffer holds columns and a Slang matrix constructor fills rows, as in HLSL.
            transposed_type = (
                f"{access['component_type']}{access['matrix_columns']}"
                f"x{access['matrix_rows']}"
            )
            return f"transpose({transposed_type}({', '.join(columns)}))"

        if hlsl_byteaddress_is_typed_access(access):
            return hlsl_byteaddress_typed_load(self, buffer_name, offset, access)

        if access["component_type"] == "bool" and access["components"] > 1:
            values = []
            for _, component_offset in vector_component_offsets(
                offset, access["components"]
            ):
                values.append(f"({buffer_name}.Load({component_offset}) != 0u)")
            return f"{access['slang_type']}({', '.join(values)})"

        load = self.slang_byteaddress_word_load(
            buffer_name, access["components"], offset
        )
        if access["component_type"] == "bool":
            return f"({load} != 0u)"
        if access["component_type"] == "float":
            return f"asfloat({load})"
        if access["component_type"] == "int":
            return f"asint({load})"
        return load

    def slang_byteaddress_store_value(self, value, access):
        if access["component_type"] == "bool":
            if access["components"] > 1:
                value_expr = self.slang_indexable_expression(value)
                fields = "xyzw"[: access["components"]]
                values = [f"({value_expr}.{field} ? 1u : 0u)" for field in fields]
                return f"uint{access['components']}({', '.join(values)})"
            return f"(({value}) ? 1u : 0u)"
        if access.get("layout_type") != access.get("type"):
            if access["component_type"] == "int":
                return f"asuint(int({value}))"
            if access["component_type"] == "uint":
                if access["components"] == 1:
                    return f"uint({value})"
                return f"uint{access['components']}({value})"
        if access["component_type"] in {"float", "int"}:
            return f"asuint({value})"
        return value

    def next_slang_byteaddress_temp_variable(self, prefix):
        name = f"__crossgl_{prefix}_{self.slang_byteaddress_temp_variable_index}"
        self.slang_byteaddress_temp_variable_index += 1
        return name

    def slang_indexable_expression(self, expression):
        expression = str(expression)
        if expression.isidentifier():
            return expression
        if all(part.isidentifier() for part in expression.split(".")):
            return expression
        return f"({expression})"

    def slang_byteaddress_matrix_store(self, buffer_name, offset, value, access):
        value_expr = self.slang_indexable_expression(value)
        lines = []
        for column, column_offset in matrix_column_offsets(
            offset, access["matrix_columns"], access["column_stride"]
        ):
            # A Slang m[i] is a row, as in HLSL; the swizzle selects column ``column``.
            swizzle = "".join(
                f"_m{row}{column}" for row in range(access["matrix_rows"])
            )
            lines.append(
                self.slang_byteaddress_word_store(
                    buffer_name,
                    access["matrix_rows"],
                    column_offset,
                    f"asuint({value_expr}.{swizzle})",
                )
            )
        return "\n".join(lines)

    def slang_byteaddress_bool_vector_store(self, buffer_name, offset, value, access):
        temp_name = self.next_slang_byteaddress_temp_variable("bool_store")
        store_value = self.slang_byteaddress_store_value(temp_name, access)
        return "\n".join(
            [
                f"{access['slang_type']} {temp_name} = {value}",
                self.slang_byteaddress_word_store(
                    buffer_name, access["components"], offset, store_value
                ),
            ]
        )

    def slang_glsl_buffer_aggregate_helper_suffix(self, access):
        return "".join(
            char if char.isalnum() or char == "_" else "_"
            for char in access["slang_type"]
        )

    def slang_glsl_buffer_aggregate_layout_signature(self, access):
        parts = []

        def visit(member_name, member):
            fields = [
                member_name,
                str(member.get("type")),
                str(member.get("layout_type")),
                str(member.get("offset")),
                str(member.get("size")),
                str(member.get("align")),
                str(member.get("components")),
                str(member.get("component_type")),
                str(member.get("matrix_columns")),
                str(member.get("matrix_rows")),
                str(member.get("column_stride")),
                str(member.get("is_array")),
                str(member.get("array_count")),
                str(member.get("stride")),
                str(member.get("runtime_array")),
            ]
            if member.get("inner_arrays"):
                fields.append(str(member["inner_arrays"]))
            parts.append(":".join(fields))
            for child_name, child in (member.get("members") or {}).items():
                visit(f"{member_name}.{child_name}", child)

        for field_name, member in access["members"].items():
            visit(field_name, member)
        return sha1("|".join(parts).encode("utf-8")).hexdigest()[:10]

    def slang_byteaddress_aggregate_load_helper_name(self, access):
        buffer_type = (
            "ByteAddressBuffer" if access.get("readonly") else "RWByteAddressBuffer"
        )
        kind = "ro" if access.get("readonly") else "rw"
        helper_name = (
            f"__crossgl_load_{kind}_glsl_buffer_"
            f"{self.slang_glsl_buffer_aggregate_helper_suffix(access)}_"
            f"{self.slang_glsl_buffer_aggregate_layout_signature(access)}"
        )
        self.required_glsl_buffer_aggregate_load_helpers[(helper_name, buffer_type)] = (
            access
        )
        return helper_name

    def slang_byteaddress_aggregate_load_assignments(
        self, target_name, buffer_name, offset, access, indent=1
    ):
        indent_str = "    " * indent
        lines = []
        for field_name, member in access["members"].items():
            member_offset = byte_offset_add(offset, member["offset"])
            member_target = f"{target_name}.{field_name}"
            field_access = {
                **member,
                "buffer": buffer_name,
                "member": f"{access['member']}.{field_name}",
                "readonly": access.get("readonly", False),
                "writeonly": access.get("writeonly", False),
            }
            if member.get("is_array"):
                array_count = member.get("array_count")
                if member.get("runtime_array") or array_count is None:
                    return None
                for subscripts, element_offset in glsl_buffer_array_elements(
                    member, member_offset
                ):
                    element_target = f"{member_target}{subscripts}"
                    if member.get("members"):
                        nested_lines = (
                            self.slang_byteaddress_aggregate_load_assignments(
                                element_target,
                                buffer_name,
                                element_offset,
                                field_access,
                                indent,
                            )
                        )
                        if nested_lines is None:
                            return None
                        lines.extend(nested_lines)
                    else:
                        value = self.slang_byteaddress_load(
                            buffer_name, element_offset, field_access
                        )
                        lines.append(f"{indent_str}{element_target} = {value};")
                continue
            if member.get("members"):
                nested_lines = self.slang_byteaddress_aggregate_load_assignments(
                    member_target, buffer_name, member_offset, field_access, indent
                )
                if nested_lines is None:
                    return None
                lines.extend(nested_lines)
            else:
                value = self.slang_byteaddress_load(
                    buffer_name, member_offset, field_access
                )
                lines.append(f"{indent_str}{member_target} = {value};")
        return lines

    def generate_slang_glsl_buffer_aggregate_load_helpers(self):
        if not self.required_glsl_buffer_aggregate_load_helpers:
            return ""

        helpers = []
        for (helper_name, buffer_type), access in sorted(
            self.required_glsl_buffer_aggregate_load_helpers.items()
        ):
            lines = [
                f"{access['slang_type']} {helper_name}({buffer_type} buffer, uint offset) {{",
                f"    {access['slang_type']} result;",
            ]
            assignments = self.slang_byteaddress_aggregate_load_assignments(
                "result", "buffer", "offset", access
            )
            if assignments is None:
                continue
            lines.extend(assignments)
            lines.extend(["    return result;", "}"])
            helpers.append("\n".join(lines) + "\n\n")
        return "".join(helpers)

    def slang_byteaddress_aggregate_load(self, buffer_name, offset, access):
        helper_name = self.slang_byteaddress_aggregate_load_helper_name(access)
        return f"{helper_name}({buffer_name}, {offset})"

    def slang_byteaddress_leaf_store(self, buffer_name, offset, value, access):
        if access.get("matrix_columns"):
            return self.slang_byteaddress_matrix_store(
                buffer_name, offset, value, access
            )
        if hlsl_byteaddress_is_typed_access(access):
            return hlsl_byteaddress_typed_store(
                self, buffer_name, offset, value, access
            )
        if access["component_type"] == "bool" and access["components"] > 1:
            return self.slang_byteaddress_bool_vector_store(
                buffer_name, offset, value, access
            )
        store_value = self.slang_byteaddress_store_value(value, access)
        return self.slang_byteaddress_word_store(
            buffer_name, access["components"], offset, store_value
        )

    def slang_byteaddress_aggregate_store_members(
        self, buffer_name, offset, value, access
    ):
        lines = []
        for field_name, member in access["members"].items():
            member_offset = byte_offset_add(offset, member["offset"])
            member_value = f"{value}.{field_name}"
            field_access = {
                **member,
                "buffer": buffer_name,
                "member": f"{access['member']}.{field_name}",
                "readonly": access.get("readonly", False),
                "writeonly": access.get("writeonly", False),
            }
            if member.get("is_array"):
                array_count = member.get("array_count")
                if member.get("runtime_array") or array_count is None:
                    return None
                for subscripts, element_offset in glsl_buffer_array_elements(
                    member, member_offset
                ):
                    element_value = f"{member_value}{subscripts}"
                    if member.get("members"):
                        nested_stores = self.slang_byteaddress_aggregate_store_members(
                            buffer_name, element_offset, element_value, field_access
                        )
                        if nested_stores is None:
                            return None
                        lines.extend(nested_stores)
                    else:
                        store = self.slang_byteaddress_leaf_store(
                            buffer_name, element_offset, element_value, field_access
                        )
                        if store is None:
                            return None
                        lines.extend(store.splitlines())
                continue
            if member.get("members"):
                nested_stores = self.slang_byteaddress_aggregate_store_members(
                    buffer_name, member_offset, member_value, field_access
                )
                if nested_stores is None:
                    return None
                lines.extend(nested_stores)
            else:
                store = self.slang_byteaddress_leaf_store(
                    buffer_name, member_offset, member_value, field_access
                )
                if store is None:
                    return None
                lines.extend(store.splitlines())
        return lines

    def slang_byteaddress_aggregate_store(self, buffer_name, offset, value, access):
        temp_name = self.next_slang_byteaddress_temp_variable("aggregate_store")
        stores = self.slang_byteaddress_aggregate_store_members(
            buffer_name, offset, temp_name, access
        )
        if stores is None:
            return self.refuse_lowering(
                "Slang GLSL buffer block aggregate store: "
                "array fields require element-wise stores"
            )
        return "\n".join([f"{access['slang_type']} {temp_name} = {value}", *stores])

    def slang_byteaddress_matrix_compound_store(
        self, buffer_name, offset, value, op, access
    ):
        compound_ops = {
            "+=": "+",
            "-=": "-",
            "*=": "*",
            "/=": "/",
        }
        binary_op = compound_ops.get(op)
        if binary_op is None:
            return self.refuse_lowering(
                "Slang GLSL buffer block matrix compound store: "
                "requires explicit matrix operation lowering"
            )
        temp_name = self.next_slang_byteaddress_temp_variable("matrix_store")
        current = self.slang_byteaddress_load(buffer_name, offset, access)
        temp = f"{access['slang_type']} {temp_name} = ({current} {binary_op} {value})"
        stores = self.slang_byteaddress_matrix_store(
            buffer_name, offset, temp_name, access
        )
        return f"{temp}\n{stores}"

    def slang_byteaddress_compound_store_diagnostic(self, op, access):
        return self.refuse_lowering(
            "Slang GLSL buffer block compound store: "
            f"operator {op} is not supported for "
            f"{access['component_type']} buffer members"
        )

    def terminate_slang_statement_lines(self, text):
        lines = []
        for line in str(text).splitlines():
            stripped = line.rstrip()
            if not stripped:
                lines.append(stripped)
            elif stripped.endswith((";", "}", "*/")) and stripped.startswith("/*"):
                lines.append(stripped)
            elif stripped.endswith(";"):
                lines.append(stripped)
            else:
                lines.append(f"{stripped};")
        return "\n".join(lines)

    def unsupported_glsl_buffer_block_call(self, operation, reason, result_type=None):
        placeholder = (
            "" if result_type is None else self.zero_value_for_type(result_type)
        )
        return self.refuse_lowering(
            f"Slang GLSL buffer block: {operation} {reason}", placeholder
        )

    def slang_glsl_buffer_whole_array_refusal(self, access, *, indexed, expr=None):
        """The whole block array ``access`` read as one value, or ``None`` for one element.

        See glsl_buffer_array_values; subscripts with side effects are refused.
        """
        if glsl_buffer_whole_array_reason(access, indexed=indexed) is None:
            return None
        if not glsl_buffer_subscripts_are_repeatable(expr):
            return self.unsupported_glsl_buffer_block_call(
                "array value",
                f"{access.get('member')} is read element by element, and its "
                "subscripts have side effects",
                glsl_buffer_array_value_type(access),
            )

        def load_element(element):
            if element.get("members"):
                return self.slang_byteaddress_aggregate_load(
                    element["buffer"], element["offset"], element
                )
            return self.slang_byteaddress_load(
                element["buffer"], element["offset"], element
            )

        return glsl_buffer_array_value(
            access,
            load_element=load_element,
            value_struct=self.slang_array_values.value_struct,
            map_type=self.map_type,
        )

    def slang_glsl_buffer_array_store(self, target, access, offset, value, op):
        """Statements storing array ``value`` into the whole block array ``access``."""
        if op != "=" or not glsl_buffer_subscripts_are_repeatable(target):
            return self.refuse_lowering(
                f"Slang GLSL buffer block store: {access.get('member')} is an array, "
                "written element by element, which needs '=' and subscripts without "
                "side effects",
                ";",
            )
        value_type = glsl_buffer_array_value_type(access)
        rhs = self.generate_expression_with_expected(value, value_type)

        def store_element(element, element_offset, element_value):
            if element.get("members"):
                lines = self.slang_byteaddress_aggregate_store_members(
                    element["buffer"], element_offset, element_value, element
                )
                return (
                    lines
                    if lines is not None
                    else [
                        self.refuse_lowering(
                            "Slang GLSL buffer block store: the array elements hold "
                            "runtime arrays",
                            ";",
                        )
                    ]
                )
            return self.slang_byteaddress_leaf_store(
                element["buffer"], element_offset, element_value, element
            ).splitlines()

        lines = glsl_buffer_array_store_lines(
            access,
            offset,
            rhs,
            self.next_slang_byteaddress_temp_variable("array_store"),
            store_element=store_element,
            map_type=self.map_type,
        )
        return self.terminate_slang_statement_lines("\n".join(lines))

    def generate_glsl_buffer_block_member_load(self, expr):
        access = self.glsl_buffer_block_member_access(expr)
        if access is None or access.get("runtime_array"):
            return None
        refusal = self.slang_glsl_buffer_whole_array_refusal(
            access, indexed=False, expr=expr
        )
        if refusal is not None:
            return refusal
        if access.get("writeonly"):
            return self.unsupported_glsl_buffer_block_call(
                "load", "requires readable buffer block resource", access["type"]
            )
        if access.get("members"):
            return self.slang_byteaddress_aggregate_load(
                access["buffer"], access["offset"], access
            )
        return self.slang_byteaddress_load(access["buffer"], access["offset"], access)

    def generate_glsl_buffer_block_array_load(self, expr):
        access = self.glsl_buffer_block_array_access(expr)
        if access is None:
            return None
        refusal = self.slang_glsl_buffer_whole_array_refusal(
            access, indexed=True, expr=expr
        )
        if refusal is not None:
            return refusal
        if access.get("writeonly"):
            return self.unsupported_glsl_buffer_block_call(
                "load", "requires readable buffer block resource", access["type"]
            )
        if access.get("members"):
            return self.slang_byteaddress_aggregate_load(
                access["buffer"], access["offset_expr"], access
            )
        return self.slang_byteaddress_load(
            access["buffer"], access["offset_expr"], access
        )

    def generate_glsl_buffer_block_store(self, target, value, op):
        access = self.glsl_buffer_block_array_access(
            target
        ) or glsl_buffer_matrix_column_access(self, target)
        indexed = access is not None
        if access is None:
            access = self.glsl_buffer_block_member_access(target)
            if access is None or access.get("runtime_array"):
                return slang_buffer_component_store(self, target, value, op)
            offset = access["offset"]
        else:
            offset = access["offset_expr"]

        whole_array = glsl_buffer_whole_array_reason(access, indexed=indexed)
        if access.get("readonly"):
            return self.refuse_lowering(
                "Slang GLSL buffer block store: "
                "readonly ByteAddressBuffer cannot be written",
                ";",
            )
        if whole_array is not None:
            return self.slang_glsl_buffer_array_store(target, access, offset, value, op)
        if access.get("members"):
            if op != "=":
                return self.refuse_lowering(
                    "Slang GLSL buffer block aggregate compound "
                    "store: assign a full aggregate value explicitly",
                    ";",
                )
            rhs = self.generate_expression_with_expected(value, access["type"])
            store = self.slang_byteaddress_aggregate_store(
                access["buffer"], offset, rhs, access
            )
            return self.terminate_slang_statement_lines(store)

        rhs = self.generate_expression_with_expected(value, access["type"])
        if access.get("matrix_columns"):
            if op != "=":
                store = self.slang_byteaddress_matrix_compound_store(
                    access["buffer"], offset, rhs, op, access
                )
            else:
                store = self.slang_byteaddress_matrix_store(
                    access["buffer"], offset, rhs, access
                )
            return self.terminate_slang_statement_lines(store)

        if op != "=":
            binary_op = glsl_buffer_compound_binary_operator(
                op, access["component_type"]
            )
            if binary_op is None:
                return (
                    self.slang_byteaddress_compound_store_diagnostic(op, access) + ";"
                )
            current = self.slang_byteaddress_load(access["buffer"], offset, access)
            rhs = f"({current} {binary_op} {rhs})"

        store = self.slang_byteaddress_leaf_store(access["buffer"], offset, rhs, access)
        return self.terminate_slang_statement_lines(store)

    def glsl_buffer_block_atomic_access(self, target):
        access = self.glsl_buffer_block_array_access(target)
        if access is not None:
            return access, access["offset_expr"]
        access = self.glsl_buffer_block_member_access(target)
        if access is None or access.get("runtime_array"):
            return None, None
        return access, access["offset"]

    def slang_byteaddress_atomic_operations(self):
        return {
            "atomicAdd": ("add", "InterlockedAdd", 2),
            "atomicMin": ("min", "InterlockedMin", 2),
            "atomicMax": ("max", "InterlockedMax", 2),
            "atomicAnd": ("and", "InterlockedAnd", 2),
            "atomicOr": ("or", "InterlockedOr", 2),
            "atomicXor": ("xor", "InterlockedXor", 2),
            "atomicExchange": ("exchange", "InterlockedExchange", 2),
            "atomicCompSwap": ("compare_exchange", "InterlockedCompareExchange", 3),
            "atomicCompareExchange": (
                "compare_exchange",
                "InterlockedCompareExchange",
                3,
            ),
        }

    def slang_byteaddress_atomic_helper_name(self, operation, component_type):
        return f"__crossgl_slang_byteaddress_atomic_{operation}_{component_type}"

    def slang_byteaddress_atomic_uses_uint_bits(self, operation, component_type):
        return component_type == "int" and operation in {
            "add",
            "and",
            "or",
            "xor",
            "exchange",
            "compare_exchange",
        }

    def generate_slang_byteaddress_atomic_helpers(self):
        if not self.required_byteaddress_atomic_helpers:
            return ""

        helpers = []
        for operation, intrinsic, component_type in sorted(
            self.required_byteaddress_atomic_helpers
        ):
            helper_name = self.slang_byteaddress_atomic_helper_name(
                operation, component_type
            )
            value_type = self.map_type(component_type)
            use_uint_bits = self.slang_byteaddress_atomic_uses_uint_bits(
                operation, component_type
            )
            original_type = "uint" if use_uint_bits else value_type
            return_value = "asint(original)" if use_uint_bits else "original"
            value_expr = "asuint(value)" if use_uint_bits else "value"
            if operation == "compare_exchange":
                compare_value_expr = (
                    "asuint(compareValue)" if use_uint_bits else "compareValue"
                )
                helpers.append(
                    f"{value_type} {helper_name}(RWByteAddressBuffer buffer, uint offset, "
                    f"{value_type} compareValue, {value_type} value) {{\n"
                    f"    {original_type} original;\n"
                    f"    buffer.{intrinsic}(offset, {compare_value_expr}, {value_expr}, original);\n"
                    f"    return {return_value};\n"
                    "}\n\n"
                )
                continue
            helpers.append(
                f"{value_type} {helper_name}(RWByteAddressBuffer buffer, uint offset, "
                f"{value_type} value) {{\n"
                f"    {original_type} original;\n"
                f"    buffer.{intrinsic}(offset, {value_expr}, original);\n"
                f"    return {return_value};\n"
                "}\n\n"
            )
        return "".join(helpers)

    def unsupported_glsl_buffer_block_atomic_call(
        self, target, operation, reason, access=None
    ):
        result_type = self.expression_result_type(target) or "uint"
        component_type = access.get("component_type") if access else None
        if component_type is not None:
            zero_value = "0u" if component_type == "uint" else "0"
        else:
            zero_value = "0u" if self.type_name_string(result_type) == "uint" else "0"
        return self.refuse_lowering(
            f"Slang GLSL buffer block atomic: {operation} {reason}", zero_value
        )

    def generate_glsl_buffer_block_atomic_call(self, func_name, args):
        operation_info = self.slang_byteaddress_atomic_operations().get(func_name)
        if operation_info is None or not args:
            return None

        operation, intrinsic, expected_args = operation_info
        target = args[0]
        access, offset = self.glsl_buffer_block_atomic_access(target)
        if access is None:
            return None
        if len(args) != expected_args:
            return self.unsupported_glsl_buffer_block_atomic_call(
                target,
                func_name,
                f"requires {expected_args} argument(s), got {len(args)}",
                access,
            )
        if access.get("readonly"):
            return self.unsupported_glsl_buffer_block_atomic_call(
                target, func_name, "cannot write readonly ByteAddressBuffer", access
            )
        if access.get("components") != 1 or access.get("matrix_columns"):
            return self.unsupported_glsl_buffer_block_atomic_call(
                target, func_name, "requires a scalar int or uint buffer member", access
            )
        if access.get("component_type") not in {"int", "uint"}:
            return self.unsupported_glsl_buffer_block_atomic_call(
                target,
                func_name,
                "currently supports only int or uint buffer members",
                access,
            )

        component_type = access["component_type"]
        helper_name = self.slang_byteaddress_atomic_helper_name(
            operation, component_type
        )
        self.required_byteaddress_atomic_helpers.add(
            (operation, intrinsic, component_type)
        )

        if operation == "compare_exchange":
            compare_value = self.generate_expression_with_expected(
                args[1], access["type"]
            )
            replacement = self.generate_expression_with_expected(
                args[2], access["type"]
            )
            return (
                f"{helper_name}({access['buffer']}, {offset}, "
                f"{compare_value}, {replacement})"
            )

        value = self.generate_expression_with_expected(args[1], access["type"])
        return f"{helper_name}({access['buffer']}, {offset}, {value})"
