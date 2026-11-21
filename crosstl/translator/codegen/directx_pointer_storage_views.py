"""Pointer storage views in the HLSL generator: struct and array views of resources and workgroup memory, element records and bindings, null and bare pointer selection and transfer, and escape checks."""

from ..ast import (
    ArrayAccessNode,
    ArrayType,
    AssignmentNode,
    BinaryOpNode,
    ForNode,
    IdentifierNode,
    PointerReinterpretNode,
    PointerType,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from . import hlsl_byte_records
from .directx_values import (
    HLSL_NULL_WORKGROUP_POINTER_CONTRACT,
    DirectXResourcePointerParameterError,
)
from .image_access_contracts import image_access_satisfies_requirement
from .pointer_reinterpret import (
    PointerReinterpretationError,
    same_scalar_lane_view,
    scalar_storage_layout,
)


class DirectXPointerStorageViewMixin:
    """Pointer storage view members of ``HLSLCodeGen``."""

    def prepare_hlsl_resource_pointer_array(self, node, following_statements):
        declaration_type = self.hlsl_storage_pointer_array_declaration_type(node)
        if declaration_type is None:
            return
        array_type, pointer_type, address_space = declaration_type
        binding = {
            "array_name": node.name,
            "parameter_name": node.name,
            "address_space": address_space,
            "declaration_node": node,
        }

        def reject(reason, source=node):
            raise self.hlsl_resource_pointer_array_error(
                f"DirectX cannot prove the bounded same-root contract for storage "
                f"pointer array '{node.name}'",
                binding=binding,
                node=source,
                reason=reason,
            )

        extent = self.literal_int_value(
            array_type.size, self.hlsl_current_visible_int_constants()
        )
        if not isinstance(extent, int) or isinstance(extent, bool) or extent <= 0:
            reject("unknown-extent")
        if getattr(node, "initial_value", None) is not None:
            reject("initializer-unsupported")
        if following_statements is None:
            following = []
        elif isinstance(following_statements, (list, tuple)):
            following = list(following_statements)
        else:
            following = [following_statements]

        def assignment_parts(statement):
            expression = self.hlsl_single_statement_expression(statement)
            if not isinstance(expression, AssignmentNode):
                return None
            target = getattr(expression, "target", getattr(expression, "left", None))
            value = getattr(expression, "value", getattr(expression, "right", None))
            operator = getattr(expression, "operator", getattr(expression, "op", "="))
            target_array = getattr(target, "array", getattr(target, "array_expr", None))
            target_index = getattr(target, "index", getattr(target, "index_expr", None))
            if (
                not isinstance(target, ArrayAccessNode)
                or getattr(target_array, "name", None) != node.name
                or operator != "="
            ):
                return None
            return expression, target, target_index, value

        def loop_assignment(statement, *, start):
            if not isinstance(statement, ForNode):
                return None
            loop_range = self.hlsl_for_index_range(statement)
            if loop_range is None or loop_range[1:] != (start, extent):
                return None
            body = getattr(statement, "body", None)
            statements = getattr(
                body, "statements", body if isinstance(body, list) else []
            )
            if len(statements) != 1:
                return None
            parts = assignment_parts(statements[0])
            if parts is None or getattr(parts[2], "name", None) != loop_range[0]:
                return None
            return loop_range, parts

        initialization_assignments = []
        replacement = None
        if following:
            direct_loop = loop_assignment(following[0], start=0)
        else:
            direct_loop = None
        if direct_loop is not None:
            _loop_range, parts = direct_loop
            if self.hlsl_resource_pointer_array_offset_has_side_effects(parts[3]):
                reject("side-effecting-offset", parts[3])
            replacement = self.hlsl_resource_pointer_binding(parts[3])
            initialization_assignments.append(parts)
        elif len(following) >= 2:
            seed = assignment_parts(following[0])
            chain = loop_assignment(following[1], start=1)
            seed_index = (
                self.literal_int_value(
                    seed[2], self.hlsl_current_visible_int_constants()
                )
                if seed is not None
                else None
            )
            if seed is None or seed_index != 0 or chain is None:
                reject("initialization-unproven", following[0])
            loop_range, chain_parts = chain
            chain_accesses = []
            for child in self.walk_ast(chain_parts[3]):
                if not isinstance(child, ArrayAccessNode):
                    continue
                child_array = getattr(
                    child, "array", getattr(child, "array_expr", None)
                )
                if getattr(child_array, "name", None) == node.name:
                    chain_accesses.append(child)
            if len(chain_accesses) != 1:
                reject("initialization-unproven", chain_parts[3])
            source_index = getattr(
                chain_accesses[0],
                "index",
                getattr(chain_accesses[0], "index_expr", None),
            )
            if not (
                isinstance(source_index, BinaryOpNode)
                and source_index.op == "-"
                and getattr(source_index.left, "name", None) == loop_range[0]
                and self.literal_int_value(
                    source_index.right, self.hlsl_current_visible_int_constants()
                )
                == 1
            ):
                reject("initialization-unproven", chain_parts[3])
            if self.hlsl_resource_pointer_array_offset_has_side_effects(
                seed[3]
            ) or self.hlsl_resource_pointer_array_offset_has_side_effects(
                chain_parts[3]
            ):
                reject("side-effecting-offset", chain_parts[3])
            replacement = self.hlsl_resource_pointer_binding(seed[3])
            initialization_assignments.extend((seed, chain_parts))
        else:
            reject("initialization-unproven")

        element_type = self.hlsl_buffer_pointer_pointee_type(pointer_type)
        qualifiers = {str(value).lower() for value in node.qualifiers or []}
        required_access = (
            "read"
            if address_space == "constant"
            or "const" in qualifiers
            or str(pointer_type.access_mode or "").lower() in {"read", "readonly"}
            else "read_write"
        )
        actual_type = (
            (replacement or {}).get("view_element_type")
            or (replacement or {}).get("source_element_type")
            or (replacement or {}).get("element_type")
        )
        reinterpretation = (replacement or {}).get("pointer_reinterpretation")
        if (
            replacement is None
            or self.hlsl_resource_type_name(replacement.get("resource_type"))
            not in {"StructuredBuffer", "RWStructuredBuffer"}
            or reinterpretation is not None
            and not reinterpretation.get("physical_storage_mapping")
            or self.map_type(actual_type) != self.map_type(element_type)
            or required_access == "read_write"
            and replacement.get("access") != "read_write"
        ):
            reject("backing-contract-unproven", initialization_assignments[0][3])

        used_names = set(self.current_identifier_reserved_names)
        used_names.update(self.local_variable_types)
        offset_name = self.hlsl_unique_local_identifier(
            f"{node.name}_offsets", used_names
        )
        self.current_identifier_reserved_names.add(offset_name)
        binding.update(
            {
                "kind": "resource-pointer-array",
                "extent": extent,
                "element_type": element_type,
                "view_element_type": (
                    replacement.get("view_element_type") or element_type
                ),
                "source_element_type": (
                    replacement.get("source_element_type")
                    or actual_type
                    or element_type
                ),
                "byte_offset": replacement.get("byte_offset", "0"),
                "pointer_reinterpretation": reinterpretation,
                "narrow_storage_origin": replacement.get("narrow_storage_origin"),
                "required_access": required_access,
                "access": required_access,
                "offset_array": offset_name,
                "root": replacement["root"],
                "resource_type": replacement["resource_type"],
                "initialization_target_ids": {
                    id(parts[1]) for parts in initialization_assignments
                },
                "declaration_id": id(node),
            }
        )
        self.current_hlsl_resource_pointer_aliases[node.name] = binding
        self.local_variable_types[offset_name] = f"int64_t[{extent}]"

    def generate_hlsl_resource_pointer_array_assignment(self, target, value, op):
        binding = self.hlsl_resource_pointer_array_for_access(target)
        if binding is None:
            return None
        parameter_binding = bool(binding.get("parameter_binding"))
        if not parameter_binding and (
            op != "="
            or id(target) not in binding.get("initialization_target_ids", set())
        ):
            raise self.hlsl_resource_pointer_array_error(
                "DirectX storage pointer array element reassignment is outside the "
                f"proven initialization plan for '{binding['array_name']}'",
                binding=binding,
                node=target,
                reason="element-reassignment-unsupported",
            )
        index_expression = getattr(target, "index", getattr(target, "index_expr", None))
        rendered_index, _index_range = self.hlsl_resource_pointer_array_index(
            binding, index_expression
        )
        selected_offset = f"{binding['offset_array']}[uint({rendered_index})]"
        if op in {"+=", "-="} and parameter_binding:
            rendered = self.generate_expression(value)
            return (
                f"{selected_offset} {op} "
                f"{self.hlsl_resource_pointer_delta_type(value)}({rendered})"
            )
        if op != "=":
            raise self.hlsl_resource_pointer_array_error(
                "DirectX storage pointer array offset mutation uses an unsupported "
                f"operator '{op}'",
                binding=binding,
                node=target,
                reason="element-reassignment-unsupported",
            )
        replacement = self.hlsl_resource_pointer_binding(value)
        actual_type = (
            (replacement or {}).get("view_element_type")
            or (replacement or {}).get("source_element_type")
            or (replacement or {}).get("element_type")
        )
        if (
            replacement is None
            or replacement.get("root") != binding.get("root")
            or self.map_type(actual_type) != self.map_type(binding.get("element_type"))
            or replacement.get("pointer_reinterpretation")
            != binding.get("pointer_reinterpretation")
            or str(replacement.get("byte_offset", "0"))
            != str(binding.get("byte_offset", "0"))
            or not self.hlsl_resource_pointer_access_satisfies(
                replacement.get("access"), binding.get("required_access")
            )
        ):
            raise self.hlsl_resource_pointer_array_error(
                "DirectX storage pointer array element assignment changes its "
                "proven same-root contract",
                binding=binding,
                node=value,
                reason="backing-contract-unproven",
            )
        return f"{selected_offset} = int64_t({replacement.get('offset', '0')})"

    def hlsl_resource_pointer_array_selection_binding(self, expression):
        binding = self.hlsl_resource_pointer_array_for_access(expression)
        if binding is None:
            return None
        index_expression = getattr(
            expression, "index", getattr(expression, "index_expr", None)
        )
        rendered_index, _index_range = self.hlsl_resource_pointer_array_index(
            binding, index_expression
        )
        return {
            **binding,
            "kind": "resource-pointer-array-element",
            "offset": f"{binding['offset_array']}[uint({rendered_index})]",
            "access": binding["required_access"],
            "is_pointer_alias": True,
            "pointer_array_binding": binding,
        }

    def hlsl_resource_pointer_array_bare_expression_error(self, expression):
        name = (
            expression
            if isinstance(expression, str)
            else getattr(expression, "name", None)
        )
        binding = self.current_hlsl_resource_pointer_aliases.get(name)
        if binding is None or binding.get("kind") != "resource-pointer-array":
            return
        raise self.hlsl_resource_pointer_array_error(
            "DirectX cannot emit storage pointer array "
            f"'{binding['array_name']}' as a first-class value",
            binding=binding,
            node=expression,
            reason="pointer-array-escape",
        )

    def hlsl_resource_pointer_array_selection_escape_error(self, binding, node):
        array_binding = binding and binding.get("pointer_array_binding")
        if array_binding is None:
            return
        raise self.hlsl_resource_pointer_array_error(
            "DirectX cannot materialize a selected storage pointer array element "
            "as another pointer alias",
            binding=array_binding,
            node=node,
            reason="pointer-element-escape",
        )

    # Keys that tie a selected element binding to its pointer array; a
    # materialized pointer alias drops them and keeps only the element's view.
    HLSL_RESOURCE_POINTER_ARRAY_ELEMENT_KEYS = frozenset(
        {
            "pointer_array_binding",
            "array_name",
            "parameter_name",
            "declaration_node",
            "declaration_id",
            "extent",
            "offset_array",
            "initialization_target_ids",
            "required_access",
        }
    )

    def hlsl_materialized_resource_pointer_array_element(self, binding, node):
        """Turn ``xs[i] + k`` into a plain pointer alias of the array's root.

        Every element of a concrete (non-parameter) storage pointer array shares
        one proven backing root, so copying a selected element only snapshots
        its offset, exactly like any other pointer alias initialisation.  The
        selection index was already proven in bounds when the element binding
        was built.  Elements of pointer-array parameters keep refusing.
        """

        array_binding = binding and binding.get("pointer_array_binding")
        if array_binding is None:
            return binding
        if array_binding.get("parameter_binding"):
            self.hlsl_resource_pointer_array_selection_escape_error(binding, node)
        materialized = {
            key: value
            for key, value in binding.items()
            if key not in self.HLSL_RESOURCE_POINTER_ARRAY_ELEMENT_KEYS
        }
        materialized["kind"] = "resource-pointer"
        materialized["is_pointer_alias"] = True
        return materialized

    def hlsl_resource_pointer_alias_expression(self, expression):
        if isinstance(expression, str):
            binding = self.current_hlsl_resource_pointer_aliases.get(expression)
            return bool(binding and binding.get("is_pointer_alias"))
        if isinstance(expression, (IdentifierNode, VariableNode)):
            binding = self.current_hlsl_resource_pointer_aliases.get(expression.name)
            return bool(binding and binding.get("is_pointer_alias"))
        if isinstance(expression, ArrayAccessNode):
            return self.hlsl_resource_pointer_array_for_access(expression) is not None
        if isinstance(expression, BinaryOpNode) and expression.op in {"+", "-"}:
            return self.hlsl_resource_pointer_alias_expression(
                expression.left
            ) or self.hlsl_resource_pointer_alias_expression(expression.right)
        return False

    def hlsl_workgroup_pointer_bare_expression_error(self, expression):
        name = (
            expression
            if isinstance(expression, str)
            else getattr(expression, "name", None) or self.expression_name(expression)
        )
        binding = self.current_hlsl_resource_pointer_aliases.get(name)
        if (
            binding is None
            or binding.get("kind") != "workgroup-pointer"
            or not binding.get("is_pointer_alias")
        ):
            return
        raise self.hlsl_workgroup_pointer_error(
            "DirectX cannot emit a workgroup pointer as a first-class value: "
            f"{name}",
            function_name=self.current_function_name,
            parameter_name=name,
            reason="bare-pointer-expression",
            node=expression if not isinstance(expression, str) else None,
        )

    def hlsl_null_pointer_expression(self, expression):
        if isinstance(expression, str):
            return expression.strip() == "nullptr"
        if isinstance(expression, (IdentifierNode, VariableNode)):
            return str(getattr(expression, "name", "")).strip() == "nullptr"
        return False

    @staticmethod
    def hlsl_null_workgroup_pointer_binding(element_type, *, offset="0"):
        """Binding for a workgroup pointer that is statically ``nullptr``.

        The parameter keeps its integer offset slot so every specialization has
        the same call shape, but no groupshared array backs it: analysis rejects
        any reachable observation and emission refuses to index it.
        """

        root, extent = HLSL_NULL_WORKGROUP_POINTER_CONTRACT
        return {
            "kind": "workgroup-pointer",
            "root": root,
            "root_kind": "null",
            "offset": offset,
            "array_size": extent,
            "element_type": element_type,
            "source_element_type": element_type,
            "resource_type": "groupshared",
            "access": "read_write",
            "selected_null": True,
        }

    def hlsl_storage_root_element_type(self, name, resource_type):
        """The logical element type of storage root ``name`` in ``resource_type``.

        A narrow source element widened into a 32-bit carrier keeps its source
        type, one carrier element per source element; a source element with the
        carrier's layout takes the carrier's spelling. ``None`` when ``name``
        has no source pointee type.
        """
        source_resource_type = self.local_variable_source_types.get(
            name
        ) or self.local_variable_types.get(name)
        source_element_type = self.hlsl_typed_buffer_element_type(
            source_resource_type,
            {
                "Buffer",
                "StructuredBuffer",
                "RWBuffer",
                "RWStructuredBuffer",
                "RasterizerOrderedBuffer",
                "RasterizerOrderedStructuredBuffer",
            },
        ) or self.hlsl_pointer_element_type(source_resource_type)
        source_element_type = (
            self.hlsl_bfloat16_scalar_type_name(source_element_type)
            or source_element_type
        )
        if source_element_type is None:
            return None
        physical_element_type = self.hlsl_resource_pointer_element_type(resource_type)
        source_layout = scalar_storage_layout(source_element_type)
        if source_layout is not None and source_layout == scalar_storage_layout(
            physical_element_type
        ):
            return physical_element_type
        return source_element_type

    def hlsl_resource_pointer_binding(self, expression):
        if expression is None:
            return None

        if isinstance(expression, ArrayAccessNode):
            pointer_array_binding = self.hlsl_resource_pointer_array_selection_binding(
                expression
            )
            if pointer_array_binding is not None:
                return pointer_array_binding
            binding = self.hlsl_resource_pointer_binding(expression.array)
            if binding is None:
                return None
            index = self.generate_expression(expression.index)
            binding["offset"] = self.hlsl_resource_pointer_offset_sum(
                binding.get("offset"), index
            )
            return binding

        if isinstance(expression, PointerReinterpretNode):
            binding = self.hlsl_resource_pointer_binding(expression.expression)
            if binding is None:
                return None
            return self.hlsl_pointer_reinterpret_binding(expression, binding)

        if isinstance(expression, (str, IdentifierNode, VariableNode)):
            raw_name = (
                expression
                if isinstance(expression, str)
                else getattr(expression, "name", None)
            )
            if raw_name in self.current_hlsl_resource_pointer_aliases:
                binding = self.current_hlsl_resource_pointer_aliases[raw_name]
                if binding.get("kind") == "resource-pointer-array":
                    return None
                if binding.get("deferred_unbound"):
                    raise DirectXResourcePointerParameterError(
                        "DirectX deferred storage pointer "
                        f"'{self.current_function_name}.{raw_name}' is used before "
                        "a concrete backing resource is assigned",
                        function_name=self.current_function_name,
                        parameter_name=raw_name,
                        address_space=binding.get("address_space"),
                        expected_access=binding.get("access"),
                        reason="deferred-use-before-assignment",
                        source_location=(
                            getattr(expression, "source_location", None)
                            if not isinstance(expression, str)
                            else binding.get("declaration_source_location")
                        ),
                    )
                return dict(binding)

            resource_type = self.current_hlsl_parameter_resource_types.get(
                raw_name
            ) or self.hlsl_buffer_helper_resource_type(expression)
            resource_name = self.hlsl_resource_type_name(resource_type)
            if resource_name not in {
                "Buffer",
                "StructuredBuffer",
                "RWBuffer",
                "RWStructuredBuffer",
                "RasterizerOrderedBuffer",
                "RasterizerOrderedStructuredBuffer",
            }:
                return None
            access = (
                "read_write"
                if resource_name.startswith(("RW", "RasterizerOrdered"))
                else "read"
            )
            contract_element_type = self.hlsl_storage_root_element_type(
                raw_name, resource_type
            )
            rendered_root = self.hlsl_identifier_name(raw_name)
            offset = self.current_hlsl_resource_pointer_offsets.get(rendered_root, "0")
            if contract_element_type is not None:
                return self.hlsl_resource_pointer_parameter_alias_binding(
                    {
                        "address_space": "storage",
                        "access": access,
                        "element_type": contract_element_type,
                        "resource_type": resource_type,
                    },
                    root=rendered_root,
                    offset=offset,
                    logical_storage=True,
                )
            return {
                "root": rendered_root,
                "offset": offset,
                "resource_type": resource_type,
                "element_type": self.hlsl_resource_pointer_element_type(resource_type),
                "source_element_type": self.hlsl_resource_pointer_element_type(
                    resource_type
                ),
                "access": access,
            }

        if isinstance(expression, TernaryOpNode):
            condition = self.literal_int_value(
                expression.condition,
                self.current_hlsl_visible_int_constants or self.literal_int_constants,
            )
            if condition is not None:
                selected = expression.true_expr if condition else expression.false_expr
                binding = self.hlsl_resource_pointer_binding(selected)
                if binding is not None:
                    return binding
                if self.hlsl_null_pointer_expression(selected):
                    alternate = (
                        expression.false_expr if condition else expression.true_expr
                    )
                    binding = self.hlsl_resource_pointer_binding(alternate)
                    if (
                        binding is not None
                        and binding.get("kind") == "workgroup-pointer"
                    ):
                        binding["selected_null"] = True
                        return binding
                return None
            true_binding = self.hlsl_resource_pointer_binding(expression.true_expr)
            false_binding = self.hlsl_resource_pointer_binding(expression.false_expr)
            if (
                true_binding is not None
                and true_binding.get("kind") == "workgroup-pointer"
                or false_binding is not None
                and false_binding.get("kind") == "workgroup-pointer"
            ):
                raise self.hlsl_workgroup_pointer_error(
                    "DirectX cannot prove the backing object selected by a "
                    "workgroup pointer conditional",
                    function_name=self.current_function_name,
                    reason="conditional-backing-unresolved",
                    node=expression,
                )
            return None

        if isinstance(expression, UnaryOpNode) and expression.op == "&":
            operand = expression.operand
            if not isinstance(operand, ArrayAccessNode):
                binding = self.hlsl_resource_pointer_binding(operand)
                if binding is not None and binding.get("kind") == "workgroup-pointer":
                    raise self.hlsl_workgroup_pointer_error(
                        "DirectX cannot lower the address of a workgroup pointer",
                        function_name=self.current_function_name,
                        parameter_name=self.expression_name(operand),
                        reason="address-of-pointer",
                        node=expression,
                    )
                return None
            binding = self.hlsl_resource_pointer_binding(operand.array)
            if binding is None:
                return None
            index = self.generate_expression(operand.index)
            binding["offset"] = self.hlsl_resource_pointer_offset_sum(
                binding.get("offset"), index
            )
            return binding

        if isinstance(expression, BinaryOpNode) and expression.op in {"+", "-"}:
            binding = self.hlsl_resource_pointer_binding(expression.left)
            delta = expression.right
            if (
                expression.op == "-"
                and binding is not None
                and binding.get("kind") == "workgroup-pointer"
                and self.hlsl_resource_pointer_binding(expression.right) is not None
            ):
                raise self.hlsl_workgroup_pointer_error(
                    "DirectX workgroup pointer difference is unsupported",
                    function_name=self.current_function_name,
                    reason="pointer-difference-unsupported",
                    node=expression,
                )
            if binding is None and expression.op == "+":
                binding = self.hlsl_resource_pointer_binding(expression.right)
                delta = expression.left
            if binding is None:
                return None
            rendered_delta = self.generate_expression(delta)
            if expression.op == "-":
                rendered_delta = f"-({rendered_delta})"
            binding["offset"] = self.hlsl_resource_pointer_offset_sum(
                binding.get("offset"), rendered_delta
            )
            return binding

        return None

    def hlsl_same_scalar_lane_gather_binding(
        self,
        binding,
        target_type,
        *,
        source_type_name,
        source_layout,
        view_layout,
    ):
        """Gather a ``vec<T, N>`` view of a 16-bit structured buffer.

        DirectX keeps one 16-bit structured-buffer element per logical ``T``,
        so a vector view of a ``T`` buffer is N consecutive element reads, each
        through the buffer's normal logical conversion.
        """

        if source_layout is None or source_layout.bit_width != 16:
            return None
        lane_view = same_scalar_lane_view(target_type, source_layout)
        if (
            lane_view is None
            or view_layout != source_layout
            or binding.get("kind") == "workgroup-pointer"
            or binding.get("pointer_reinterpretation") is not None
            or str(binding.get("byte_offset", "0")) not in {"0", "0u"}
        ):
            return None
        lanes, component = lane_view
        vector_type = f"vec<{component}, {lanes}>"
        rendered_offset = str(binding.get("offset", "0"))
        byte_offset = rendered_offset
        if rendered_offset not in {"0", "0u"}:
            byte_offset = f"({rendered_offset} * {source_layout.byte_width})"
        return {
            **binding,
            "offset": "0",
            "byte_offset": byte_offset,
            "element_type": vector_type,
            "view_element_type": vector_type,
            "source_element_type": source_type_name,
            "pointer_reinterpretation": {
                "source_layout": source_layout,
                "target_layout": source_layout,
                "target_width": lanes,
                "lane_gather": True,
            },
        }

    def hlsl_storage_struct_view_error(
        self,
        expression,
        *,
        source_type,
        target_type,
        reason,
        detail,
        binding=None,
        access="read",
        alignment=4,
    ):
        pointer_type = getattr(expression, "target_type", None)
        address_space = (
            getattr(pointer_type, "address_space", None)
            or (binding or {}).get("address_space")
            or "storage"
        )
        # Preserve native Metal qualifiers in the IR for round-trip fidelity, but
        # expose DirectX's target-neutral storage contract in diagnostics.
        if str(address_space).lower() in {"device", "global", "storage"}:
            address_space = "storage"
        raise PointerReinterpretationError(
            "DirectX cannot preserve a storage aggregate view from "
            f"'{source_type}' to '{target_type}': {detail}",
            source_type=source_type,
            target_type=target_type,
            address_space=address_space,
            alignment=alignment,
            access=access,
            target_backend="directx",
            reason=reason,
            source_location=getattr(expression, "source_location", None),
        )

    def hlsl_workgroup_record_transfer(self, target, value):
        def reinterpretation(expression):
            if (
                isinstance(expression, UnaryOpNode)
                and expression.op == "*"
                and not getattr(expression, "is_postfix", False)
                and isinstance(expression.operand, PointerReinterpretNode)
            ):
                return expression.operand
            return None

        destination = reinterpretation(target)
        source = reinterpretation(value)
        if destination is None or source is None:
            return None
        destination_space = str(destination.target_type.address_space or "").lower()
        source_space = str(source.target_type.address_space or "").lower()
        target_name = self.type_name_string(destination.target_type.pointee_type)
        if (
            destination_space not in {"threadgroup", "workgroup"}
            or source_space not in {"device", "constant", "storage", "global"}
            or target_name not in self.structs_by_name
        ):
            return None
        if target_name != self.type_name_string(source.target_type.pointee_type):
            self.hlsl_storage_struct_view_error(
                source,
                source_type=None,
                target_type=target_name,
                reason="storage-workgroup-aggregate-target-mismatch",
                detail="the source and destination record types differ",
            )
        return destination, source

    def hlsl_workgroup_record_layout(self, reinterpretation, constants):
        target_name = self.type_name_string(reinterpretation.target_type.pointee_type)
        struct = self.structs_by_name[target_name]

        def reject(detail):
            self.hlsl_storage_struct_view_error(
                reinterpretation,
                source_type=None,
                target_type=target_name,
                reason="unsupported-storage-workgroup-aggregate-layout",
                detail=detail,
            )

        members = [
            member
            for member in struct.members
            if not self.hlsl_static_struct_member(member)
        ]
        if (
            len(members) != 1
            or getattr(struct, "generic_params", None)
            or getattr(struct, "inheritance", None)
            or self.hlsl_union_layout_for_type(target_name) is not None
        ):
            reject("the record must have one fixed byte-array member")
        alignment = 1
        attributes = list(getattr(struct, "attributes", None) or [])
        if attributes:
            attribute = attributes[0]
            arguments = list(getattr(attribute, "arguments", None) or [])
            if (
                len(attributes) != 1
                or attribute.name != "metal_alignas"
                or len(arguments) != 1
            ):
                reject("the record has unsupported layout attributes")
            alignment = self.literal_int_value(arguments[0], constants)
            if (
                not isinstance(alignment, int)
                or isinstance(alignment, bool)
                or alignment <= 0
                or alignment & (alignment - 1)
            ):
                reject("the alignment must be a fixed positive power of two")
        member = members[0]
        member_type = getattr(member, "member_type", None)
        if (
            not isinstance(member_type, ArrayType)
            or getattr(member, "attributes", None)
            or getattr(member, "resource_qualifiers", None)
        ):
            reject("the member must be an unqualified fixed byte array")
        layout = scalar_storage_layout(self.type_name_string(member_type.element_type))
        extent = self.literal_int_value(member_type.size, constants)
        if (
            layout is None
            or layout.kind != "integer"
            or layout.bit_width != 8
            or not isinstance(extent, int)
            or isinstance(extent, bool)
            or extent <= 0
            or extent % alignment
        ):
            reject("the byte extent must be fixed and have no alignment padding")
        return extent, alignment

    def generate_hlsl_workgroup_record_transfer(
        self, target, value, operator, *, statement_context
    ):
        transfer = self.hlsl_workgroup_record_transfer(target, value)
        if transfer is None:
            return None
        destination, source = transfer
        extent, alignment = self.hlsl_workgroup_record_layout(
            source, self.current_hlsl_visible_int_constants
        )
        target_name = self.type_name_string(source.target_type.pointee_type)

        def reject(reason, detail):
            self.hlsl_storage_struct_view_error(
                source,
                source_type=None,
                target_type=target_name,
                reason=reason,
                detail=detail,
                alignment=alignment,
            )

        if operator != "=" or not statement_context:
            reject(
                "storage-workgroup-aggregate-expression-unsupported",
                "the record copy must be a standalone simple assignment",
            )
        if any(
            getattr(view.target_type, "resource_qualifiers", None) for view in transfer
        ):
            reject(
                "unsupported-storage-workgroup-aggregate-layout",
                "qualified pointer views cannot be expanded into ordinary word accesses",
            )
        if any(
            self.hlsl_private_pointer_expression_has_side_effects(view.expression)
            for view in transfer
        ):
            reject(
                "storage-workgroup-aggregate-offset-unprovable",
                "the source and destination offsets must not have side effects",
            )
        destination_binding = self.hlsl_resource_pointer_binding(destination.expression)
        source_binding = self.hlsl_resource_pointer_binding(source.expression)
        if (
            destination_binding is None
            or destination_binding.get("kind") != "workgroup-pointer"
            or source_binding is None
            or source_binding.get("kind") == "workgroup-pointer"
            or not source_binding.get("root")
        ):
            reject(
                "storage-workgroup-aggregate-backing-unresolved",
                "the copy requires a storage-buffer source and a shared-array destination",
            )
        layouts = [
            scalar_storage_layout(
                binding.get("source_element_type") or binding.get("element_type")
            )
            for binding in (destination_binding, source_binding)
        ]
        if (
            layouts[0] is None
            or layouts[0].bit_width != 32
            or layouts[0] != layouts[1]
            or any(
                binding.get("pointer_reinterpretation") is not None
                or str(binding.get("byte_offset", "0")) not in {"0", "0u"}
                for binding in (destination_binding, source_binding)
            )
        ):
            reject(
                "storage-workgroup-aggregate-backing-mismatch",
                "the backings must have the same non-reinterpreted 32-bit scalar layout",
            )
        if alignment > layouts[0].byte_width or extent % layouts[0].byte_width:
            reject(
                "unsupported-storage-workgroup-aggregate-alignment",
                "the record must occupy whole backing words with scalar-guaranteed alignment",
            )
        if not image_access_satisfies_requirement("read", source_binding.get("access")):
            reject(
                "storage-workgroup-aggregate-source-not-readable",
                "the source is not readable",
            )
        if not getattr(
            destination.target_type, "is_mutable", True
        ) or not image_access_satisfies_requirement(
            "write", destination_binding.get("access")
        ):
            reject(
                "storage-workgroup-aggregate-destination-read-only",
                "the destination is not writable",
            )
        if not getattr(destination, "_hlsl_record_transfer_bounds_checked", False):
            reject(
                "storage-workgroup-aggregate-offset-unprovable",
                "the complete destination range has not been proven within its backing",
            )
        # Bind both addresses before any write, including offsets loaded from memory.
        assignments = []
        offsets = []
        for role, binding in (
            ("source", source_binding),
            ("destination", destination_binding),
        ):
            if binding.get("offset") is None:
                reject(
                    "storage-workgroup-aggregate-offset-unprovable",
                    "a backing offset is unresolved",
                )
            used_names = set(self.local_variable_types) | set(
                self.global_variable_types
            )
            used_names.update(self.current_identifier_reserved_names)
            used_names.update(self.current_identifier_aliases.values())
            name = self.hlsl_unique_local_identifier(
                f"__crossgl_copy_{role}", used_names
            )
            self.current_identifier_reserved_names.add(name)
            self.local_variable_types[name] = "uint"
            assignments.append(f"uint {name} = uint({binding.get('offset', '0')})")
            offsets.append(name)
        for index in range(extent // layouts[0].byte_width):
            left = f"{destination_binding['root']}[{offsets[1]} + {index}u]"
            right = f"{source_binding['root']}[{offsets[0]} + {index}u]"
            assignments.append(f"{left} = {right}")
        return "\n".join(assignments)

    def hlsl_storage_struct_view_offset_is_provable(self, expression):
        if isinstance(expression, (str, IdentifierNode, VariableNode)):
            return True
        if isinstance(expression, PointerReinterpretNode):
            return self.hlsl_storage_struct_view_offset_is_provable(
                expression.expression
            )
        if not isinstance(expression, BinaryOpNode) or expression.op not in {"+", "-"}:
            return False

        left_binding = self.hlsl_resource_pointer_binding(expression.left)
        right_binding = self.hlsl_resource_pointer_binding(expression.right)
        if left_binding is not None and right_binding is None:
            pointer_expression = expression.left
            delta_expression = expression.right
        elif (
            expression.op == "+" and right_binding is not None and left_binding is None
        ):
            pointer_expression = expression.right
            delta_expression = expression.left
        else:
            return False

        delta_value = self.literal_int_value(
            delta_expression, self.current_hlsl_visible_int_constants
        )
        if delta_value is None:
            delta_type = self.hlsl_integer_arithmetic_type_info(
                self.expression_result_type(delta_expression)
            )
            if delta_type is None or delta_type["width"] != 1:
                return False
        return self.hlsl_storage_struct_view_offset_is_provable(pointer_expression)

    def hlsl_storage_struct_view_layout(self, expression, binding):
        pointer_type = getattr(expression, "target_type", None)
        target_type = getattr(pointer_type, "pointee_type", None)
        target_name = self.type_name_string(target_type)
        struct_node = self.structs_by_name.get(target_name)
        if struct_node is None:
            return None

        address_space = str(
            getattr(pointer_type, "address_space", None)
            or binding.get("address_space")
            or "storage"
        ).lower()
        if address_space not in {"device", "global", "storage"}:
            return None

        source_name = (
            binding.get("view_element_type")
            or binding.get("element_type")
            or binding.get("source_element_type")
            or "unknown"
        )
        reinterpretation = binding.get("pointer_reinterpretation")
        source_layout = (
            reinterpretation.get("target_layout")
            if reinterpretation is not None
            else scalar_storage_layout(source_name)
        )

        def reject(reason, detail, *, source_type=source_name):
            self.hlsl_storage_struct_view_error(
                expression,
                source_type=self.type_name_string(source_type),
                target_type=target_name,
                reason=reason,
                detail=detail,
                binding=binding,
                access=binding.get("access", "read"),
            )

        if binding.get("access") == "write":
            reject(
                "storage-view-read-unsupported",
                "the backing pointer does not provide readable storage",
            )
        byte_source = hlsl_byte_records.is_byte_layout(source_layout)
        if not byte_source and (
            source_layout is None
            or source_layout.bit_width != 32
            or self.map_type(source_layout.name) not in {"float", "int", "uint"}
        ):
            reject(
                "unsupported-storage-view-source-layout",
                "the backing pointer must expose homogeneous 32-bit scalar words",
            )
        _alignment, canonical_attributes = hlsl_byte_records.record_alignment(
            struct_node,
            lambda value: self.literal_int_value(value, self.literal_int_constants),
        )
        if (
            not canonical_attributes
            or getattr(struct_node, "inheritance", None)
            or self.hlsl_union_layout_for_type(target_type) is not None
        ):
            reject(
                "unsupported-storage-view-aggregate-layout",
                "the target struct has noncanonical layout metadata",
            )
        members = [
            member
            for member in getattr(struct_node, "members", []) or []
            if not self.hlsl_static_struct_member(member)
        ]
        if len(members) != 1:
            reject(
                "heterogeneous-storage-view-layout",
                "the target must have exactly one fixed scalar-array member",
            )
        member = members[0]
        if getattr(member, "attributes", None):
            reject(
                "unsupported-storage-view-aggregate-layout",
                "the target array member has noncanonical layout metadata",
            )
        member_type = getattr(member, "member_type", None)
        if not isinstance(member_type, ArrayType):
            reject(
                "heterogeneous-storage-view-layout",
                "the sole target member is not a fixed scalar array",
            )
        extent = self.literal_int_value(
            getattr(member_type, "size", None), self.literal_int_constants
        )
        if extent is None or extent <= 0:
            reject(
                "dynamic-storage-view-extent",
                "the target word-array extent is not a positive constant",
            )
        target_element_name = self.type_name_string(member_type.element_type)
        target_layout = scalar_storage_layout(target_element_name)
        if byte_source:
            if not hlsl_byte_records.storage_byte_copy_allowed(
                source_layout, target_layout
            ):
                reject(
                    "unsupported-storage-view-target-layout",
                    "a byte storage view can only fill a fixed byte array",
                )
        elif (
            target_layout is None
            or target_layout.bit_width != 32
            or self.map_type(target_element_name) not in {"float", "int", "uint"}
        ):
            reject(
                "unsupported-storage-view-target-layout",
                "the target array must contain homogeneous 32-bit scalar words",
            )
        return {
            "source_layout": source_layout,
            "target_layout": target_layout,
            "target_type": target_name,
            "member": member.name,
            "extent": extent,
        }

    def hlsl_storage_struct_view_has_resource_source(self, expression):
        if isinstance(expression, (str, IdentifierNode, VariableNode)):
            return self.hlsl_resource_pointer_binding(expression) is not None
        if isinstance(expression, PointerReinterpretNode):
            return self.hlsl_storage_struct_view_has_resource_source(
                expression.expression
            )
        if isinstance(expression, BinaryOpNode):
            return self.hlsl_storage_struct_view_has_resource_source(
                expression.left
            ) or self.hlsl_storage_struct_view_has_resource_source(expression.right)
        if isinstance(expression, TernaryOpNode):
            return self.hlsl_storage_struct_view_has_resource_source(
                expression.true_expr
            ) or self.hlsl_storage_struct_view_has_resource_source(
                expression.false_expr
            )
        return False

    def hlsl_storage_struct_view_parts(self, expression):
        if not (
            isinstance(expression, UnaryOpNode)
            and expression.op == "*"
            and not getattr(expression, "is_postfix", False)
            and isinstance(expression.operand, PointerReinterpretNode)
        ):
            return None
        if self.hlsl_private_scalar_struct_reinterpret_node(expression) is not None:
            return None
        reinterpret = expression.operand
        pointer_type = getattr(reinterpret, "target_type", None)
        target_type = getattr(pointer_type, "pointee_type", None)
        target_name = self.type_name_string(target_type)
        if target_name not in self.structs_by_name:
            return None
        address_space = str(
            getattr(pointer_type, "address_space", None) or "storage"
        ).lower()
        if address_space in {"thread", "private", "function"}:
            return None
        if address_space not in {"device", "global", "storage"}:
            self.hlsl_storage_struct_view_error(
                reinterpret,
                source_type=self.type_name_string(
                    self.expression_result_type(reinterpret.expression)
                )
                or "unknown",
                target_type=target_name,
                reason="unsupported-storage-view-address-space",
                detail=f"address space '{address_space}' is not typed storage",
            )
        if self.hlsl_private_pointer_expression_has_side_effects(
            reinterpret.expression
        ):
            self.hlsl_storage_struct_view_error(
                reinterpret,
                source_type=self.type_name_string(
                    self.expression_result_type(reinterpret.expression)
                )
                or "unknown",
                target_type=target_name,
                reason="side-effecting-storage-view-offset",
                detail="the source pointer or offset has side effects",
            )
        binding = self.hlsl_resource_pointer_binding(reinterpret.expression)
        if binding is None or not binding.get("root"):
            has_resource_source = self.hlsl_storage_struct_view_has_resource_source(
                reinterpret.expression
            )
            self.hlsl_storage_struct_view_error(
                reinterpret,
                source_type=self.type_name_string(
                    self.expression_result_type(reinterpret.expression)
                )
                or "unknown",
                target_type=target_name,
                reason=(
                    "unprovable-storage-view-offset"
                    if has_resource_source
                    else "missing-storage-view-backing"
                ),
                detail=(
                    "the source pointer cannot be reduced to one typed backing and "
                    "element offset"
                    if has_resource_source
                    else "the source has no typed storage-buffer backing object"
                ),
            )
        if binding.get("offset") is None or not (
            self.hlsl_storage_struct_view_offset_is_provable(reinterpret.expression)
        ):
            self.hlsl_storage_struct_view_error(
                reinterpret,
                source_type=(
                    binding.get("view_element_type")
                    or binding.get("element_type")
                    or binding.get("source_element_type")
                    or "unknown"
                ),
                target_type=target_name,
                reason="unprovable-storage-view-offset",
                detail=(
                    "the source element offset cannot be proven from typed pointer "
                    "arithmetic"
                ),
                binding=binding,
            )
        layout = self.hlsl_storage_struct_view_layout(reinterpret, binding)
        return reinterpret, binding, layout

    def hlsl_storage_struct_view_word_expression(
        self, pointer_expression, index, source_layout, target_layout
    ):
        source_value = self.generate_hlsl_resource_pointer_access(
            pointer_expression, index
        )
        if source_value is None:
            return None
        if source_layout.name == target_layout.name:
            return source_value
        source_bits = (
            f"asuint({source_value})"
            if source_layout.kind == "floating" or source_layout.signed
            else source_value
        )
        if target_layout.kind == "floating":
            return f"asfloat({source_bits})"
        if target_layout.signed:
            return f"asint({source_bits})"
        return source_bits

    def generate_hlsl_storage_struct_view_copy(
        self,
        target_name,
        target_type,
        expression,
        indent,
        *,
        declaration=None,
    ):
        parts = self.hlsl_storage_struct_view_parts(expression)
        if parts is None:
            return None
        reinterpret, _binding, layout = parts
        expected_type = self.type_name_string(target_type)
        if expected_type != layout["target_type"]:
            self.hlsl_storage_struct_view_error(
                reinterpret,
                source_type=layout["source_layout"].name,
                target_type=layout["target_type"],
                reason="storage-view-target-mismatch",
                detail=f"the dereferenced value cannot initialize '{expected_type}'",
            )
        indent_str = "    " * indent
        statements = []
        if declaration is not None:
            statements.append(f"{indent_str}{declaration};")
        for index in range(layout["extent"]):
            value = self.hlsl_storage_struct_view_word_expression(
                reinterpret.expression,
                index,
                layout["source_layout"],
                layout["target_layout"],
            )
            statements.append(
                f"{indent_str}{target_name}.{layout['member']}[{index}] = {value};"
            )
        return "\n".join(statements) + "\n"

    def hlsl_storage_struct_view_write_error(self, target):
        if self.hlsl_private_scalar_struct_reinterpret_node(target) is not None:
            return
        found = None

        def visit(expression):
            nonlocal found
            if found is not None or expression is None:
                return
            if isinstance(expression, PointerReinterpretNode):
                target_type = getattr(expression.target_type, "pointee_type", None)
                address_space = str(
                    getattr(expression.target_type, "address_space", None) or "storage"
                ).lower()
                if self.type_name_string(
                    target_type
                ) in self.structs_by_name and address_space in {
                    "device",
                    "global",
                    "storage",
                }:
                    found = expression
                    return
            if not hasattr(expression, "__dict__"):
                return
            for key, child in vars(expression).items():
                if key in {"parent", "annotations", "name", "args"}:
                    continue
                if isinstance(child, (list, tuple)):
                    for item in child:
                        visit(item)
                else:
                    visit(child)

        visit(target)
        if found is None:
            return
        target_name = self.type_name_string(
            getattr(found.target_type, "pointee_type", None)
        )
        self.hlsl_storage_struct_view_error(
            found,
            source_type=self.type_name_string(
                self.expression_result_type(found.expression)
            )
            or "unknown",
            target_type=target_name,
            reason="storage-view-write-unsupported",
            detail=(
                "writes through a reinterpreted typed-storage aggregate are "
                "unsupported"
            ),
            access="write",
        )

    def hlsl_const_auto_resource_pointer_pointee_type(self, node, vtype, binding):
        if not isinstance(vtype, PointerType):
            return None
        pointee_type = self.type_name_string(getattr(vtype, "pointee_type", None))
        if str(pointee_type or "").strip().lower() != "auto":
            return None

        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        if not qualifiers.intersection({"const", "constant"}):
            return None
        if self.hlsl_resource_type_name(binding.get("resource_type")) not in {
            "StructuredBuffer",
            "RWStructuredBuffer",
        }:
            return None
        if binding.get("pointer_reinterpretation") is not None:
            return None

        backing_element_type = binding.get("element_type")
        if backing_element_type is None:
            return None
        mapped_element_type = str(self.map_type(backing_element_type) or "").strip()
        if mapped_element_type.lower() in {"", "auto", "let", "void"} or any(
            marker in mapped_element_type for marker in ("*", "&", "[")
        ):
            return None
        return backing_element_type
