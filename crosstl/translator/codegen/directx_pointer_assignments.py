"""Pointer assignments and aliases in the HLSL generator: base and post-increment updates, declarations and steps, unary operations, load helpers, reinterpreted reads and dereferences."""

from hashlib import sha1

from ..ast import (
    ArrayAccessNode,
    ArrayType,
    BinaryOpNode,
    CooperativeMatrixOpNode,
    CooperativeMatrixType,
    FunctionCallNode,
    IdentifierNode,
    PointerType,
    UnaryOpNode,
    VariableNode,
)
from .directx_pointer_alias_errors import DirectXResourcePointerAliasError
from .directx_values import (
    DirectXResourcePointerParameterError,
    _directx_cooperative_matrix_contract_type,
)
from .hlsl_word_views import hlsl_word_view_element
from .pointer_reinterpret import PointerReinterpretationError
from .storage_word_views import is_same_width_word_view


class DirectXPointerAssignmentMixin:
    """Pointer assignment members of ``HLSLCodeGen``."""

    def generate_hlsl_resource_pointer_alias_declaration(self, node, indent):
        pointer_array_declaration = (
            self.generate_hlsl_resource_pointer_array_declaration(node, indent)
        )
        if pointer_array_declaration is not None:
            return pointer_array_declaration
        vtype = getattr(node, "var_type", getattr(node, "vtype", None))
        workgroup_declaration = self.hlsl_workgroup_pointer_declaration(
            node, self.current_hlsl_resource_pointer_aliases
        )
        if not isinstance(vtype, PointerType) and not workgroup_declaration:
            return None
        initial_value = getattr(node, "initial_value", None)
        binding = self.hlsl_materialized_resource_pointer_array_element(
            self.hlsl_resource_pointer_binding(initial_value), node
        )
        if (
            binding is None
            and initial_value is None
            and self.hlsl_resource_pointer_parameter(node)
        ):
            pointer_type = self.hlsl_resource_pointer_parameter_type_node(node)
            pointee_type = getattr(pointer_type, "pointee_type", None)
            if (
                isinstance(pointee_type, (PointerType, ArrayType))
                or pointee_type is None
            ):
                raise DirectXResourcePointerParameterError(
                    "DirectX deferred storage pointer "
                    f"'{self.current_function_name}.{node.name}' has an unsupported "
                    "pointee type",
                    function_name=self.current_function_name,
                    parameter_name=node.name,
                    address_space=self.hlsl_resource_pointer_parameter_address_space(
                        node
                    ),
                    reason="unsupported-pointee-type",
                    source_location=getattr(node, "source_location", None),
                )
            element_type = self.type_name_string(pointee_type)
            access = self.hlsl_resource_pointer_parameter_access(node)
            used_names = set(self.current_identifier_reserved_names)
            used_names.update(self.local_variable_types)
            offset_name = self.hlsl_unique_local_identifier(
                f"{node.name}_offset", used_names
            )
            self.current_identifier_reserved_names.add(offset_name)
            self.local_variable_types[offset_name] = "int64_t"
            self.current_hlsl_resource_pointer_aliases[node.name] = {
                "kind": "resource-pointer",
                "root": None,
                "root_kind": "deferred",
                "offset": offset_name,
                "element_type": element_type,
                "source_element_type": element_type,
                "resource_type": None,
                "address_space": self.hlsl_resource_pointer_parameter_address_space(
                    node
                ),
                "access": access,
                "is_pointer_alias": True,
                "deferred_unbound": True,
                "declaration_source_location": getattr(node, "source_location", None),
            }
            return f"{'    ' * indent}int64_t {offset_name} = int64_t(0);\n"
        if binding is None:
            if workgroup_declaration:
                raise self.hlsl_workgroup_pointer_error(
                    "DirectX cannot lower workgroup pointer alias "
                    f"'{node.name}' without a concrete shared-array backing object",
                    function_name=self.current_function_name,
                    parameter_name=node.name,
                    reason="backing-unresolved",
                    node=node,
                )
            return None

        is_workgroup_pointer = binding.get("kind") == "workgroup-pointer"
        if is_workgroup_pointer and binding.get("array_size") is None:
            raise self.hlsl_workgroup_pointer_error(
                "DirectX cannot lower workgroup pointer alias "
                f"'{node.name}' because its backing array extent is unresolved",
                function_name=self.current_function_name,
                parameter_name=node.name,
                reason="missing-concrete-size",
                node=node,
            )

        declared_element_type = (
            self.hlsl_workgroup_pointer_element_type(node)
            if workgroup_declaration
            else self.hlsl_buffer_pointer_pointee_type(vtype)
        )
        backing_element_type = binding.get("element_type")
        inferred_element_type = self.hlsl_const_auto_resource_pointer_pointee_type(
            node, vtype, binding
        )
        if inferred_element_type is not None:
            declared_element_type = inferred_element_type
            self.local_variable_types[node.name] = f"{inferred_element_type}*"
        if (
            declared_element_type is not None
            and backing_element_type is not None
            and self.map_type(declared_element_type)
            != self.map_type(backing_element_type)
            and binding.get("pointer_reinterpretation") is None
        ):
            if is_workgroup_pointer:
                raise self.hlsl_workgroup_pointer_error(
                    "DirectX workgroup pointer alias "
                    f"'{node.name}' changes element type from "
                    f"{backing_element_type} to {declared_element_type}",
                    function_name=self.current_function_name,
                    parameter_name=node.name,
                    reason="element-type-mismatch",
                    node=node,
                )
            raise DirectXResourcePointerAliasError(
                "DirectX resource pointer alias "
                f"'{node.name}' changes element type from {backing_element_type} "
                f"to {declared_element_type}",
                function_name=self.current_function_name,
                pointer_name=node.name,
                reason="element-type-mismatch",
                source_location=getattr(node, "source_location", None),
            )

        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        required_access = (
            "read"
            if qualifiers.intersection({"const", "constant", "readonly", "in"})
            else "read_write"
        )
        if required_access == "read_write" and binding.get("access") != "read_write":
            if is_workgroup_pointer:
                raise self.hlsl_workgroup_pointer_error(
                    "DirectX workgroup pointer alias "
                    f"'{node.name}' requires writable shared storage",
                    function_name=self.current_function_name,
                    parameter_name=node.name,
                    reason="access-mismatch",
                    node=node,
                )
            raise DirectXResourcePointerAliasError(
                "DirectX resource pointer alias "
                f"'{node.name}' requires writable storage",
                function_name=self.current_function_name,
                pointer_name=node.name,
                reason="access-mismatch",
                source_location=getattr(node, "source_location", None),
            )

        used_names = set(self.current_identifier_reserved_names)
        used_names.update(self.local_variable_types)
        used_names.update(self.current_identifier_aliases.values())
        offset_name = self.hlsl_unique_local_identifier(
            f"{node.name}_offset", used_names
        )
        self.current_identifier_reserved_names.add(offset_name)
        rendered_offset = binding["offset"]
        binding.update(
            {
                "offset": offset_name,
                "element_type": declared_element_type or backing_element_type,
                "access": required_access,
                "is_pointer_alias": True,
            }
        )
        self.current_hlsl_resource_pointer_aliases[node.name] = binding
        offset_type = "int" if is_workgroup_pointer else "int64_t"
        self.local_variable_types[offset_name] = offset_type
        return (
            f"{'    ' * indent}{offset_type} {offset_name} = "
            f"{offset_type}({rendered_offset});\n"
        )

    def generate_hlsl_resource_pointer_array_base_assignment(
        self, target, value, op, *, statement_context=False
    ):
        if not isinstance(target, (str, IdentifierNode, VariableNode)):
            return None
        target_name = (
            target if isinstance(target, str) else getattr(target, "name", None)
        )
        binding = self.current_hlsl_resource_pointer_aliases.get(target_name)
        if binding is None or binding.get("kind") != "resource-pointer-array":
            return None
        if not binding.get("parameter_binding"):
            raise self.hlsl_resource_pointer_array_error(
                "DirectX cannot move the base of a concrete storage pointer array",
                binding=binding,
                node=target,
                reason="nonparameter-base-mutation-unsupported",
            )
        if op not in {"+=", "-="}:
            raise self.hlsl_resource_pointer_array_error(
                "DirectX cannot lower storage pointer-array parameter base "
                f"operator '{op}'",
                binding=binding,
                node=target,
                reason="base-assignment-operator-unsupported",
            )
        if not statement_context:
            raise self.hlsl_resource_pointer_array_error(
                "DirectX cannot materialize the value produced by a storage "
                "pointer-array parameter base assignment",
                binding=binding,
                node=target,
                reason="base-assignment-value-unsupported",
            )
        base_index = binding.get("base_index")
        if not base_index:
            raise self.hlsl_resource_pointer_array_error(
                "DirectX storage pointer-array parameter has no logical base state",
                binding=binding,
                node=target,
                reason="base-state-unavailable",
            )
        rendered = self.generate_expression(value)
        delta_type = self.hlsl_resource_pointer_delta_type(value)
        return f"{base_index} {op} {delta_type}({rendered})"

    def generate_hlsl_resource_pointer_step(
        self, expression, *, statement_context=False
    ):
        """``p++``, ``++p``, ``p--`` or ``--p`` of a storage or workgroup pointer.

        Such a pointer lowers to its backing object and an element offset, so
        a step whose value is discarded moves the offset by one element, as
        ``p += 1`` does. A step whose value is used needs the pointer as a
        first-class value and is refused.
        """
        if not isinstance(expression, UnaryOpNode):
            return None
        operator = self.map_operator(expression.op)
        if expression.op in {"PRE_INCREMENT", "POST_INCREMENT"}:
            operator = "++"
        elif expression.op in {"PRE_DECREMENT", "POST_DECREMENT"}:
            operator = "--"
        if operator not in {"++", "--"}:
            return None
        operand = expression.operand
        if not isinstance(operand, (str, IdentifierNode, VariableNode)):
            return None
        name = operand if isinstance(operand, str) else getattr(operand, "name", None)
        binding = self.current_hlsl_resource_pointer_aliases.get(name)
        if (
            binding is None
            or binding.get("kind")
            not in {None, "resource-pointer", "workgroup-pointer"}
            or not isinstance(binding.get("offset"), str)
        ):
            return None
        workgroup = binding.get("kind") == "workgroup-pointer"
        if not statement_context or binding.get("deferred_unbound"):
            message = (
                f"DirectX cannot step pointer '{self.current_function_name}.{name}' "
                + (
                    "before a concrete backing object is assigned"
                    if binding.get("deferred_unbound")
                    else "where the step's value is used"
                )
            )
            reason = (
                "deferred-use-before-assignment"
                if binding.get("deferred_unbound")
                else "step-value-unsupported"
            )
            if workgroup:
                raise self.hlsl_workgroup_pointer_error(
                    message,
                    function_name=self.current_function_name,
                    parameter_name=name,
                    reason=reason,
                    node=expression,
                )
            raise DirectXResourcePointerParameterError(
                message,
                function_name=self.current_function_name,
                parameter_name=name,
                address_space=binding.get("address_space"),
                expected_access=binding.get("access"),
                reason=reason,
                source_location=getattr(expression, "source_location", None),
            )
        return f"{binding['offset']} {'+=' if operator == '++' else '-='} 1"

    def generate_hlsl_resource_pointer_array_base_unary(
        self, expression, *, statement_context=False
    ):
        if not isinstance(expression, UnaryOpNode):
            return None
        operator = self.map_operator(expression.op)
        if operator not in {"++", "--"}:
            return None
        operand = expression.operand
        if not isinstance(operand, (str, IdentifierNode, VariableNode)):
            return None
        operand_name = (
            operand if isinstance(operand, str) else getattr(operand, "name", None)
        )
        binding = self.current_hlsl_resource_pointer_aliases.get(operand_name)
        if binding is None or binding.get("kind") != "resource-pointer-array":
            return None
        if not binding.get("parameter_binding"):
            raise self.hlsl_resource_pointer_array_error(
                "DirectX cannot move the base of a concrete storage pointer array",
                binding=binding,
                node=expression,
                reason="nonparameter-base-mutation-unsupported",
            )
        if not statement_context:
            raise self.hlsl_resource_pointer_array_error(
                "DirectX cannot materialize the value produced by storage "
                "pointer-array parameter increment or decrement",
                binding=binding,
                node=expression,
                reason="base-unary-value-unsupported",
            )
        base_index = binding.get("base_index")
        if not base_index:
            raise self.hlsl_resource_pointer_array_error(
                "DirectX storage pointer-array parameter has no logical base state",
                binding=binding,
                node=expression,
                reason="base-state-unavailable",
            )
        if getattr(expression, "is_postfix", False):
            return f"{base_index}{operator}"
        return f"{operator}{base_index}"

    def generate_hlsl_resource_pointer_alias_assignment(
        self, target, value, op, *, statement_context=False
    ):
        pointer_array_base_assignment = (
            self.generate_hlsl_resource_pointer_array_base_assignment(
                target,
                value,
                op,
                statement_context=statement_context,
            )
        )
        if pointer_array_base_assignment is not None:
            return pointer_array_base_assignment
        pointer_array_assignment = self.generate_hlsl_resource_pointer_array_assignment(
            target, value, op
        )
        if pointer_array_assignment is not None:
            return pointer_array_assignment
        if not isinstance(target, (str, IdentifierNode, VariableNode)):
            return None
        target_name = (
            target if isinstance(target, str) else getattr(target, "name", None)
        )
        current = self.current_hlsl_resource_pointer_aliases.get(target_name)
        if current is None:
            return None

        offset_name = current["offset"]
        offset_type = "int" if current.get("kind") == "workgroup-pointer" else "int64_t"
        if current.get("deferred_unbound") and op != "=":
            raise DirectXResourcePointerParameterError(
                "DirectX deferred storage pointer "
                f"'{self.current_function_name}.{target_name}' is mutated before "
                "a concrete backing resource is assigned",
                function_name=self.current_function_name,
                parameter_name=target_name,
                address_space=current.get("address_space"),
                expected_access=current.get("access"),
                reason="deferred-use-before-assignment",
                source_location=getattr(target, "source_location", None),
            )
        if op in {"+=", "-="}:
            delta = self.generate_expression(value)
            delta_type = (
                "int"
                if current.get("kind") == "workgroup-pointer"
                else self.hlsl_resource_pointer_delta_type(value)
            )
            return f"{offset_name} {op} {delta_type}({delta})"
        if op != "=":
            if current.get("kind") == "workgroup-pointer":
                raise self.hlsl_workgroup_pointer_error(
                    "DirectX cannot lower workgroup pointer operator "
                    f"'{op}' for '{target_name}'",
                    function_name=self.current_function_name,
                    parameter_name=target_name,
                    reason="assignment-operator-unsupported",
                    node=target,
                )
            raise DirectXResourcePointerAliasError(
                f"DirectX cannot lower resource pointer operator '{op}' for "
                f"'{target_name}'",
                function_name=self.current_function_name,
                pointer_name=target_name,
                reason="assignment-operator-unsupported",
                source_location=getattr(target, "source_location", None),
            )

        replacement = self.hlsl_materialized_resource_pointer_array_element(
            self.hlsl_resource_pointer_binding(value), value
        )
        if replacement is None:
            if current.get("kind") == "workgroup-pointer":
                raise self.hlsl_workgroup_pointer_error(
                    "DirectX workgroup pointer assignment has no concrete "
                    f"shared-array backing object: {target_name}",
                    function_name=self.current_function_name,
                    parameter_name=target_name,
                    reason="assignment-backing-unresolved",
                    node=value,
                )
            raise DirectXResourcePointerParameterError(
                "DirectX resource pointer assignment has no concrete buffer "
                f"backing object: {target_name}",
                function_name=self.current_function_name,
                parameter_name=target_name,
                address_space=current.get("address_space"),
                expected_access=current.get("access"),
                reason="assignment-backing-unresolved",
                source_location=getattr(value, "source_location", None),
            )
        if current.get("deferred_unbound"):
            actual_type = (
                replacement.get("view_element_type")
                or replacement.get("source_element_type")
                or replacement.get("element_type")
            )
            if self.map_type(actual_type) != self.map_type(current.get("element_type")):
                raise DirectXResourcePointerParameterError(
                    "DirectX deferred storage pointer assignment changes element "
                    f"type for '{target_name}'",
                    function_name=self.current_function_name,
                    parameter_name=target_name,
                    address_space=current.get("address_space"),
                    expected_access=current.get("access"),
                    actual_access=replacement.get("access"),
                    reason="element-type-mismatch",
                    source_location=getattr(value, "source_location", None),
                )
            if not self.hlsl_resource_pointer_access_satisfies(
                replacement.get("access"), current.get("access")
            ):
                raise DirectXResourcePointerParameterError(
                    "DirectX deferred storage pointer assignment cannot satisfy "
                    f"the access contract for '{target_name}'",
                    function_name=self.current_function_name,
                    parameter_name=target_name,
                    address_space=current.get("address_space"),
                    expected_access=current.get("access"),
                    actual_access=replacement.get("access"),
                    reason="access-mismatch",
                    source_location=getattr(value, "source_location", None),
                )
            self.current_hlsl_resource_pointer_aliases[target_name] = {
                **replacement,
                "offset": offset_name,
                "element_type": current.get("element_type"),
                "access": current.get("access"),
                "is_pointer_alias": True,
                "deferred_unbound": False,
            }
            return f"{offset_name} = int64_t({replacement['offset']})"
        if replacement.get("root") != current.get("root"):
            if current.get("kind") == "workgroup-pointer":
                raise self.hlsl_workgroup_pointer_error(
                    "DirectX cannot dynamically change the backing array for "
                    f"workgroup pointer '{target_name}'",
                    function_name=self.current_function_name,
                    parameter_name=target_name,
                    reason="dynamic-backing-reassignment",
                    node=value,
                )
            raise DirectXResourcePointerAliasError(
                "DirectX cannot dynamically change the backing buffer for resource "
                f"pointer '{target_name}'",
                function_name=self.current_function_name,
                pointer_name=target_name,
                reason="dynamic-backing-reassignment",
                source_location=getattr(value, "source_location", None),
            )
        if replacement.get("element_type") != current.get("element_type"):
            if current.get("kind") == "workgroup-pointer":
                raise self.hlsl_workgroup_pointer_error(
                    "DirectX workgroup pointer assignment changes element type "
                    f"for '{target_name}'",
                    function_name=self.current_function_name,
                    parameter_name=target_name,
                    reason="element-type-mismatch",
                    node=value,
                )
            raise DirectXResourcePointerAliasError(
                "DirectX resource pointer assignment changes element type for "
                f"'{target_name}'",
                function_name=self.current_function_name,
                pointer_name=target_name,
                reason="element-type-mismatch",
                source_location=getattr(value, "source_location", None),
            )
        if (
            current.get("access") == "read_write"
            and replacement.get("access") != "read_write"
        ):
            raise DirectXResourcePointerAliasError(
                f"DirectX resource pointer '{target_name}' requires writable storage",
                function_name=self.current_function_name,
                pointer_name=target_name,
                reason="access-mismatch",
                source_location=getattr(value, "source_location", None),
            )
        self.current_hlsl_resource_pointer_aliases[target_name] = {
            **replacement,
            "offset": offset_name,
            "access": current.get("access"),
            "is_pointer_alias": current.get("is_pointer_alias", False),
        }
        return f"{offset_name} = {offset_type}({replacement['offset']})"

    def hlsl_resource_pointer_postincrement_load(self, expression):
        if not (
            isinstance(expression, UnaryOpNode)
            and self.map_operator(expression.op) == "*"
            and not getattr(expression, "is_postfix", False)
            and isinstance(expression.operand, UnaryOpNode)
            and self.map_operator(expression.operand.op) == "++"
            and getattr(expression.operand, "is_postfix", False)
        ):
            return None
        pointer_expression = expression.operand.operand
        binding = self.hlsl_resource_pointer_binding(pointer_expression)
        if binding is None:
            return None
        if binding.get("kind") == "workgroup-pointer":
            raise self.hlsl_workgroup_pointer_error(
                "DirectX cannot lower a post-increment dereference of a "
                "workgroup pointer",
                function_name=self.current_function_name,
                parameter_name=self.expression_name(pointer_expression),
                reason="postincrement-dereference-unsupported",
                node=expression,
            )
        if not self.hlsl_resource_pointer_access_satisfies(
            binding.get("access"), "read"
        ):
            raise DirectXResourcePointerParameterError(
                "DirectX cannot read through a post-incremented write-only "
                "storage pointer",
                function_name=self.current_function_name,
                parameter_name=self.expression_name(pointer_expression),
                address_space=binding.get("address_space", "storage"),
                expected_access="read",
                actual_access=binding.get("access"),
                reason="read-through-writeonly-pointer",
                source_location=getattr(expression, "source_location", None),
            )
        offset = str(binding.get("offset", ""))
        if not self.hlsl_resource_pointer_offset_lvalue(offset):
            raise DirectXResourcePointerParameterError(
                "DirectX post-increment storage-pointer dereference requires an "
                "assignable logical offset",
                function_name=self.current_function_name,
                parameter_name=self.expression_name(pointer_expression),
                address_space=binding.get("address_space", "storage"),
                expected_access=binding.get("access"),
                reason="mutable-offset-lvalue-unresolved",
                source_location=getattr(expression, "source_location", None),
            )
        resource_type = binding.get("resource_type")
        if self.hlsl_resource_type_name(resource_type) not in {
            "Buffer",
            "StructuredBuffer",
            "RWBuffer",
            "RWStructuredBuffer",
            "RasterizerOrderedBuffer",
            "RasterizerOrderedStructuredBuffer",
        }:
            raise DirectXResourcePointerParameterError(
                "DirectX post-increment storage-pointer dereference requires an "
                "indexable typed buffer",
                function_name=self.current_function_name,
                parameter_name=self.expression_name(pointer_expression),
                address_space=binding.get("address_space", "storage"),
                reason="resource-kind-mismatch",
                source_location=getattr(expression, "source_location", None),
            )
        logical_type = (
            binding.get("view_element_type")
            or binding.get("element_type")
            or binding.get("source_element_type")
        )
        reinterpretation = binding.get("pointer_reinterpretation")
        key = (
            str(resource_type),
            self.map_type(logical_type),
            (
                (
                    reinterpretation["source_layout"].name,
                    reinterpretation["target_layout"].name,
                    reinterpretation.get("target_width", 1),
                )
                if reinterpretation is not None
                else None
            ),
            str(binding.get("byte_offset", "0")),
        )
        helper = self.hlsl_resource_pointer_postincrement_helpers.get(key)
        if helper is None:
            digest = sha1(repr(key).encode("utf-8")).hexdigest()[:12]
            helper_name = f"__crossgl_resource_pointer_postincrement_{digest}"
            reserved = set(self.global_variable_types)
            reserved.update(self.structs_by_name)
            reserved.update(self.current_hlsl_available_functions)
            reserved.update(
                item["name"]
                for item in self.hlsl_resource_pointer_postincrement_helpers.values()
            )
            while helper_name in reserved:
                helper_name += "_"
            helper = {
                "name": helper_name,
                "resource_type": str(resource_type),
                "return_type": self.map_type(logical_type),
                "binding": dict(binding),
            }
            self.hlsl_resource_pointer_postincrement_helpers[key] = helper
        return f"{helper['name']}({binding['root']}, {offset})"

    def generate_hlsl_resource_pointer_postincrement_helpers(self):
        code = ""
        for helper in self.hlsl_resource_pointer_postincrement_helpers.values():
            binding = {
                **helper["binding"],
                "root": "source",
                "offset": "offset",
            }
            load = self.hlsl_pointer_reinterpret_read_expression(binding, "offset")
            if load is None:
                load = "source[uint(offset)]"
                logical = self.hlsl_half_storage.binding_type(binding)
                if logical is not None:
                    load = self.hlsl_half_storage.convert(logical, load)
            code += (
                f"{helper['return_type']} {helper['name']}("
                f"{helper['resource_type']} source, inout int64_t offset) {{\n"
                f"    {helper['return_type']} value = {load};\n"
                "    offset += int64_t(1);\n"
                "    return value;\n"
                "}\n\n"
            )
        return code

    def generate_hlsl_resource_pointer_access(
        self,
        pointer_expression,
        index_expression=0,
        *,
        require_write=False,
        physical=False,
        word_view=False,
    ):
        binding = self.hlsl_resource_pointer_binding(pointer_expression)
        if binding is None:
            return None
        if binding.get("root_kind") == "null":
            self.hlsl_workgroup_pointer_null_observation_error(
                function_name=self.current_function_name,
                parameter_name=self.expression_name(pointer_expression),
                node=pointer_expression,
            )
        binding_access = binding.get("access")
        if require_write and not self.hlsl_resource_pointer_access_satisfies(
            binding_access, "write"
        ):
            pointer_array_binding = binding.get("pointer_array_binding")
            if pointer_array_binding is not None:
                raise self.hlsl_resource_pointer_array_error(
                    "DirectX storage pointer array "
                    f"'{pointer_array_binding['array_name']}' is read-only",
                    binding=pointer_array_binding,
                    node=pointer_expression,
                    reason="write-through-readonly-alias",
                )
            name = getattr(pointer_expression, "name", None) or str(pointer_expression)
            raise DirectXResourcePointerAliasError(
                f"DirectX resource pointer '{name}' does not provide writable storage",
                function_name=self.current_function_name,
                pointer_name=name,
                reason="write-through-readonly-pointer",
                source_location=getattr(pointer_expression, "source_location", None),
            )
        if not require_write and binding_access == "write":
            name = getattr(pointer_expression, "name", None) or str(pointer_expression)
            raise DirectXResourcePointerParameterError(
                f"DirectX resource pointer '{name}' does not provide readable storage",
                function_name=self.current_function_name,
                parameter_name=getattr(pointer_expression, "name", None),
                address_space=binding.get("address_space", "storage"),
                expected_access="read",
                actual_access=binding_access,
                reason="read-through-writeonly-pointer",
                source_location=getattr(pointer_expression, "source_location", None),
            )
        reinterpretation = binding.get("pointer_reinterpretation")
        if word_view and is_same_width_word_view(reinterpretation):
            # The caller stores whole backing words through the bit-cast view.
            return hlsl_word_view_element(self, binding, index_expression)["lvalue"]
        if require_write and reinterpretation is not None:
            error = PointerReinterpretationError(
                "DirectX cannot preserve writes through a reinterpreted typed "
                "storage-buffer view",
                source_type=reinterpretation["source_layout"].name,
                target_type=reinterpretation["target_layout"].name,
                address_space="storage",
                alignment=reinterpretation["target_layout"].byte_width,
                access="write",
                reason="write-requires-byte-address-storage",
                source_location=getattr(pointer_expression, "source_location", None),
            )
            # A helper parameter's packed view is only its default layout; the
            # narrow-layout passes may still prove every caller passes widened
            # storage, which makes this write an element store.
            error.hlsl_narrow_storage_origin = binding.get("narrow_storage_origin")
            raise error

        resource_name = self.hlsl_resource_type_name(binding.get("resource_type"))
        if binding.get("kind") != "workgroup-pointer" and resource_name not in {
            "Buffer",
            "StructuredBuffer",
            "RWBuffer",
            "RWStructuredBuffer",
            "RasterizerOrderedBuffer",
            "RasterizerOrderedStructuredBuffer",
        }:
            raise DirectXResourcePointerAliasError(
                "DirectX resource pointer dereference requires an indexable typed "
                f"buffer, got {binding.get('resource_type') or 'unknown'}",
                function_name=self.current_function_name,
                pointer_name=self.expression_name(pointer_expression),
                reason="non-indexable-backing",
                source_location=getattr(pointer_expression, "source_location", None),
            )

        rendered_index = (
            str(index_expression)
            if isinstance(index_expression, (int, float))
            else self.generate_expression(index_expression)
        )
        effective_index = self.hlsl_resource_pointer_offset_sum(
            binding.get("offset"), rendered_index
        )
        if (
            reinterpretation is not None
            and reinterpretation["source_layout"].bit_width == 8
            and reinterpretation["target_layout"].bit_width > 8
            and (
                self.hlsl_private_pointer_expression_has_side_effects(
                    pointer_expression
                )
                or self.hlsl_private_pointer_expression_has_side_effects(
                    index_expression
                )
            )
        ):
            raise PointerReinterpretationError(
                "DirectX byte assembly requires a stable source offset",
                source_type=reinterpretation["source_layout"].name,
                target_type=reinterpretation["target_layout"].name,
                address_space="storage",
                access=binding_access,
                reason="side-effecting-byte-assembly-offset",
                source_location=getattr(pointer_expression, "source_location", None),
            )
        reinterpret_read = self.hlsl_pointer_reinterpret_read_expression(
            binding, effective_index
        )
        if reinterpret_read is not None:
            return reinterpret_read
        rendered = f"{binding['root']}[uint({effective_index})]"
        logical = self.hlsl_half_storage.binding_type(binding)
        if logical is not None and not require_write and not physical:
            return self.hlsl_half_storage.convert(logical, rendered)
        return rendered

    def hlsl_pointer_reinterpret_read_expression(self, binding, element_index):
        reinterpretation = binding.get("pointer_reinterpretation")
        if reinterpretation is None:
            return None

        source_layout = reinterpretation["source_layout"]
        target_layout = reinterpretation["target_layout"]
        target_width = reinterpretation.get("target_width", 1)
        source_member = reinterpretation.get("source_member")
        member_suffix = f".{source_member}" if source_member else ""
        byte_offset = str(binding.get("byte_offset", "0"))
        element_bytes = str(element_index)
        target_byte_width = target_layout.byte_width * target_width
        if target_byte_width != 1:
            element_bytes = f"({element_bytes} * {target_byte_width})"
        if byte_offset in {"0", "0u"}:
            byte_index = element_bytes
        elif element_bytes in {"0", "0u"}:
            byte_index = byte_offset
        else:
            byte_index = f"({byte_offset} + {element_bytes})"

        if source_layout.bit_width == 8:
            components = []
            for lane in range(target_layout.byte_width):
                index = byte_index if lane == 0 else f"({byte_index} + {lane})"
                value = f"(uint({binding['root']}[uint({index})]) & 255u)"
                components.append(value if lane == 0 else f"({value} << {lane * 8}u)")
            bits = "(" + " | ".join(components) + ")"
            if target_layout.kind == "floating":
                return f"asfloat({bits})"
            if target_layout.signed:
                if target_layout.bit_width == 32:
                    return f"asint({bits})"
                shift = 32 - target_layout.bit_width
                return f"(int({bits} << {shift}u) >> {shift})"
            return bits

        if target_width > 1:
            lane_gather = bool(reinterpretation.get("lane_gather"))
            if (
                lane_gather
                and source_layout.name == "bfloat16"
                and not getattr(self, "hlsl_bfloat16_vector_conversion_depth", 0)
            ):
                raise PointerReinterpretationError(
                    "DirectX bfloat16 vector storage views must be converted "
                    "to a float vector immediately",
                    source_type=source_layout.name,
                    target_type=binding.get("view_element_type"),
                    address_space="storage",
                    alignment=source_layout.byte_width,
                    access="read",
                    target_backend="directx",
                    reason="bfloat16-vector-view-requires-float-conversion",
                )
            logical = (
                self.hlsl_half_storage.binding_type(binding) if lane_gather else None
            )
            components = []
            for lane in range(target_width):
                lane_byte_index = byte_index
                if lane:
                    lane_byte_index = (
                        f"({byte_index} + {lane * target_layout.byte_width})"
                    )
                source_index = f"uint(({lane_byte_index}) / {source_layout.byte_width})"
                source_value = f"{binding['root']}[{source_index}]{member_suffix}"
                if lane_gather:
                    # One structured-buffer element holds each logical lane.
                    components.append(
                        self.hlsl_half_storage.convert(logical, source_value)
                        if logical is not None
                        else source_value
                    )
                    continue
                source_bits = (
                    f"asuint({source_value})"
                    if source_layout.kind == "floating" or source_layout.signed
                    else source_value
                )
                if target_layout.kind == "floating":
                    components.append(f"asfloat({source_bits})")
                elif target_layout.signed:
                    components.append(f"asint({source_bits})")
                else:
                    components.append(source_bits)
            constructor = self.map_type(
                binding.get("view_element_type") or binding.get("element_type")
            )
            return f"{constructor}({', '.join(components)})"

        source_index = f"uint(({byte_index}) / {source_layout.byte_width})"
        private_word_view = reinterpretation.get("private_word_view")
        if (
            private_word_view is not None
            and private_word_view.get("backing_kind") == "aggregate"
        ):
            source_bits = (
                f"{private_word_view['helper_name']}"
                f"({binding['root']}, {source_index})"
            )
        else:
            source_value = f"{binding['root']}[{source_index}]{member_suffix}"
            if source_layout.kind == "floating" or source_layout.signed:
                source_bits = f"asuint({source_value})"
            else:
                source_bits = source_value

        if target_layout.bit_width == 32:
            if target_layout.kind == "floating":
                return f"asfloat({source_bits})"
            if target_layout.signed:
                return f"asint({source_bits})"
            return source_bits

        bit_offset = f"uint((({byte_index}) % {source_layout.byte_width}) * 8)"
        if target_layout.signed:
            left_shift = f"(32u - {bit_offset} - {target_layout.bit_width}u)"
            return (
                f"(int({source_bits} << {left_shift}) >> "
                f"{32 - target_layout.bit_width})"
            )
        mask = (1 << target_layout.bit_width) - 1
        return f"(({source_bits} >> {bit_offset}) & {mask}u)"

    def generate_hlsl_resource_pointer_dereference_assignment(self, target, value, op):
        if not (
            isinstance(target, UnaryOpNode)
            and not getattr(target, "is_postfix", False)
            and target.op == "*"
        ):
            return None
        lhs = self.generate_hlsl_resource_pointer_access(
            target.operand, require_write=True
        )
        if lhs is None:
            return None
        binding = self.hlsl_resource_pointer_binding(target.operand)
        compound_assignment = self.generate_hlsl_bfloat16_compound_assignment(
            target,
            target,
            value,
            op,
            lhs=lhs,
            target_type=binding.get("element_type") if binding else None,
        )
        if compound_assignment is not None:
            return compound_assignment
        compound_assignment = self.generate_hlsl_native_16_bit_compound_assignment(
            target,
            target,
            value,
            op,
            lhs=lhs,
            target_type=binding.get("element_type") if binding else None,
        )
        if compound_assignment is not None:
            return compound_assignment
        compound_assignment = (
            self.generate_hlsl_minimum_precision_integer_compound_assignment(
                target,
                target,
                value,
                op,
                lhs=lhs,
                target_type=binding.get("element_type") if binding else None,
            )
        )
        if compound_assignment is not None:
            return compound_assignment
        target_type = self.hlsl_resource_pointer_assignment_type(binding)
        compound_assignment = self.generate_hlsl_wide_integer_compound_assignment(
            target, target, value, op, lhs=lhs, target_type=target_type
        )
        if compound_assignment is not None:
            return compound_assignment
        rhs = self.generate_expression_with_expected(
            value,
            target_type,
            source_location=getattr(target, "source_location", None),
        )
        rhs = self.hlsl_bfloat16_storage_assignment_expression(
            target,
            rhs,
            target_type,
            source_type=self.hlsl_source_expression_type(value),
        )
        rhs = self.hlsl_precise_stored_value(target, value, rhs, target_type, op)
        return f"{lhs} {op} {rhs}"

    def generate_hlsl_bfloat16_compound_assignment(
        self, node, target, value, operator, *, target_type=None, lhs=None
    ):
        target_type = target_type or self.hlsl_source_expression_type(target)
        mapped_operator = self.map_operator(operator)
        vector_assignment = self.hlsl_bfloat16_vector_compound_assignment(
            node, target, value, mapped_operator, target_type=target_type, lhs=lhs
        )
        if vector_assignment is not None:
            return vector_assignment
        if not self.is_hlsl_bfloat16_type(target_type):
            if mapped_operator not in {"+=", "-=", "*=", "/="}:
                return None
            value_type = self.hlsl_source_expression_type(value)
            integer_info = self.hlsl_integer_arithmetic_type_info(target_type)
            if (
                self.is_hlsl_bfloat16_type(value_type)
                and integer_info is not None
                and integer_info["width"] == 1
            ):
                reason = None
                if integer_info["base_type"] not in {"int", "uint"}:
                    reason = "unsupported-bfloat16-compound-integer-width"
                elif not self.hlsl_expression_is_repeatable(
                    target
                ) or self.hlsl_expression_has_observable_side_effects(target):
                    reason = "side-effecting-assignment-target"
                elif self.hlsl_expression_has_observable_side_effects(
                    value,
                    allow_integer_constructors=True,
                    allow_bfloat_constructors=True,
                ):
                    reason = "rhs-may-change-assignment-target"
                if reason is not None:
                    raise self.directx_bfloat16_unsupported(
                        "DirectX bfloat compound assignment requires a stable "
                        "32-bit integer target and a right operand without "
                        "observable side effects; materialize side effects "
                        "before the assignment",
                        operation=mapped_operator,
                        source_type=target_type,
                        reason=reason,
                        source_location=getattr(node, "source_location", None),
                    )
                operation = BinaryOpNode(
                    FunctionCallNode("bfloat", [target]),
                    mapped_operator[0],
                    value,
                    source_location=getattr(node, "source_location", None),
                )
                rhs = self.generate_expression_with_expected(operation, target_type)
                return f"{lhs or self.generate_expression(target)} = {rhs}"
            return None
        if mapped_operator not in {"+=", "-=", "*=", "/="}:
            if mapped_operator == "=":
                return None
            raise self.directx_bfloat16_unsupported(
                "DirectX exact bfloat16 compound assignment does not support "
                f"'{mapped_operator}'",
                operation=mapped_operator,
                source_type=target_type,
                reason="unsupported-bfloat16-compound-assignment",
                source_location=getattr(node, "source_location", None),
            )
        if lhs is None and not self.hlsl_expression_is_repeatable(target):
            raise self.directx_bfloat16_unsupported(
                "DirectX exact bfloat16 compound assignment requires a stable lvalue",
                operation=mapped_operator,
                source_type=target_type,
                reason="side-effecting-assignment-target",
                source_location=getattr(node, "source_location", None),
            )

        lhs = lhs or self.generate_expression(target)
        rhs = self.generate_expression_with_expected(value, None)
        rhs_type = self.hlsl_source_expression_type(value)
        if self.is_hlsl_bfloat16_type(rhs_type):
            rhs = self.hlsl_bfloat16_to_float_expression(rhs)
        elif not self.is_scalar_value_type(self.map_type(rhs_type)):
            raise self.directx_bfloat16_unsupported(
                "DirectX exact bfloat16 compound assignment requires a scalar "
                f"right operand, got '{self.type_name_string(rhs_type)}'",
                operation=mapped_operator,
                source_type=rhs_type,
                reason="unsupported-compound-assignment-source",
                source_location=getattr(node, "source_location", None),
            )
        left = self.hlsl_bfloat16_to_float_expression(lhs)
        arithmetic_operator = mapped_operator[0]
        rounded = self.hlsl_float_to_bfloat16_expression(
            f"({left} {arithmetic_operator} {rhs})"
        )
        rounded = self.hlsl_bfloat16_storage_assignment_expression(
            target, rounded, target_type, source_type=target_type
        )
        return f"{lhs} = {rounded}"

    def directx_cooperative_matrix_element_type(self, expr):
        """Source element type of one lane-local fragment element."""

        operands = getattr(expr, "arguments", None) or ()
        if operands:
            matrix_type = _directx_cooperative_matrix_contract_type(
                self.directx_cooperative_matrix_expression_type(operands[0])
            )
            if isinstance(matrix_type, CooperativeMatrixType):
                return matrix_type.element_type
        return getattr(expr, "expression_type", None)

    def hlsl_assignment_target_type(self, target):
        if (
            isinstance(target, CooperativeMatrixOpNode)
            and target.operation == "element"
        ):
            # A lane-local fragment element holds the fragment element type, so
            # a bfloat16 source is decoded instead of storing its raw bits.
            return self.directx_cooperative_matrix_element_type(target)
        inferred_type = self.expression_result_type(target)
        if not isinstance(target, ArrayAccessNode):
            return inferred_type
        if self.is_hlsl_bfloat16_type(inferred_type) or (
            self.hlsl_byte_integer_shape(inferred_type) is not None
        ):
            return inferred_type

        resource_types = {
            "RWBuffer",
            "RWStructuredBuffer",
            "RasterizerOrderedBuffer",
            "RasterizerOrderedStructuredBuffer",
        }
        array_expression = target.array
        resource_type = self.hlsl_struct_buffer_expression_resource_type(target)
        if resource_type is None:
            binding = self.hlsl_resource_pointer_binding(array_expression)
            if binding is not None:
                resource_type = binding.get("resource_type")
        if resource_type is None:
            resource_type = self.expression_result_type(array_expression)
        if resource_type is None:
            return inferred_type
        physical_resource_type = self.map_type(
            self.directx_resource_declaration_type(resource_type)
        )
        element_type = self.hlsl_typed_buffer_element_type(
            physical_resource_type, resource_types
        )
        return element_type or inferred_type

    def hlsl_resource_pointer_assignment_type(self, binding):
        if binding is None:
            return None
        source_type = binding.get("source_element_type")
        if self.hlsl_byte_integer_shape(source_type) is not None:
            return source_type
        return binding.get("element_type")
