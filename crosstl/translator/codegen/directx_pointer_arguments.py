"""Resource array pointer arguments in the HLSL generator: indices, bindings and offsets, workgroup ranges, nested entry storage access and rebasing."""

import re

from ..ast import (
    ArrayAccessNode,
    ArrayType,
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    DoWhileNode,
    ExpressionStatementNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    LoopNode,
    MatchNode,
    PointerType,
    SwitchNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)
from .directx_values import (
    DirectXResourcePointerArrayError,
    DirectXResourcePointerParameterError,
)
from .pointer_reinterpret import scalar_storage_layout


class DirectXPointerArgumentMixin:
    """Pointer argument members of ``HLSLCodeGen``."""

    def hlsl_nested_entry_resource_rebase_names(self, function):
        """Return entry pointer parameters whose rebases need an entry offset.

        The first rebase of an entry pointer in a scope declares its offset
        where it occurs, which is exact when that scope runs once and no later
        statement outside it refers to the pointer. A first rebase inside a
        loop, or one whose pointer is referenced after the enclosing statement,
        must instead update one offset declared at entry. A local declaration
        that shadows the parameter hides it for the rest of its scope, so
        assignments to that local are not pointer rebases.
        """
        parameter_names = {
            getattr(parameter, "name", None)
            for parameter in getattr(function, "parameters", None) or []
        }
        parameter_names.discard(None)
        hoisted = set()
        if not parameter_names:
            return hoisted

        def statements_of(body):
            if body is None:
                return []
            if isinstance(body, list):
                return body
            if isinstance(body, BlockNode):
                return list(getattr(body, "statements", None) or [])
            return [body]

        def rebased_name(node, shadowed):
            if not isinstance(node, AssignmentNode):
                return None
            if getattr(node, "operator", "=") not in {"=", "+=", "-="}:
                return None
            target = getattr(node, "target", None)
            if (
                isinstance(target, (IdentifierNode, VariableNode))
                and target.name in parameter_names
                and target.name not in shadowed
            ):
                return target.name
            return None

        def shadowing_declaration(statement):
            if isinstance(statement, VariableNode) and statement.name in (
                parameter_names
            ):
                return statement.name
            return None

        def references(statements, name):
            return any(
                isinstance(node, (IdentifierNode, VariableNode)) and node.name == name
                for node in self.walk_ast(statements)
            )

        def children(statement):
            # Returns nested statement bodies, the expressions evaluated by the
            # statement itself, whether it repeats, and the parameter names its
            # own declarations shadow inside it. Switch and match arms may fall
            # through or share scope, so they are handled like loops and share
            # every arm's declarations.
            if isinstance(statement, IfNode):
                bodies = [
                    getattr(statement, "then_branch", None),
                    *(getattr(statement, "else_if_bodies", None) or []),
                    getattr(statement, "else_branch", None),
                ]
                expressions = [
                    getattr(statement, "condition", None),
                    *(getattr(statement, "else_if_conditions", None) or []),
                ]
                return bodies, expressions, False, set()
            if isinstance(statement, ForNode):
                local = shadowing_declaration(statement.init)
                return (
                    [statement.body],
                    [statement.init, statement.condition, statement.update],
                    True,
                    {local} if local else set(),
                )
            if isinstance(statement, ForInNode):
                return [statement.body], [statement.iterable], True, set()
            if isinstance(statement, (WhileNode, DoWhileNode)):
                return [statement.body], [statement.condition], True, set()
            if isinstance(statement, LoopNode):
                return [statement.body], [], True, set()
            if isinstance(statement, SwitchNode):
                bodies = [
                    getattr(case, "statements", None) or getattr(case, "body", None)
                    for case in getattr(statement, "cases", None) or []
                ]
                default = getattr(statement, "default_case", None)
                if default is not None:
                    bodies.append(
                        getattr(default, "statements", default)
                        if not isinstance(default, list)
                        else default
                    )
                shared = {
                    local
                    for body in bodies
                    for local in map(shadowing_declaration, statements_of(body))
                    if local
                }
                return bodies, [statement.expression], True, shared
            if isinstance(statement, MatchNode):
                bodies = [
                    getattr(arm, "body", None)
                    for arm in getattr(statement, "arms", None) or []
                ]
                return bodies, [statement.expression], True, set()
            if isinstance(statement, BlockNode):
                return [statement], [], False, set()
            return [], [statement], False, set()

        def visit(statements, frames, in_loop, declared, shadowed):
            declared = set(declared)
            shadowed = set(shadowed)
            for index, statement in enumerate(statements):
                local = shadowing_declaration(statement)
                if local is not None:
                    shadowed.add(local)
                    continue
                expression = (
                    statement.expression
                    if isinstance(statement, ExpressionStatementNode)
                    else statement
                )
                name = rebased_name(expression, shadowed)
                if name is not None:
                    if name not in declared:
                        declared.add(name)
                        if frames and (
                            in_loop
                            or any(
                                references(block[position + 1 :], name)
                                for block, position in frames
                            )
                        ):
                            hoisted.add(name)
                    continue
                bodies, expressions, is_loop, scope_locals = children(statement)
                inner_shadowed = shadowed | scope_locals
                for node in self.walk_ast(expressions):
                    embedded = rebased_name(node, inner_shadowed)
                    if embedded is not None:
                        hoisted.add(embedded)
                for body in bodies:
                    visit(
                        statements_of(body),
                        frames + [(statements, index)],
                        in_loop or is_loop,
                        declared,
                        inner_shadowed,
                    )

        body = getattr(function, "body", None)
        visit(statements_of(body), [], False, set(), set())
        return hoisted

    def hlsl_mutable_resource_pointer_argument_names(self, function):
        names = set()
        for node in self.walk_ast(getattr(function, "body", [])):
            if not isinstance(node, FunctionCallNode):
                continue
            function_name = self.function_call_name(node)
            arguments = list(
                getattr(node, "arguments", getattr(node, "args", [])) or []
            )
            for index, argument in enumerate(arguments):
                parameter, _materialized_name = (
                    self.hlsl_resource_pointer_parameter_for_call(
                        function_name,
                        index,
                    )
                )
                if (
                    parameter is None
                    or self.hlsl_resource_pointer_offset_parameter_direction(parameter)
                    not in {"out", "inout"}
                ):
                    continue
                if isinstance(argument, (IdentifierNode, VariableNode)):
                    names.add(argument.name)
        return names

    def hlsl_resource_pointer_parameter_indices_for_function(self, function_name):
        indices = self.function_hlsl_resource_pointer_parameter_indices.get(
            function_name
        )
        if indices is not None:
            return indices
        materialized_name = self.hlsl_materialized_function_name(function_name)
        return self.function_hlsl_resource_pointer_parameter_indices.get(
            materialized_name, {}
        )

    def hlsl_resource_pointer_parameter_for_call(self, function_name, argument_index):
        parameter_name = self.hlsl_resource_pointer_parameter_indices_for_function(
            function_name
        ).get(argument_index)
        if parameter_name is None:
            return None, function_name
        materialized_name = self.hlsl_materialized_function_name(function_name)
        parameters = self.function_hlsl_resource_pointer_parameters.get(
            function_name
        ) or self.function_hlsl_resource_pointer_parameters.get(materialized_name, [])
        parameter = next(
            (candidate for candidate in parameters if candidate.name == parameter_name),
            None,
        )
        return parameter, materialized_name

    def hlsl_resource_pointer_array_parameter_indices_for_function(self, function_name):
        indices = self.function_hlsl_resource_pointer_array_parameter_indices.get(
            function_name
        )
        if indices is not None:
            return indices
        materialized_name = self.hlsl_materialized_function_name(function_name)
        return self.function_hlsl_resource_pointer_array_parameter_indices.get(
            materialized_name, {}
        )

    def hlsl_resource_pointer_array_parameter_for_call(
        self, function_name, argument_index
    ):
        parameter_name = (
            self.hlsl_resource_pointer_array_parameter_indices_for_function(
                function_name
            ).get(argument_index)
        )
        if parameter_name is None:
            return None, function_name, None, False
        materialized_name = self.hlsl_materialized_function_name(function_name)
        parameters = self.function_hlsl_resource_pointer_array_parameters.get(
            function_name
        ) or self.function_hlsl_resource_pointer_array_parameters.get(
            materialized_name, []
        )
        parameter = next(
            (candidate for candidate in parameters if candidate.name == parameter_name),
            None,
        )
        contract = (
            self.hlsl_resource_pointer_array_parameter_contract(
                parameter, materialized_name
            )
            if parameter is not None
            else None
        )
        mutated = parameter_name in (
            self.function_hlsl_resource_pointer_array_mutated_parameters.get(
                function_name
            )
            or self.function_hlsl_resource_pointer_array_mutated_parameters.get(
                materialized_name, set()
            )
        )
        return parameter, materialized_name, contract, mutated

    def hlsl_resource_pointer_array_argument_binding(self, expression, contract):
        binding = None
        if isinstance(expression, (str, IdentifierNode, VariableNode)):
            name = self.expression_name(expression)
            candidate = self.current_hlsl_resource_pointer_aliases.get(name)
            if (
                candidate is not None
                and candidate.get("kind") == "resource-pointer-array"
            ):
                binding = dict(candidate)
        elif (
            isinstance(expression, UnaryOpNode)
            and self.map_operator(expression.op) == "&"
        ):
            scalar_binding = self.hlsl_resource_pointer_binding(expression.operand)
            if scalar_binding is not None:
                if contract.get("extent") != 1:
                    raise DirectXResourcePointerArrayError(
                        "DirectX cannot bind one storage pointer to a fixed "
                        f"pointer-array parameter of extent {contract.get('extent')}",
                        function_name=self.current_function_name,
                        address_space=contract.get("address_space"),
                        reason="pointer-array-extent-mismatch",
                        source_location=getattr(expression, "source_location", None),
                    )
                binding = {
                    **scalar_binding,
                    "kind": "resource-pointer-array",
                    "extent": 1,
                    "offset_array": None,
                    "offset_expressions": [scalar_binding.get("offset", "0")],
                    # The selected scalar offset already includes any source
                    # pointer-array base.  A new extent-one array view starts at
                    # that selected pointer object, so its own base is zero.
                    "base_index": None,
                }
        if binding is None:
            return None
        extent = binding.get("extent")
        if extent != contract.get("extent"):
            raise DirectXResourcePointerArrayError(
                "DirectX fixed storage pointer-array argument extent "
                f"{extent} does not match parameter extent {contract.get('extent')}",
                function_name=self.current_function_name,
                address_space=contract.get("address_space"),
                reason="pointer-array-extent-mismatch",
                source_location=getattr(expression, "source_location", None),
            )
        offset_array = binding.get("offset_array")
        if offset_array:
            offsets = [f"{offset_array}[{index}]" for index in range(extent)]
        else:
            offsets = list(binding.get("offset_expressions") or [])
        if len(offsets) != extent:
            raise DirectXResourcePointerArrayError(
                "DirectX fixed storage pointer-array argument has incomplete "
                "logical-offset storage",
                function_name=self.current_function_name,
                address_space=contract.get("address_space"),
                reason="pointer-array-offsets-unresolved",
                source_location=getattr(expression, "source_location", None),
            )
        binding["offset_expressions"] = offsets
        return binding

    def hlsl_resource_pointer_array_call_argument_binding(
        self, function_name, argument_index, argument
    ):
        parameter, materialized_name, contract, mutated = (
            self.hlsl_resource_pointer_array_parameter_for_call(
                function_name, argument_index
            )
        )
        if parameter is None:
            return None
        parameter_name = parameter.name
        binding = self.hlsl_resource_pointer_array_argument_binding(argument, contract)
        if binding is None:
            raise DirectXResourcePointerArrayError(
                "DirectX fixed storage pointer-array argument for "
                f"'{materialized_name}.{parameter_name}' has no concrete "
                "same-root offset contract",
                function_name=materialized_name,
                array_name=parameter_name,
                address_space=contract.get("address_space"),
                reason="call-backing-unresolved",
                source_location=getattr(argument, "source_location", None),
            )
        resource_name = self.hlsl_resource_type_name(binding.get("resource_type"))
        if resource_name not in {"StructuredBuffer", "RWStructuredBuffer"}:
            raise DirectXResourcePointerArrayError(
                "DirectX fixed storage pointer-array argument for "
                f"'{materialized_name}.{parameter_name}' requires a typed "
                "StructuredBuffer backing resource",
                function_name=materialized_name,
                array_name=parameter_name,
                address_space=contract.get("address_space"),
                backing_root=binding.get("root"),
                reason="resource-kind-mismatch",
                source_location=getattr(argument, "source_location", None),
            )
        expected_physical_type = self.hlsl_resource_pointer_element_type(
            contract.get("resource_type")
        )
        actual_physical_type = self.hlsl_resource_pointer_element_type(
            binding.get("resource_type")
        )
        if self.is_hlsl_bfloat16_type(
            actual_physical_type
        ) or self.hlsl_half_storage.lowered(actual_physical_type):
            actual_physical_type = self.hlsl_structured_storage_type(
                actual_physical_type,
                operation=(
                    "storage pointer-array argument "
                    f"'{materialized_name}.{parameter_name}'"
                ),
                source_location=getattr(argument, "source_location", None),
            )
        if not actual_physical_type or actual_physical_type != expected_physical_type:
            raise DirectXResourcePointerArrayError(
                "DirectX fixed storage pointer-array argument for "
                f"'{materialized_name}.{parameter_name}' changes physical "
                f"backing element type from "
                f"{actual_physical_type or 'unknown'} to "
                f"{expected_physical_type or 'unknown'}",
                function_name=materialized_name,
                array_name=parameter_name,
                address_space=contract.get("address_space"),
                backing_root=binding.get("root"),
                reason="element-type-mismatch",
                source_location=getattr(argument, "source_location", None),
            )
        actual_type = (
            binding.get("view_element_type")
            or binding.get("source_element_type")
            or binding.get("element_type")
        )
        if self.map_type(actual_type) != self.map_type(contract.get("element_type")):
            raise DirectXResourcePointerArrayError(
                "DirectX fixed storage pointer-array argument for "
                f"'{materialized_name}.{parameter_name}' changes element type",
                function_name=materialized_name,
                array_name=parameter_name,
                address_space=contract.get("address_space"),
                backing_root=binding.get("root"),
                reason="element-type-mismatch",
                source_location=getattr(argument, "source_location", None),
            )
        reinterpretation = binding.get("pointer_reinterpretation")
        if reinterpretation is not None and not reinterpretation.get(
            "physical_storage_mapping"
        ):
            raise DirectXResourcePointerArrayError(
                "DirectX fixed storage pointer-array argument for "
                f"'{materialized_name}.{parameter_name}' uses an explicit "
                "pointer reinterpretation without a translator-owned physical "
                "storage mapping",
                function_name=materialized_name,
                array_name=parameter_name,
                address_space=contract.get("address_space"),
                backing_root=binding.get("root"),
                reason="backing-contract-unproven",
                source_location=getattr(argument, "source_location", None),
            )
        if not self.hlsl_resource_pointer_access_satisfies(
            binding.get("access"), contract.get("access")
        ):
            raise DirectXResourcePointerArrayError(
                "DirectX fixed storage pointer-array argument for "
                f"'{materialized_name}.{parameter_name}' cannot satisfy its "
                "storage-access contract",
                function_name=materialized_name,
                array_name=parameter_name,
                address_space=contract.get("address_space"),
                backing_root=binding.get("root"),
                reason="access-mismatch",
                source_location=getattr(argument, "source_location", None),
            )
        if mutated and any(
            not self.hlsl_resource_pointer_offset_lvalue(offset)
            for offset in binding["offset_expressions"]
        ):
            raise DirectXResourcePointerArrayError(
                "DirectX mutable storage pointer-array argument for "
                f"'{materialized_name}.{parameter_name}' has a nonassignable "
                "logical offset",
                function_name=materialized_name,
                array_name=parameter_name,
                address_space=contract.get("address_space"),
                backing_root=binding.get("root"),
                reason="mutable-offset-lvalue-unresolved",
                source_location=getattr(argument, "source_location", None),
            )
        self.record_hlsl_narrow_pointer_argument(
            (
                function_name
                if function_name in self.function_hlsl_resource_pointer_array_parameters
                else materialized_name
            ),
            parameter_name,
            contract.get("element_type"),
            binding,
            argument,
        )
        return {
            **binding,
            "contract": contract,
            "mutated": mutated,
            "call_function_name": materialized_name,
            "call_parameter_name": parameter_name,
        }

    @staticmethod
    def hlsl_resource_pointer_offset_storage_identity(offset):
        """Return the caller-side logical pointer-object storage for an offset."""

        text = str(offset or "").strip()
        if re.fullmatch(r"[A-Za-z_]\w*", text):
            return ("scalar", text, None)
        match = re.fullmatch(r"([A-Za-z_]\w*)\s*\[(.+)\]", text)
        if match is None:
            return None

        index = match.group(2).strip()

        def strip_outer_parentheses(value):
            value = value.strip()
            while value.startswith("(") and value.endswith(")"):
                depth = 0
                encloses_all = True
                for position, character in enumerate(value):
                    if character == "(":
                        depth += 1
                    elif character == ")":
                        depth -= 1
                        if depth == 0 and position != len(value) - 1:
                            encloses_all = False
                            break
                    if depth < 0:
                        encloses_all = False
                        break
                if depth != 0 or not encloses_all:
                    break
                value = value[1:-1].strip()
            return value

        index = strip_outer_parentheses(index)
        changed = True
        while changed:
            changed = False
            for cast_name in ("uint", "int", "uint64_t", "int64_t"):
                prefix = f"{cast_name}("
                if index.startswith(prefix) and index.endswith(")"):
                    index = strip_outer_parentheses(index[len(cast_name) :])
                    changed = True
                    break
        index = strip_outer_parentheses(index)
        literal_index = None
        if re.fullmatch(r"[+-]?\d+", index):
            literal_index = int(index, 10)
        return ("array", match.group(1), literal_index)

    @staticmethod
    def hlsl_resource_pointer_offset_storages_may_alias(left, right):
        if left is None or right is None or left[0] != right[0]:
            return False
        if left[0] == "scalar":
            return left[1] == right[1]
        if left[1] != right[1]:
            return False
        if left[2] is not None and right[2] is not None:
            return left[2] == right[2]
        return True

    def hlsl_validate_resource_pointer_call_aliases(
        self,
        function_name,
        args,
        pointer_array_bindings,
        resource_bindings,
    ):
        """Reject call expansions that would snapshot an aliased pointer object."""

        slots = []
        for argument_index, binding in pointer_array_bindings.items():
            for element_index, offset in enumerate(binding["offset_expressions"]):
                storage = self.hlsl_resource_pointer_offset_storage_identity(offset)
                if storage is None:
                    continue
                slots.append(
                    {
                        "argument_index": argument_index,
                        "element_index": element_index,
                        "kind": "pointer-array",
                        "binding": binding,
                        "function_name": (
                            binding.get("call_function_name") or function_name
                        ),
                        "parameter_name": binding.get("call_parameter_name"),
                        "storage": storage,
                        "mutates": bool(binding.get("mutated")),
                    }
                )

        # A by-value resource pointer intentionally snapshots its pointer value
        # at call entry. Only out/inout pointer-reference parameters continue to
        # observe the caller's pointer object and can alias an array slot or
        # another reference parameter.
        for argument_index, binding in resource_bindings.items():
            direction = binding.get("call_offset_direction")
            if direction not in {"out", "inout"}:
                continue
            offset = self.hlsl_resource_pointer_call_offset(binding)
            storage = self.hlsl_resource_pointer_offset_storage_identity(offset)
            if storage is None:
                continue
            slots.append(
                {
                    "argument_index": argument_index,
                    "element_index": None,
                    "kind": "pointer-reference",
                    "binding": binding,
                    "function_name": binding.get("call_function_name") or function_name,
                    "parameter_name": binding.get("call_parameter_name"),
                    "storage": storage,
                    "mutates": True,
                }
            )

        for left_index, left in enumerate(slots):
            for right in slots[left_index + 1 :]:
                if left["argument_index"] == right["argument_index"]:
                    continue
                if not (left["mutates"] or right["mutates"]):
                    continue
                if not self.hlsl_resource_pointer_offset_storages_may_alias(
                    left["storage"], right["storage"]
                ):
                    continue

                def label(slot):
                    name = (
                        slot["parameter_name"] or f"argument_{slot['argument_index']}"
                    )
                    if slot["element_index"] is not None:
                        return f"{name}[{slot['element_index']}]"
                    return name

                array_slots = [
                    slot for slot in (left, right) if slot["kind"] == "pointer-array"
                ]
                if array_slots:
                    array_slot = next(
                        (slot for slot in array_slots if slot["mutates"]),
                        array_slots[0],
                    )
                    other_slot = right if array_slot is left else left
                    contract = array_slot["binding"]["contract"]
                    raise DirectXResourcePointerArrayError(
                        "DirectX cannot preserve aliasing between storage "
                        f"pointer parameters '{label(left)}' and "
                        f"'{label(right)}' because HLSL expands their logical "
                        "offsets into independent values",
                        function_name=array_slot["function_name"],
                        array_name=array_slot["parameter_name"],
                        address_space=contract.get("address_space"),
                        backing_root=array_slot["binding"].get("root"),
                        conflicting_root=other_slot["binding"].get("root"),
                        reason="cross-parameter-alias-unsupported",
                        source_location=getattr(
                            args[array_slot["argument_index"]],
                            "source_location",
                            None,
                        ),
                    )

                mutable_slot = left if left["mutates"] else right
                contract = mutable_slot["binding"].get("call_contract") or {}
                raise DirectXResourcePointerParameterError(
                    "DirectX cannot preserve aliasing between mutable storage "
                    f"pointer-reference parameters '{label(left)}' and "
                    f"'{label(right)}' because HLSL expands their logical "
                    "offsets into independent values",
                    function_name=mutable_slot["function_name"],
                    parameter_name=mutable_slot["parameter_name"],
                    address_space=contract.get("address_space"),
                    expected_access=contract.get("access"),
                    actual_access=mutable_slot["binding"].get("access"),
                    reason="cross-parameter-alias-unsupported",
                    source_location=getattr(
                        args[mutable_slot["argument_index"]],
                        "source_location",
                        None,
                    ),
                )

    def hlsl_resource_pointer_access_satisfies(self, actual_access, required_access):
        if required_access == "read":
            return actual_access in {"read", "read_write"}
        if required_access == "write":
            return actual_access in {"write", "read_write"}
        return actual_access == "read_write"

    def hlsl_resource_pointer_call_argument_binding(
        self, function_name, argument_index, argument
    ):
        parameter, materialized_name = self.hlsl_resource_pointer_parameter_for_call(
            function_name, argument_index
        )
        if parameter is None:
            return None
        parameter_name = parameter.name
        contract = self.hlsl_resource_pointer_parameter_contract(
            parameter, materialized_name
        )
        binding = self.hlsl_resource_pointer_binding(argument)
        if binding is None or binding.get("kind") == "workgroup-pointer":
            raise DirectXResourcePointerParameterError(
                "DirectX storage pointer argument for "
                f"'{materialized_name}.{parameter_name}' has no concrete "
                "StructuredBuffer or RWStructuredBuffer backing resource",
                function_name=materialized_name,
                parameter_name=parameter_name,
                address_space=contract["address_space"],
                expected_access=contract["access"],
                reason="call-backing-unresolved",
                source_location=getattr(argument, "source_location", None),
            )

        resource_name = self.hlsl_resource_type_name(binding.get("resource_type"))
        if resource_name not in {"StructuredBuffer", "RWStructuredBuffer"}:
            raise DirectXResourcePointerParameterError(
                "DirectX storage pointer argument for "
                f"'{materialized_name}.{parameter_name}' requires a "
                "StructuredBuffer or RWStructuredBuffer root",
                function_name=materialized_name,
                parameter_name=parameter_name,
                address_space=contract["address_space"],
                expected_access=contract["access"],
                actual_access=binding.get("access"),
                reason="resource-kind-mismatch",
                source_location=getattr(argument, "source_location", None),
            )

        expected_type = self.hlsl_resource_pointer_element_type(
            contract["resource_type"]
        )
        actual_type = self.hlsl_resource_pointer_element_type(
            binding.get("resource_type")
        )
        if self.is_hlsl_bfloat16_type(actual_type) or self.hlsl_half_storage.lowered(
            actual_type
        ):
            actual_type = self.hlsl_structured_storage_type(
                actual_type,
                operation=(
                    "storage pointer argument "
                    f"'{materialized_name}.{parameter_name}'"
                ),
                source_location=getattr(argument, "source_location", None),
            )
        if not actual_type or actual_type != expected_type:
            raise DirectXResourcePointerParameterError(
                "DirectX storage pointer argument for "
                f"'{materialized_name}.{parameter_name}' changes element type "
                f"from {actual_type or 'unknown'} to {expected_type}",
                function_name=materialized_name,
                parameter_name=parameter_name,
                address_space=contract["address_space"],
                expected_access=contract["access"],
                actual_access=binding.get("access"),
                reason="element-type-mismatch",
                source_location=getattr(argument, "source_location", None),
            )

        actual_access = binding.get("access")
        if not self.hlsl_resource_pointer_access_satisfies(
            actual_access, contract["access"]
        ):
            raise DirectXResourcePointerParameterError(
                "DirectX storage pointer argument for "
                f"'{materialized_name}.{parameter_name}' requires "
                f"{contract['access']} access but its backing resource provides "
                f"{actual_access or 'unknown'} access",
                function_name=materialized_name,
                parameter_name=parameter_name,
                address_space=contract["address_space"],
                expected_access=contract["access"],
                actual_access=actual_access,
                reason="access-mismatch",
                source_location=getattr(argument, "source_location", None),
            )
        expected_view_layout = scalar_storage_layout(contract["element_type"])
        actual_view_layout = scalar_storage_layout(
            binding.get("view_element_type")
            or binding.get("source_element_type")
            or binding.get("element_type")
        )
        if (
            expected_view_layout is not None
            and actual_view_layout is not None
            and expected_view_layout != actual_view_layout
        ):
            raise DirectXResourcePointerParameterError(
                "DirectX storage pointer argument for "
                f"'{materialized_name}.{parameter_name}' changes logical element "
                f"type from {actual_view_layout.name} to {expected_view_layout.name}",
                function_name=materialized_name,
                parameter_name=parameter_name,
                address_space=contract["address_space"],
                expected_access=contract["access"],
                actual_access=actual_access,
                reason="element-type-mismatch",
                source_location=getattr(argument, "source_location", None),
            )
        binding["call_element_type"] = contract["element_type"]
        binding["call_contract"] = contract
        binding["call_function_name"] = materialized_name
        binding["call_parameter_name"] = parameter_name
        self.record_hlsl_narrow_pointer_argument(
            (
                function_name
                if function_name in self.function_hlsl_resource_pointer_parameters
                else materialized_name
            ),
            parameter_name,
            contract["element_type"],
            binding,
            argument,
        )
        binding["call_offset_direction"] = (
            self.hlsl_resource_pointer_offset_parameter_direction(parameter)
        )
        if binding["call_offset_direction"] is not None:
            call_offset = self.hlsl_resource_pointer_call_offset(binding)
            if not self.hlsl_resource_pointer_offset_lvalue(call_offset):
                raise DirectXResourcePointerParameterError(
                    "DirectX mutable storage pointer argument for "
                    f"'{materialized_name}.{parameter_name}' has no assignable "
                    "logical offset",
                    function_name=materialized_name,
                    parameter_name=parameter_name,
                    address_space=contract["address_space"],
                    expected_access=contract["access"],
                    actual_access=actual_access,
                    reason="mutable-offset-lvalue-unresolved",
                    source_location=getattr(argument, "source_location", None),
                )
        return binding

    @staticmethod
    def hlsl_resource_pointer_offset_lvalue(offset):
        text = str(offset or "").strip()
        return re.fullmatch(r"[A-Za-z_]\w*(?:\s*\[[^\]]+\])?", text) is not None

    def hlsl_resource_pointer_call_offset(self, binding):
        element_offset = str(binding.get("offset", "0"))
        reinterpretation = binding.get("pointer_reinterpretation")
        if reinterpretation is None:
            return element_offset

        view_layout = reinterpretation["target_layout"]
        parameter_layout = scalar_storage_layout(binding.get("call_element_type"))
        if parameter_layout is None or parameter_layout != view_layout:
            return element_offset

        byte_offset = str(binding.get("byte_offset", "0"))
        element_bytes = element_offset
        if view_layout.byte_width != 1 and element_offset not in {"0", "0u"}:
            element_bytes = f"({element_offset} * {view_layout.byte_width})"
        if byte_offset in {"0", "0u"}:
            total_bytes = element_bytes
        elif element_bytes in {"0", "0u"}:
            total_bytes = byte_offset
        else:
            total_bytes = f"({byte_offset} + {element_bytes})"
        if view_layout.byte_width == 1:
            return total_bytes
        return f"(({total_bytes}) / {view_layout.byte_width})"

    def hlsl_workgroup_pointer_call_argument_binding(
        self, function_name, argument_index, argument
    ):
        parameter_name = self.function_hlsl_workgroup_pointer_parameter_indices.get(
            function_name, {}
        ).get(argument_index)
        if parameter_name is None:
            materialized_name = self.hlsl_materialized_function_name(function_name)
            parameter_name = self.function_hlsl_workgroup_pointer_parameter_indices.get(
                materialized_name, {}
            ).get(argument_index)
            function_name = materialized_name
        if parameter_name is None:
            return None

        parameters = self.function_hlsl_workgroup_pointer_parameters.get(
            function_name, []
        )
        parameter = next(
            (parameter for parameter in parameters if parameter.name == parameter_name),
            None,
        )
        expected_type = (
            self.hlsl_workgroup_pointer_element_type(parameter)
            if parameter is not None
            else None
        )
        binding = self.hlsl_resource_pointer_binding(argument)
        if binding is None and self.hlsl_null_pointer_expression(argument):
            binding = self.hlsl_null_workgroup_pointer_binding(expected_type)
        if binding is None or binding.get("kind") != "workgroup-pointer":
            raise self.hlsl_workgroup_pointer_error(
                "DirectX workgroup pointer argument for "
                f"'{function_name}.{parameter_name}' has no concrete shared-array "
                "backing object",
                function_name=function_name,
                parameter_name=parameter_name,
                reason="call-backing-unresolved",
                node=argument,
            )
        if binding.get("root_kind") == "null":
            # Null contracts carry no storage; the callee specialization was
            # proven never to observe the parameter, so its offset slot is inert.
            return {
                **self.hlsl_null_workgroup_pointer_binding(
                    binding.get("element_type") or expected_type
                ),
                "offset": "0",
            }
        extent = binding.get("array_size")
        if not isinstance(extent, int) or extent <= 0:
            raise self.hlsl_workgroup_pointer_error(
                "DirectX workgroup pointer argument for "
                f"'{function_name}.{parameter_name}' has no concrete shared-array "
                "extent",
                function_name=function_name,
                parameter_name=parameter_name,
                reason="missing-concrete-size",
                node=argument,
            )
        if expected_type and binding.get("element_type") != expected_type:
            raise self.hlsl_workgroup_pointer_error(
                "DirectX workgroup pointer argument for "
                f"'{function_name}.{parameter_name}' changes element type",
                function_name=function_name,
                parameter_name=parameter_name,
                reason="element-type-mismatch",
                node=argument,
            )
        return binding

    def hlsl_workgroup_pointer_call_variant(self, function_name, arguments):
        resolved_name = function_name
        parameter_indices = self.function_hlsl_workgroup_pointer_parameter_indices.get(
            resolved_name
        )
        if parameter_indices is None:
            resolved_name = self.hlsl_materialized_function_name(function_name)
            parameter_indices = (
                self.function_hlsl_workgroup_pointer_parameter_indices.get(
                    resolved_name
                )
            )
        if not parameter_indices:
            return None

        contracts = {}
        for index, parameter_name in parameter_indices.items():
            argument = arguments[index] if index < len(arguments) else None
            binding = self.hlsl_workgroup_pointer_call_argument_binding(
                resolved_name, index, argument
            )
            contracts[parameter_name] = (
                binding["root"],
                binding["array_size"],
            )
        parameters = self.function_hlsl_workgroup_pointer_parameters[resolved_name]
        variant = tuple(contracts[parameter.name] for parameter in parameters)
        return resolved_name, variant

    def hlsl_workgroup_pointer_specialization_name(
        self, function_name, variant, base_name=None
    ):
        parameters = self.function_hlsl_workgroup_pointer_parameters.get(
            function_name, ()
        )
        if not parameters or not variant:
            return base_name or self.hlsl_function_call_name(function_name)

        suffix_parts = []
        for parameter, contract in zip(parameters, variant):
            root, extent = contract
            suffix_parts.extend(
                (
                    self.hlsl_sanitized_identifier(parameter.name),
                    self.hlsl_sanitized_identifier(root),
                    str(extent),
                )
            )
        base_name = base_name or self.hlsl_function_call_name(function_name)
        suffix = "_".join(suffix_parts)
        return self.hlsl_sanitized_identifier(
            f"{base_name}__crosstl_workgroup_{suffix}"
        )

    def hlsl_workgroup_pointer_call_name(self, function_name, arguments):
        call_variant = self.hlsl_workgroup_pointer_call_variant(
            function_name, arguments
        )
        if call_variant is None:
            return None
        resolved_name, variant = call_variant
        return self.hlsl_workgroup_pointer_specialization_name(resolved_name, variant)

    def hlsl_resource_pointer_element_type(self, resource_type):
        resource_type = str(resource_type or "").strip()
        if "<" not in resource_type or not resource_type.endswith(">"):
            return None
        return resource_type.split("<", 1)[1][:-1].strip() or None

    def hlsl_storage_pointer_array_declaration_type(self, node):
        vtype = getattr(node, "var_type", getattr(node, "vtype", None))
        if not isinstance(vtype, ArrayType):
            self.hlsl_reject_unresolved_generic_pointer_array(
                vtype,
                array_name=getattr(node, "name", None),
                function_name=self.current_function_name,
                source_location=getattr(node, "source_location", None),
            )
            return None
        array_types = []
        element_type = vtype
        while isinstance(element_type, ArrayType):
            array_types.append(element_type)
            element_type = getattr(element_type, "element_type", None)
        if not isinstance(element_type, PointerType):
            self.hlsl_reject_unresolved_generic_pointer_array(
                vtype,
                array_name=getattr(node, "name", None),
                function_name=self.current_function_name,
                source_location=getattr(node, "source_location", None),
            )
            return None
        pointer_type = element_type
        address_space = str(pointer_type.address_space or "").strip().lower()
        if not address_space:
            qualifiers = {
                str(qualifier).lower()
                for qualifier in getattr(node, "qualifiers", []) or []
            }
            address_space = next(
                (
                    candidate
                    for candidate in (
                        "constant",
                        "device",
                        "global",
                        "storage",
                        "threadgroup",
                        "thread",
                        "private",
                        "function",
                    )
                    if candidate in qualifiers
                ),
                "thread",
            )
        if len(array_types) != 1:
            raise DirectXResourcePointerArrayError(
                "DirectX cannot faithfully lower multidimensional storage "
                f"pointer array '{getattr(node, 'name', None)}'",
                function_name=self.current_function_name,
                array_name=getattr(node, "name", None),
                address_space=address_space,
                reason="multidimensional-pointer-array-unsupported",
                source_location=getattr(node, "source_location", None),
            )
        if address_space not in {"constant", "device", "global", "storage"}:
            raise DirectXResourcePointerArrayError(
                "DirectX cannot faithfully lower fixed pointer array "
                f"'{getattr(node, 'name', None)}' from address space "
                f"'{address_space}'",
                function_name=self.current_function_name,
                array_name=getattr(node, "name", None),
                address_space=address_space,
                reason="pointer-array-address-space-unsupported",
                source_location=getattr(node, "source_location", None),
            )
        return array_types[0], pointer_type, address_space

    def hlsl_resource_pointer_array_error(
        self,
        message,
        *,
        binding=None,
        node=None,
        reason=None,
    ):
        binding = binding or {}
        declaration = binding.get("declaration_node")
        source_node = node or declaration
        return DirectXResourcePointerArrayError(
            message,
            function_name=self.current_function_name,
            array_name=binding.get("array_name") or getattr(declaration, "name", None),
            address_space=binding.get("address_space"),
            backing_root=binding.get("root"),
            conflicting_root=binding.get("conflicting_root"),
            reason=reason,
            source_location=getattr(source_node, "source_location", None),
        )

    def generate_hlsl_resource_pointer_array_declaration(self, node, indent):
        declaration_type = self.hlsl_storage_pointer_array_declaration_type(node)
        if declaration_type is None:
            return None
        binding = self.current_hlsl_resource_pointer_aliases.get(node.name)
        if binding is None or binding.get("declaration_id") != id(node):
            raise self.hlsl_resource_pointer_array_error(
                f"DirectX storage pointer array '{node.name}' has no proven "
                "canonical initialization loop",
                binding={
                    "array_name": node.name,
                    "address_space": declaration_type[2],
                    "declaration_node": node,
                },
                node=node,
                reason="initialization-unproven",
            )
        return (
            f"{'    ' * indent}int64_t {binding['offset_array']}"
            f"[{binding['extent']}];\n"
        )

    def hlsl_resource_pointer_array_for_access(self, expression):
        if not isinstance(expression, ArrayAccessNode):
            return None
        array = getattr(expression, "array", getattr(expression, "array_expr", None))
        if not isinstance(array, (IdentifierNode, VariableNode)):
            return None
        binding = self.current_hlsl_resource_pointer_aliases.get(array.name)
        if binding is None or binding.get("kind") != "resource-pointer-array":
            return None
        return binding

    def hlsl_resource_pointer_array_index_range(self, expression):
        literal_index = self.literal_int_value(
            expression, self.hlsl_current_visible_int_constants()
        )
        if isinstance(literal_index, int) and not isinstance(literal_index, bool):
            return literal_index, literal_index + 1
        if isinstance(expression, (IdentifierNode, VariableNode)):
            return self.current_hlsl_bounded_loop_indices.get(expression.name)
        if not isinstance(expression, BinaryOpNode) or expression.op not in {"+", "-"}:
            return None
        left_range = self.hlsl_resource_pointer_array_index_range(expression.left)
        right_value = self.literal_int_value(
            expression.right, self.hlsl_current_visible_int_constants()
        )
        if left_range is not None and isinstance(right_value, int):
            delta = right_value if expression.op == "+" else -right_value
            return left_range[0] + delta, left_range[1] + delta
        if expression.op == "+":
            right_range = self.hlsl_resource_pointer_array_index_range(expression.right)
            left_value = self.literal_int_value(
                expression.left, self.hlsl_current_visible_int_constants()
            )
            if right_range is not None and isinstance(left_value, int):
                return right_range[0] + left_value, right_range[1] + left_value
        return None

    @staticmethod
    def hlsl_resource_pointer_array_effective_index(binding, rendered_index):
        """Apply a by-value array-parameter base to a proven element index."""

        base_index = binding.get("base_index")
        if not base_index:
            return rendered_index
        if str(rendered_index).strip() in {"0", "0u", "int(0)", "uint(0)"}:
            return base_index
        return f"({base_index} + int64_t({rendered_index}))"

    def hlsl_resource_pointer_array_index(self, binding, index_expression):
        index_range = self.hlsl_resource_pointer_array_index_range(index_expression)
        rendered_index = self.generate_expression(index_expression)
        if index_range is None or not (
            0 <= index_range[0] <= index_range[1] <= binding["extent"]
        ):
            reason = "index-unproven" if index_range is None else "index-out-of-bounds"
            raise self.hlsl_resource_pointer_array_error(
                f"DirectX cannot prove bounded indexing of storage pointer array "
                f"'{binding.get('array_name') or binding.get('parameter_name')}' "
                f"by '{rendered_index}'",
                binding=binding,
                node=index_expression,
                reason=reason,
            )
        return (
            self.hlsl_resource_pointer_array_effective_index(binding, rendered_index),
            index_range,
        )

    def hlsl_for_index_range(self, node):
        init = getattr(node, "init", None)
        condition = getattr(node, "condition", None)
        update = getattr(node, "update", None)
        if isinstance(update, list) and len(update) == 1:
            update = update[0]
        if not isinstance(init, VariableNode) or not isinstance(
            condition, BinaryOpNode
        ):
            return None
        name = init.name
        constants = self.hlsl_current_visible_int_constants()
        start = self.literal_int_value(getattr(init, "initial_value", None), constants)
        stop = self.literal_int_value(getattr(condition, "right", None), constants)
        condition_name = getattr(getattr(condition, "left", None), "name", None)
        operator = getattr(condition, "operator", getattr(condition, "op", None))
        if not (
            isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(stop, int)
            and not isinstance(stop, bool)
            and 0 <= start <= stop
            and condition_name == name
            and operator == "<"
            and isinstance(update, UnaryOpNode)
            and getattr(update, "operator", getattr(update, "op", None)) == "++"
            and getattr(getattr(update, "operand", None), "name", None) == name
        ):
            return None
        return name, start, stop

    def hlsl_canonical_for_index_range(self, node):
        loop_range = self.hlsl_for_index_range(node)
        if loop_range is None or loop_range[1] != 0:
            return None
        return loop_range[0], loop_range[2]

    @staticmethod
    def hlsl_single_statement_expression(statement):
        return getattr(statement, "expression", statement)
