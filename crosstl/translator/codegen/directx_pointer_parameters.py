"""Pointer and resource parameters in the HLSL generator: private pointer analysis, views and access, parameter bindings, generic contracts, omitted arguments and write refusals."""

from ..ast import (
    ArrayAccessNode,
    ArrayNode,
    ArrayType,
    AssignmentNode,
    BlockNode,
    DoWhileNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    LoopNode,
    PointerType,
    RangeNode,
    SwitchNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)
from .array_utils import split_array_type_suffix
from .directx_values import (
    DirectXPrivatePointerParameterError,
    DirectXResourcePointerArrayError,
    DirectXResourcePointerParameterError,
)
from .enum_utils import generic_type_parts
from .hlsl_storage_array_views import (
    storage_array_pointer_type,
    view_storage_array_parameters,
)
from .pointer_reinterpret import (
    PointerReinterpretationError,
    scalar_storage_layout,
)


class DirectXPointerParameterMixin:
    """Pointer parameter members of ``HLSLCodeGen``."""

    def generate_hlsl_private_pointer_view_access(
        self,
        pointer_expression,
        index_expression=0,
        *,
        access_expression=None,
    ):
        binding = self.hlsl_private_pointer_view_binding(pointer_expression)
        if binding is None or binding.get("root_kind") not in {
            "parameter",
            "scalar_parameter",
            "word_view_parameter",
        }:
            return None
        if binding["root_kind"] == "word_view_parameter":
            if self.hlsl_private_pointer_expression_has_side_effects(index_expression):
                raise DirectXPrivatePointerParameterError(
                    "DirectX cannot emit a side-effecting private scalar-view index",
                    function_name=self.current_function_name,
                    reason="side-effecting-view-offset",
                    source_location=getattr(index_expression, "source_location", None),
                )
            layout = self.hlsl_private_pointer_word_views[binding["word_view"]]
            rendered_index = (
                str(index_expression)
                if isinstance(index_expression, (int, float))
                else self.generate_expression(index_expression)
            )
            effective_index = self.hlsl_resource_pointer_offset_sum(
                binding["offset"], rendered_index
            )
            return self.hlsl_pointer_reinterpret_read_expression(
                {
                    "root": binding["backing"],
                    "byte_offset": "0",
                    "pointer_reinterpretation": {
                        "source_layout": layout["source_layout"],
                        "target_layout": layout["target_layout"],
                        "private_word_view": layout,
                    },
                },
                effective_index,
            )
        if binding["root_kind"] == "scalar_parameter":
            if self.hlsl_private_pointer_expression_has_side_effects(index_expression):
                raise DirectXPrivatePointerParameterError(
                    "DirectX cannot emit a side-effecting scalar pointer index",
                    function_name=self.current_function_name,
                    reason="side-effecting-view-offset",
                    source_location=getattr(index_expression, "source_location", None),
                )
            proven_zero_access = (
                access_expression is not None
                and id(access_expression)
                in self.hlsl_private_pointer_zero_index_access_ids
            )
            literal_index = self.literal_int_value(
                index_expression, self.literal_int_constants
            )
            if not proven_zero_access and not (
                binding.get("offset") in {None, "", "0", "0u"} and literal_index == 0
            ):
                raise DirectXPrivatePointerParameterError(
                    "DirectX scalar private pointer views support only index zero",
                    function_name=self.current_function_name,
                    reason="unprovable-view-offset",
                    source_location=getattr(index_expression, "source_location", None),
                )
            return binding["backing"]
        rendered_index = (
            str(index_expression)
            if isinstance(index_expression, (int, float))
            else self.generate_expression(index_expression)
        )
        return f"{binding['backing']}[({binding['offset']} + {rendered_index})]"

    def hlsl_private_pointer_call_argument_binding(
        self, function_name, argument_index, argument
    ):
        parameter_name = self.function_private_pointer_parameter_indices.get(
            function_name, {}
        ).get(argument_index)
        if parameter_name is None:
            return None
        binding = self.hlsl_private_pointer_view_binding(argument)
        if binding is None:
            raise DirectXPrivatePointerParameterError(
                "DirectX private pointer argument for "
                f"'{function_name}.{parameter_name}' has no concrete backing array",
                function_name=function_name,
                parameter_name=parameter_name,
                reason="missing-view-backing",
                source_location=getattr(argument, "source_location", None),
            )
        binding["scalar"] = (
            parameter_name in (
                self.function_private_pointer_scalar_parameters.get(
                    function_name, set()
                )
            )
            and binding.get("root_kind") != "local_word_view"
        )
        return binding

    def hlsl_private_pointer_word_view_write_error(self, target):
        scalar_struct = self.hlsl_private_scalar_struct_reinterpret_contract(target)
        if scalar_struct is not None:
            self.hlsl_private_pointer_word_view_error(
                scalar_struct["reinterpretation"],
                source_type=scalar_struct["source_type"],
                target_type=scalar_struct["target_type"],
                reason="private-scalar-struct-view-write-unsupported",
                detail="the scalar-to-struct reinterpretation is read-only",
                alignment=scalar_struct["member_layout"].byte_width,
                access="write",
            )
        if isinstance(target, ArrayAccessNode):
            pointer_expression = target.array
        elif (
            isinstance(target, UnaryOpNode)
            and not getattr(target, "is_postfix", False)
            and target.op == "*"
        ):
            pointer_expression = target.operand
        else:
            return
        binding = self.hlsl_private_pointer_view_binding(pointer_expression)
        if binding is None or binding.get("root_kind") != "word_view_parameter":
            return
        layout = self.hlsl_private_pointer_word_views[binding["word_view"]]
        raise PointerReinterpretationError(
            "DirectX cannot preserve writes through a reinterpreted private "
            "word-array view",
            source_type=layout["source_type"],
            target_type=layout["target_layout"].name,
            address_space="thread",
            alignment=layout["target_layout"].byte_width,
            access="write",
            target_backend="directx",
            reason="private-view-write-unsupported",
            source_location=getattr(target, "source_location", None),
        )

    def hlsl_generic_pointer_array_type_info(self, type_value):
        """Describe generic ``array<ptr<...>>`` types lacking pointer metadata."""

        rendered_type = self.type_name_string(type_value)
        if not rendered_type:
            return None
        type_name = str(rendered_type).strip()
        element_name, array_suffix = split_array_type_suffix(type_name)
        array_depth = array_suffix.count("[")
        element_name = element_name.strip()
        while element_name:
            base_name, generic_args = generic_type_parts(element_name)
            if base_name.rsplit("::", 1)[-1] != "array" or not generic_args:
                break
            array_depth += 1
            element_name = str(generic_args[0]).strip()
        if not array_depth:
            return None
        base_name, generic_args = generic_type_parts(element_name)
        if base_name.rsplit("::", 1)[-1] != "ptr":
            return None
        return {
            "array_depth": array_depth,
            "pointee_type": str(generic_args[-1]).strip() if generic_args else None,
            "type_name": type_name,
        }

    def hlsl_reject_unresolved_generic_pointer_array(
        self,
        type_value,
        *,
        array_name,
        function_name,
        source_location=None,
    ):
        type_info = self.hlsl_generic_pointer_array_type_info(type_value)
        if type_info is None:
            return
        raise DirectXResourcePointerArrayError(
            "DirectX cannot faithfully lower generic pointer-array declaration "
            f"'{function_name}.{array_name}' because type "
            f"'{type_info['type_name']}' has no pointer address-space contract",
            function_name=function_name,
            array_name=array_name,
            address_space="unresolved",
            reason="pointer-array-address-space-unresolved",
            source_location=source_location,
        )

    def hlsl_resource_pointer_array_parameter_type_node(self, parameter):
        parameter_type = getattr(
            parameter,
            "param_type",
            getattr(parameter, "var_type", getattr(parameter, "vtype", None)),
        )
        if not isinstance(parameter_type, ArrayType):
            self.hlsl_reject_unresolved_generic_pointer_array(
                parameter_type,
                array_name=getattr(parameter, "name", None),
                function_name=self.current_function_name,
                source_location=getattr(parameter, "source_location", None),
            )
            return None
        array_types = []
        element_type = parameter_type
        while isinstance(element_type, ArrayType):
            array_types.append(element_type)
            element_type = getattr(element_type, "element_type", None)
        if not isinstance(element_type, PointerType):
            self.hlsl_reject_unresolved_generic_pointer_array(
                parameter_type,
                array_name=getattr(parameter, "name", None),
                function_name=self.current_function_name,
                source_location=getattr(parameter, "source_location", None),
            )
            return None
        pointer_type = element_type
        address_space = str(pointer_type.address_space or "").strip().lower()
        if not address_space:
            qualifiers = {
                str(qualifier).lower()
                for qualifier in getattr(parameter, "qualifiers", []) or []
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
        return tuple(array_types), pointer_type, address_space

    def hlsl_resource_pointer_array_parameter_contract(
        self, parameter, function_name=None
    ):
        type_info = self.hlsl_resource_pointer_array_parameter_type_node(parameter)
        if type_info is None:
            return None
        array_types, pointer_type, address_space = type_info
        parameter_name = getattr(parameter, "name", None)
        if address_space not in {"constant", "device", "global", "storage"}:
            raise DirectXResourcePointerArrayError(
                "DirectX cannot faithfully lower fixed pointer-array parameter "
                f"'{function_name}.{parameter_name}' from address space "
                f"'{address_space}'",
                function_name=function_name,
                array_name=parameter_name,
                address_space=address_space,
                reason="pointer-array-address-space-unsupported",
                source_location=getattr(parameter, "source_location", None),
            )
        if len(array_types) != 1:
            raise DirectXResourcePointerArrayError(
                "DirectX cannot faithfully lower multidimensional storage "
                f"pointer-array parameter '{function_name}.{parameter_name}'",
                function_name=function_name,
                array_name=parameter_name,
                address_space=address_space,
                reason="multidimensional-pointer-array-unsupported",
                source_location=getattr(parameter, "source_location", None),
            )
        array_type = array_types[0]
        extent = self.literal_int_value(
            getattr(array_type, "size", None), self.literal_int_constants
        )
        if not isinstance(extent, int) or isinstance(extent, bool) or extent <= 0:
            raise DirectXResourcePointerArrayError(
                "DirectX fixed storage pointer-array parameter "
                f"'{function_name}.{parameter_name}' requires a positive "
                "compile-time extent",
                function_name=function_name,
                array_name=parameter_name,
                address_space=address_space,
                reason="pointer-array-extent-unresolved",
                source_location=getattr(parameter, "source_location", None),
            )
        pointee_type = getattr(pointer_type, "pointee_type", None)
        if isinstance(pointee_type, (PointerType, ArrayType)) or pointee_type is None:
            raise DirectXResourcePointerArrayError(
                "DirectX fixed storage pointer-array parameter "
                f"'{function_name}.{parameter_name}' has an unsupported pointee type",
                function_name=function_name,
                array_name=parameter_name,
                address_space=address_space,
                reason="unsupported-pointee-type",
                source_location=getattr(parameter, "source_location", None),
            )
        element_type = self.type_name_string(pointee_type)
        storage_type = self.hlsl_structured_storage_type(
            element_type,
            operation=(
                "storage pointer-array parameter "
                f"'{function_name}.{parameter_name}' element"
            ),
            source_location=getattr(parameter, "source_location", None),
        )
        if not storage_type or any(
            marker in str(storage_type) for marker in ("*", "&", "[")
        ):
            raise DirectXResourcePointerArrayError(
                "DirectX fixed storage pointer-array parameter "
                f"'{function_name}.{parameter_name}' has an unsupported element type",
                function_name=function_name,
                array_name=parameter_name,
                address_space=address_space,
                reason="unsupported-pointee-type",
                source_location=getattr(parameter, "source_location", None),
            )
        access_mode = str(getattr(pointer_type, "access_mode", None) or "").lower()
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(parameter, "qualifiers", []) or []
        }
        if (
            address_space == "constant"
            or access_mode in {
                "read",
                "readonly",
            }
            or qualifiers.intersection({"const", "constant", "read", "readonly", "in"})
        ):
            access = "read"
        elif access_mode in {"write", "writeonly"} or qualifiers.intersection(
            {"write", "writeonly", "out"}
        ):
            access = "write"
        else:
            access = "read_write"
        resource_name = "StructuredBuffer" if access == "read" else "RWStructuredBuffer"
        return {
            "address_space": address_space,
            "access": access,
            "element_type": element_type,
            "extent": extent,
            "resource_type": f"{resource_name}<{storage_type}>",
        }

    def hlsl_resource_pointer_array_parameter_mutation_analysis(
        self, function, parameter_name, *, lexical=None
    ):
        """Return direct offset mutation and exact whole-array forwarding edges."""

        lexical = lexical or self.hlsl_function_lexical_type_bindings(function)
        parameter_id = lexical["parameter_ids"].get(parameter_name)
        reference_bindings = lexical["reference_bindings"]

        def direct_parameter_reference(value):
            if not isinstance(value, (str, IdentifierNode, VariableNode)) or (
                self.expression_name(value) != parameter_name
            ):
                return False
            if isinstance(value, str):
                return True
            binding_id = reference_bindings.get(id(value))
            return binding_id is None or binding_id == parameter_id

        def direct_array_target(value):
            if not isinstance(value, ArrayAccessNode):
                return False
            array = getattr(value, "array", getattr(value, "array_expr", None))
            return direct_parameter_reference(array)

        direct_mutation = False
        dependencies = set()
        for node in self.walk_ast(getattr(function, "body", [])):
            if isinstance(node, AssignmentNode):
                target = getattr(node, "target", getattr(node, "left", None))
                if direct_array_target(target):
                    direct_mutation = True
            if isinstance(node, UnaryOpNode) and self.map_operator(node.op) in {
                "++",
                "--",
            }:
                if direct_array_target(node.operand):
                    direct_mutation = True
            if not isinstance(node, FunctionCallNode):
                continue
            arguments = list(
                getattr(node, "arguments", getattr(node, "args", [])) or []
            )
            callee_name = self.hlsl_resource_pointer_analysis_call_name(node)
            for index, argument in enumerate(arguments):
                if direct_parameter_reference(argument):
                    if not callee_name:
                        continue
                    callee_parameter, materialized_name, _contract, _mutated = (
                        self.hlsl_resource_pointer_array_parameter_for_call(
                            callee_name, index
                        )
                    )
                    if callee_parameter is not None:
                        dependencies.add((materialized_name, callee_parameter.name))
                    continue
                if not direct_array_target(argument):
                    continue
                callee_parameter, _materialized_name = (
                    self.hlsl_resource_pointer_analysis_call_parameter(node, index)
                )
                if callee_parameter is not None and (
                    self.hlsl_resource_pointer_offset_parameter_direction(
                        callee_parameter
                    )
                    in {"out", "inout"}
                ):
                    # A pointer-array element is itself a logical offset lvalue.
                    # Forwarding it to a pointer-reference helper must therefore
                    # write the helper's updated offset back through this array
                    # parameter and, in turn, through every caller.
                    direct_mutation = True
        return direct_mutation, dependencies

    def hlsl_resource_pointer_parameter_type_node(self, parameter):
        parameter_type = getattr(
            parameter, "param_type", getattr(parameter, "vtype", None)
        )
        if isinstance(parameter_type, PointerType):
            return parameter_type
        # A storage array parameter that receives a view into a buffer is a
        # storage pointer; see hlsl_storage_array_views.
        view = self.hlsl_view_storage_array_parameters.get(id(parameter))
        return view[1] if view is not None and view[0] is parameter else None

    def hlsl_resource_pointer_parameter_address_space(self, parameter):
        pointer_type = self.hlsl_resource_pointer_parameter_type_node(parameter)
        if pointer_type is None:
            return None
        address_space = str(pointer_type.address_space or "").strip().lower()
        if not address_space:
            qualifiers = {
                str(qualifier).lower()
                for qualifier in getattr(parameter, "qualifiers", []) or []
            }
            address_space = next(
                (
                    candidate
                    for candidate in ("device", "global", "storage")
                    if candidate in qualifiers
                ),
                "",
            )
            # CrossGL spells a storage-buffer pointer parameter `buffer T* p`.
            if not address_space and "buffer" in qualifiers:
                address_space = "storage"
        return address_space or None

    def hlsl_resource_pointer_parameter(self, parameter):
        return self.hlsl_resource_pointer_parameter_address_space(parameter) in {
            "constant",
            "device",
            "global",
            "storage",
        }

    def hlsl_resource_pointer_parameter_access(self, parameter):
        if self.hlsl_resource_access_promoted(parameter):
            return "read_write"
        pointer_type = self.hlsl_resource_pointer_parameter_type_node(parameter)
        access_mode = str(getattr(pointer_type, "access_mode", None) or "").lower()
        access_aliases = {
            "read": "read",
            "readonly": "read",
            "write": "write",
            "writeonly": "write",
            "read_write": "read_write",
            "readwrite": "read_write",
        }
        if access_mode in access_aliases:
            return access_aliases[access_mode]

        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(parameter, "qualifiers", []) or []
        }
        if qualifiers.intersection({"const", "constant", "read", "readonly", "in"}):
            return "read"
        if qualifiers.intersection({"write", "writeonly"}):
            return "write"
        if qualifiers.intersection({"read_write", "readwrite", "inout"}):
            return "read_write"
        if "out" in qualifiers:
            return "write"
        if pointer_type is not None and not getattr(pointer_type, "is_mutable", True):
            return "read"
        return "read_write"

    def hlsl_resource_pointer_offset_parameter_direction(self, parameter):
        if id(parameter) in self.hlsl_view_storage_array_parameters:
            # An array parameter decays to a pointer passed by value; its
            # ``inout`` qualifier covers the elements, not the view.
            return None
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(parameter, "qualifiers", []) or []
        }
        if "inout" in qualifiers or "read_write" in qualifiers:
            return "inout"
        if "out" in qualifiers or "writeonly" in qualifiers:
            return "out"
        return None

    def hlsl_resource_pointer_parameter_contract(self, parameter, function_name=None):
        if not self.hlsl_resource_pointer_parameter(parameter):
            return None
        pointer_type = self.hlsl_resource_pointer_parameter_type_node(parameter)
        pointee_type = getattr(pointer_type, "pointee_type", None)
        parameter_name = getattr(parameter, "name", None)
        address_space = self.hlsl_resource_pointer_parameter_address_space(parameter)
        if isinstance(pointee_type, (PointerType, ArrayType)) or pointee_type is None:
            raise DirectXResourcePointerParameterError(
                "DirectX storage pointer parameter "
                f"'{function_name}.{parameter_name}' has an unsupported pointee "
                "type",
                function_name=function_name,
                parameter_name=parameter_name,
                address_space=address_space,
                reason="unsupported-pointee-type",
                source_location=getattr(parameter, "source_location", None),
            )

        element_type = self.type_name_string(pointee_type)
        mapped_element_type = self.hlsl_structured_storage_type(
            element_type,
            operation=(f"storage pointer parameter '{function_name}.{parameter_name}'"),
            source_location=getattr(parameter, "source_location", None),
        )
        if not mapped_element_type or any(
            marker in str(mapped_element_type) for marker in ("*", "&", "[")
        ):
            raise DirectXResourcePointerParameterError(
                "DirectX storage pointer parameter "
                f"'{function_name}.{parameter_name}' has an unsupported pointee "
                "type",
                function_name=function_name,
                parameter_name=parameter_name,
                address_space=address_space,
                reason="unsupported-pointee-type",
                source_location=getattr(parameter, "source_location", None),
            )

        access = self.hlsl_resource_pointer_parameter_access(parameter)
        resource_name = "StructuredBuffer" if access == "read" else "RWStructuredBuffer"
        return {
            "address_space": address_space,
            "access": access,
            "element_type": element_type,
            "resource_type": f"{resource_name}<{mapped_element_type}>",
        }

    def hlsl_resource_pointer_parameter_alias_binding(
        self, contract, *, root, offset, logical_storage=False
    ):
        logical_element_type = contract["element_type"]
        physical_element_type = self.hlsl_resource_pointer_element_type(
            contract["resource_type"]
        )
        binding = {
            "kind": "resource-pointer",
            "root": root,
            "root_kind": "parameter",
            "offset": offset,
            "element_type": logical_element_type,
            "source_element_type": logical_element_type,
            "resource_type": contract["resource_type"],
            "address_space": contract["address_space"],
            "access": contract["access"],
            "is_pointer_alias": True,
        }
        source_layout = scalar_storage_layout(physical_element_type)
        target_layout = scalar_storage_layout(logical_element_type)
        # Root buffers have one physical element per logical source element.
        # Widening a byte carrier does not turn that buffer into packed words.
        if (
            not logical_storage
            and source_layout is not None
            and target_layout is not None
            and source_layout != target_layout
            and source_layout.bit_width == 32
            and target_layout.bit_width in {8, 16, 32}
            and not (target_layout.kind == "floating" and target_layout.bit_width != 32)
        ):
            binding.update(
                {
                    "byte_offset": "0",
                    "view_element_type": logical_element_type,
                    "source_element_type": physical_element_type,
                    "pointer_reinterpretation": {
                        "source_layout": source_layout,
                        "target_layout": target_layout,
                        "physical_storage_mapping": True,
                    },
                }
            )
        return binding

    def hlsl_function_local_type_environment(self, function):
        """Collect only unambiguous function-wide source type hints."""

        source_types = {}
        rendered_types = {}
        signatures = {}
        ambiguous = set()

        def register(declaration):
            name = getattr(declaration, "name", None)
            value_type = getattr(
                declaration,
                "param_type",
                getattr(
                    declaration,
                    "var_type",
                    getattr(declaration, "vtype", None),
                ),
            )
            type_name = self.type_name_string(value_type)
            if not name or not type_name or type_name == "auto":
                return
            signature = self.hlsl_source_type_signature(value_type)
            previous = signatures.get(name)
            if previous is not None and previous != signature:
                ambiguous.add(name)
                source_types.pop(name, None)
                rendered_types.pop(name, None)
                return
            if name not in ambiguous:
                signatures[name] = signature
                source_types[name] = value_type
                rendered_types[name] = type_name

        for parameter in (
            getattr(function, "parameters", getattr(function, "params", [])) or []
        ):
            register(parameter)
        for declaration in self.walk_ast(getattr(function, "body", [])):
            if isinstance(declaration, (VariableNode, ArrayNode)):
                register(declaration)
        return rendered_types, source_types

    def hlsl_function_lexical_type_bindings(self, function):
        """Capture exact visible source types and declaration IDs at each call."""

        call_source_types = {}
        declarations = {}
        reference_bindings = {}
        parameter_ids = {}

        def visible_source_types(visible):
            return {
                name: binding["source_type"]
                for name, binding in visible.items()
                if binding.get("source_type") is not None
            }

        def expression_source_type(value, visible):
            previous_source_types = self.local_variable_source_types
            previous_types = self.local_variable_types
            scoped_types = visible_source_types(visible)
            self.local_variable_source_types = {
                **previous_source_types,
                **scoped_types,
            }
            self.local_variable_types = {
                **previous_types,
                **scoped_types,
            }
            try:
                return self.hlsl_source_expression_type(value)
            finally:
                self.local_variable_source_types = previous_source_types
                self.local_variable_types = previous_types

        def declaration_source_type(node, visible):
            raw_type = getattr(
                node,
                "param_type",
                getattr(node, "var_type", getattr(node, "vtype", None)),
            )
            type_name = self.type_name_string(raw_type)
            if type_name not in {None, "", "auto"}:
                return raw_type
            return expression_source_type(
                getattr(node, "initial_value", None),
                visible,
            )

        def for_in_binding_source_type(node, visible):
            iterable = getattr(node, "iterable", None)
            if isinstance(iterable, RangeNode):
                return "int"
            iterable_type = expression_source_type(iterable, visible)
            if self.is_scalar_integer_type(iterable_type):
                return "int"
            outer_array = self.hlsl_for_in_outer_array_type(
                self.type_name_string(iterable_type)
            )
            if outer_array is not None:
                _extent, element_type = outer_array
                return element_type
            return getattr(
                node,
                "binding_type",
                getattr(node, "pattern_type", None),
            )

        def register(name, node, source_type, kind):
            binding = {
                "id": id(node),
                "kind": kind,
                "name": name,
                "node": node,
                "source_type": source_type,
            }
            declarations[id(node)] = binding
            return binding

        def visit_sequence(value, visible):
            if value is None:
                return
            statements = getattr(value, "statements", value)
            if not isinstance(statements, (list, tuple)):
                statements = [statements]
            for statement in statements:
                visit(statement, visible)

        def visit(node, visible):
            if node is None or isinstance(node, (str, int, float, bool)):
                return
            if isinstance(node, dict):
                for child in node.values():
                    visit(child, visible)
                return
            if isinstance(node, (list, tuple, set)):
                for child in node:
                    visit(child, visible)
                return

            if isinstance(node, IdentifierNode):
                binding = visible.get(getattr(node, "name", None))
                if binding is not None:
                    reference_bindings[id(node)] = binding["id"]
                return

            if isinstance(node, (VariableNode, ArrayNode)):
                raw_type = getattr(
                    node,
                    "param_type",
                    getattr(node, "var_type", getattr(node, "vtype", None)),
                )
                type_name = self.type_name_string(raw_type)
                is_declaration = isinstance(node, ArrayNode) or bool(type_name)
                if not is_declaration:
                    binding = visible.get(getattr(node, "name", None))
                    if binding is not None:
                        reference_bindings[id(node)] = binding["id"]
                    return
                name = getattr(node, "name", None)
                if name:
                    visible.pop(name, None)
                initializer_source_types = visible_source_types(visible)
                visit(getattr(node, "initial_value", None), visible)
                if name:
                    visible[name] = register(
                        name,
                        node,
                        declaration_source_type(node, visible),
                        "local",
                    )
                    visible[name]["initializer_source_types"] = initializer_source_types
                return

            if isinstance(node, FunctionCallNode):
                call_source_types[id(node)] = visible_source_types(visible)
                for argument in (
                    getattr(node, "arguments", getattr(node, "args", [])) or []
                ):
                    visit(argument, visible)
                return

            if isinstance(node, BlockNode):
                visit_sequence(node, dict(visible))
                return

            if isinstance(node, IfNode):
                branch_visible = dict(visible)
                condition = getattr(
                    node, "condition", getattr(node, "if_condition", None)
                )
                visit(condition, branch_visible)
                visit_sequence(node.if_body, dict(branch_visible))
                for else_if_condition, else_if_body in zip(
                    getattr(node, "else_if_conditions", []) or [],
                    getattr(node, "else_if_bodies", []) or [],
                ):
                    else_if_visible = dict(visible)
                    visit(else_if_condition, else_if_visible)
                    visit_sequence(else_if_body, else_if_visible)
                visit_sequence(getattr(node, "else_body", None), dict(branch_visible))
                return

            if isinstance(node, ForNode):
                loop_visible = dict(visible)
                visit(getattr(node, "init", None), loop_visible)
                visit(getattr(node, "condition", None), loop_visible)
                visit_sequence(getattr(node, "body", None), dict(loop_visible))
                visit(getattr(node, "update", None), loop_visible)
                return

            if isinstance(node, ForInNode):
                loop_visible = dict(visible)
                visit(getattr(node, "iterable", None), loop_visible)
                pattern = getattr(node, "pattern", None)
                if pattern:
                    loop_visible.pop(pattern, None)
                    loop_visible[pattern] = register(
                        pattern,
                        node,
                        for_in_binding_source_type(node, visible),
                        "for-in",
                    )
                visit_sequence(getattr(node, "body", None), loop_visible)
                return

            if isinstance(node, (WhileNode, DoWhileNode, LoopNode)):
                loop_visible = dict(visible)
                visit(getattr(node, "condition", None), loop_visible)
                visit_sequence(getattr(node, "body", None), loop_visible)
                return

            if isinstance(node, SwitchNode):
                switch_visible = dict(visible)
                visit(getattr(node, "expression", None), switch_visible)
                for case in getattr(node, "cases", []) or []:
                    case_visible = dict(switch_visible)
                    visit(getattr(case, "value", None), case_visible)
                    visit_sequence(getattr(case, "statements", None), case_visible)
                default_case = getattr(node, "default_case", None)
                visit_sequence(
                    getattr(default_case, "statements", default_case),
                    dict(switch_visible),
                )
                return

            if hasattr(node, "child_nodes"):
                for child in node.child_nodes():
                    visit(child, visible)
                return
            if hasattr(node, "__dict__"):
                for field, child in vars(node).items():
                    if field not in {"annotations", "parent", "source_location"}:
                        visit(child, visible)

        visible = {}
        for parameter in (
            getattr(function, "parameters", getattr(function, "params", [])) or []
        ):
            name = getattr(parameter, "name", None)
            if not name:
                continue
            binding = register(
                name,
                parameter,
                declaration_source_type(parameter, visible),
                "parameter",
            )
            visible[name] = binding
            parameter_ids[name] = binding["id"]
        visit_sequence(getattr(function, "body", []), visible)
        return {
            "call_source_types": call_source_types,
            "declarations": declarations,
            "parameter_ids": parameter_ids,
            "reference_bindings": reference_bindings,
        }

    def collect_hlsl_resource_pointer_parameters(self, functions):
        functions = list(functions or [])
        self.hlsl_view_storage_array_parameters = {
            key: (parameter, storage_array_pointer_type(parameter))
            for key, parameter in view_storage_array_parameters(
                functions, self.function_call_name
            ).items()
        }
        parameters_by_function = {}
        indices_by_function = {}
        array_parameters_by_function = {}
        array_indices_by_function = {}
        for function in functions:
            function_name = self.hlsl_function_declaration_name(function)
            if not function_name:
                continue
            parameters = getattr(
                function, "parameters", getattr(function, "params", [])
            )
            pointer_parameters = []
            pointer_indices = {}
            pointer_array_parameters = []
            pointer_array_indices = {}
            for index, parameter in enumerate(parameters):
                array_contract = self.hlsl_resource_pointer_array_parameter_contract(
                    parameter, function_name
                )
                if array_contract is not None:
                    pointer_array_parameters.append(parameter)
                    pointer_array_indices[index] = parameter.name
                    continue
                if not self.hlsl_resource_pointer_parameter(parameter):
                    continue
                pointer_parameters.append(parameter)
                pointer_indices[index] = parameter.name
            if pointer_parameters:
                parameters_by_function[function_name] = pointer_parameters
                indices_by_function[function_name] = pointer_indices
            if pointer_array_parameters:
                array_parameters_by_function[function_name] = pointer_array_parameters
                array_indices_by_function[function_name] = pointer_array_indices

        # Populate every regular/array parameter map before resolving calls. This
        # makes mutation discovery independent of source declaration order and lets
        # overload resolution inspect the exact callee parameter contract.
        self.function_hlsl_resource_pointer_parameters = parameters_by_function
        self.function_hlsl_resource_pointer_parameter_indices = indices_by_function
        self.function_hlsl_resource_pointer_array_parameters = (
            array_parameters_by_function
        )
        self.function_hlsl_resource_pointer_array_parameter_indices = (
            array_indices_by_function
        )

        array_mutations_by_function = {
            function_name: set() for function_name in array_parameters_by_function
        }
        mutation_dependencies = {
            (function_name, parameter.name): set()
            for function_name, parameters in array_parameters_by_function.items()
            for parameter in parameters
        }
        function_has_body = {
            function_name: False for function_name in array_parameters_by_function
        }
        for function in functions:
            function_name = self.hlsl_function_declaration_name(function)
            if function_name not in array_parameters_by_function:
                continue
            function_has_body[function_name] = (
                function_has_body[function_name]
                or getattr(function, "body", None) is not None
            )
            previous_local_types = self.local_variable_types
            previous_local_source_types = self.local_variable_source_types
            previous_call_source_types = (
                self.hlsl_resource_pointer_analysis_call_source_types
            )
            (
                self.local_variable_types,
                self.local_variable_source_types,
            ) = self.hlsl_function_local_type_environment(function)
            lexical = self.hlsl_function_lexical_type_bindings(function)
            self.hlsl_resource_pointer_analysis_call_source_types = lexical[
                "call_source_types"
            ]
            try:
                for parameter in array_parameters_by_function[function_name]:
                    direct_mutation, dependencies = (
                        self.hlsl_resource_pointer_array_parameter_mutation_analysis(
                            function, parameter.name, lexical=lexical
                        )
                    )
                    if direct_mutation:
                        array_mutations_by_function[function_name].add(parameter.name)
                    mutation_dependencies[(function_name, parameter.name)].update(
                        target
                        for target in dependencies
                        if target in mutation_dependencies
                    )
            finally:
                self.local_variable_types = previous_local_types
                self.local_variable_source_types = previous_local_source_types
                self.hlsl_resource_pointer_analysis_call_source_types = (
                    previous_call_source_types
                )

        # A declaration without any definition has unknown effects and therefore
        # remains fail-closed. Definitions, including mutually recursive helpers,
        # are solved by a monotone fixed point over exact overload-resolved call
        # edges. Read-only cycles stay read-only; one concrete mutation propagates
        # through every caller in the cycle regardless of declaration order.
        for function_name, has_body in function_has_body.items():
            if not has_body:
                array_mutations_by_function[function_name].update(
                    parameter.name
                    for parameter in array_parameters_by_function[function_name]
                )
        changed = True
        while changed:
            changed = False
            for (
                function_name,
                parameter_name,
            ), targets in mutation_dependencies.items():
                if parameter_name in array_mutations_by_function[function_name]:
                    continue
                if any(
                    target_parameter in array_mutations_by_function[target_function]
                    for target_function, target_parameter in targets
                ):
                    array_mutations_by_function[function_name].add(parameter_name)
                    changed = True
        self.function_hlsl_resource_pointer_array_mutated_parameters = (
            array_mutations_by_function
        )
        self.collect_hlsl_unobserved_null_resource_pointer_parameters(functions)

    def hlsl_resource_pointer_analysis_call_name(self, call):
        function_name = self.function_call_name(call)
        if not function_name:
            return None
        arguments = list(getattr(call, "arguments", getattr(call, "args", [])) or [])
        if function_name in self.hlsl_mapped_overload_names:
            previous_local_types = self.local_variable_types
            previous_local_source_types = self.local_variable_source_types
            source_types = self.hlsl_resource_pointer_analysis_call_source_types.get(
                id(call), {}
            )
            self.local_variable_types = {
                **previous_local_types,
                **source_types,
            }
            self.local_variable_source_types = {
                **previous_local_source_types,
                **source_types,
            }
            try:
                resolved = self.resolve_hlsl_function_overload(
                    function_name, arguments, call_node=call
                )
            finally:
                self.local_variable_types = previous_local_types
                self.local_variable_source_types = previous_local_source_types
            if resolved is not None:
                return self.hlsl_function_declaration_name(resolved)
        return self.hlsl_function_call_name(function_name)

    def hlsl_resource_pointer_analysis_call_parameter(self, call, argument_index):
        function_name = self.hlsl_resource_pointer_analysis_call_name(call)
        if not function_name:
            return None, None
        return self.hlsl_resource_pointer_parameter_for_call(
            function_name, argument_index
        )

    def hlsl_static_condition_value(self, expression, bool_constants=None):
        value = self.hlsl_bool_constant_value(expression, bool_constants or {})
        if value is not None:
            return bool(value)
        value = self.literal_int_value(expression, self.literal_int_constants)
        return None if value is None else value != 0

    def collect_hlsl_unobserved_null_resource_pointer_parameters(self, functions):
        """Find null-rooted pointer chains that have no reachable observation.

        HLSL resource objects cannot represent ``nullptr``. A parameter is omitted
        only when an explicit null call roots the chain and branch-aware dataflow
        proves that every forwarded parameter is neither dereferenced nor otherwise
        observed. Any unresolved or reachable use remains on the normal fail-closed
        resource-binding path.
        """
        self.hlsl_omitted_resource_pointer_parameter_keys = set()
        functions = list(functions or [])
        all_keys = set()
        function_pointer_keys = {}
        function_has_body = {}

        for function in functions:
            function_name = self.hlsl_function_declaration_name(function)
            if not function_name:
                continue
            keys_by_name = {}
            for parameter in (
                getattr(function, "parameters", getattr(function, "params", [])) or []
            ):
                if not self.hlsl_resource_pointer_parameter(parameter):
                    continue
                key = (function_name, parameter.name)
                keys_by_name[parameter.name] = key
                all_keys.add(key)
            if keys_by_name:
                function_pointer_keys[id(function)] = keys_by_name
                function_has_body[function_name] = (
                    function_has_body.get(function_name, False)
                    or getattr(function, "body", None) is not None
                )

        if not all_keys:
            return

        directly_observed = set()
        forwarded = {key: set() for key in all_keys}
        null_call_targets = set()

        def reference_name(value, pointer_names):
            if isinstance(value, str):
                name = value
            elif isinstance(value, (IdentifierNode, VariableNode)):
                name = getattr(value, "name", None)
            else:
                return None
            return name if name in pointer_names else None

        def analyze_value(value, caller_keys, bool_constants, seen=None):
            if value is None or isinstance(value, (bool, int, float)):
                return
            if seen is None:
                seen = set()
            value_id = id(value)
            if value_id in seen:
                return
            seen.add(value_id)

            pointer_names = set(caller_keys)
            direct_name = reference_name(value, pointer_names)
            if direct_name is not None:
                directly_observed.add(caller_keys[direct_name])
                return
            if isinstance(value, str):
                return
            if isinstance(value, dict):
                for child in value.values():
                    analyze_value(child, caller_keys, bool_constants, seen)
                return
            if isinstance(value, (list, tuple, set)):
                for child in value:
                    analyze_value(child, caller_keys, bool_constants, seen)
                return

            if isinstance(value, FunctionCallNode):
                arguments = list(
                    getattr(value, "arguments", getattr(value, "args", [])) or []
                )
                for index, argument in enumerate(arguments):
                    parameter, target_name = (
                        self.hlsl_resource_pointer_analysis_call_parameter(value, index)
                    )
                    target_key = (
                        (target_name, parameter.name)
                        if parameter is not None and target_name is not None
                        else None
                    )
                    if self.hlsl_null_pointer_expression(argument):
                        if target_key in all_keys:
                            null_call_targets.add(target_key)
                        continue
                    source_name = reference_name(argument, pointer_names)
                    if source_name is not None and target_key in all_keys:
                        forwarded[caller_keys[source_name]].add(target_key)
                        continue
                    analyze_value(argument, caller_keys, bool_constants, seen)
                return

            if isinstance(value, TernaryOpNode):
                condition = getattr(value, "condition", None)
                analyze_value(condition, caller_keys, bool_constants, seen)
                selected = self.hlsl_static_condition_value(condition, bool_constants)
                if selected is None:
                    analyze_value(
                        getattr(value, "true_expr", None),
                        caller_keys,
                        bool_constants,
                        seen,
                    )
                    analyze_value(
                        getattr(value, "false_expr", None),
                        caller_keys,
                        bool_constants,
                        seen,
                    )
                else:
                    analyze_value(
                        (
                            getattr(value, "true_expr", None)
                            if selected
                            else getattr(value, "false_expr", None)
                        ),
                        caller_keys,
                        bool_constants,
                        seen,
                    )
                return

            for field_name, child in vars(value).items():
                if field_name in {
                    "annotations",
                    "parent",
                    "source_location",
                    "function",
                }:
                    continue
                analyze_value(child, caller_keys, bool_constants, seen)

        def analyze_if(statement, caller_keys, bool_constants):
            conditions = [
                getattr(
                    statement,
                    "condition",
                    getattr(statement, "if_condition", None),
                ),
                *(getattr(statement, "else_if_conditions", []) or []),
            ]
            bodies = [
                getattr(
                    statement,
                    "then_branch",
                    getattr(statement, "if_body", None),
                ),
                *(getattr(statement, "else_if_bodies", []) or []),
            ]
            else_body = getattr(
                statement,
                "else_branch",
                getattr(statement, "else_body", None),
            )

            for index, condition in enumerate(conditions):
                analyze_value(condition, caller_keys, bool_constants)
                selected = self.hlsl_static_condition_value(condition, bool_constants)
                if selected is False:
                    continue
                if selected is True:
                    analyze_body(bodies[index], caller_keys, dict(bool_constants))
                    return

                analyze_body(bodies[index], caller_keys, dict(bool_constants))
                for later_condition, later_body in zip(
                    conditions[index + 1 :], bodies[index + 1 :]
                ):
                    analyze_value(later_condition, caller_keys, bool_constants)
                    analyze_body(later_body, caller_keys, dict(bool_constants))
                analyze_body(else_body, caller_keys, dict(bool_constants))
                return

            analyze_body(else_body, caller_keys, dict(bool_constants))

        def analyze_body(body, caller_keys, bool_constants):
            for statement in self.hlsl_statement_body_items(body):
                if isinstance(statement, BlockNode):
                    analyze_body(statement, caller_keys, dict(bool_constants))
                    continue
                if isinstance(statement, IfNode):
                    analyze_if(statement, caller_keys, bool_constants)
                    continue
                analyze_value(statement, caller_keys, bool_constants)
                self.hlsl_update_visible_bool_constants(statement, bool_constants)

        for function in functions:
            caller_keys = function_pointer_keys.get(id(function), {})
            if getattr(function, "body", None) is None:
                continue
            previous_local_types = self.local_variable_types
            previous_local_source_types = self.local_variable_source_types
            previous_call_source_types = (
                self.hlsl_resource_pointer_analysis_call_source_types
            )
            (
                self.local_variable_types,
                self.local_variable_source_types,
            ) = self.hlsl_function_local_type_environment(function)
            lexical = self.hlsl_function_lexical_type_bindings(function)
            self.hlsl_resource_pointer_analysis_call_source_types = lexical[
                "call_source_types"
            ]
            try:
                analyze_body(
                    function.body,
                    caller_keys,
                    self.hlsl_initial_bool_constants(function),
                )
            finally:
                self.local_variable_types = previous_local_types
                self.local_variable_source_types = previous_local_source_types
                self.hlsl_resource_pointer_analysis_call_source_types = (
                    previous_call_source_types
                )

        for key in all_keys:
            if not function_has_body.get(key[0], False):
                directly_observed.add(key)

        observed = set(directly_observed)
        changed = True
        while changed:
            changed = False
            for source, targets in forwarded.items():
                if source in observed:
                    continue
                if any(
                    target in observed or target not in all_keys for target in targets
                ):
                    observed.add(source)
                    changed = True

        omitted = set()
        pending = [key for key in null_call_targets if key not in observed]
        while pending:
            key = pending.pop()
            if key in omitted or key in observed:
                continue
            omitted.add(key)
            pending.extend(
                target
                for target in forwarded.get(key, ())
                if target in all_keys and target not in observed
            )
        self.hlsl_omitted_resource_pointer_parameter_keys = omitted

    def hlsl_resource_pointer_parameter_is_omitted(self, function, parameter):
        function_names = {
            getattr(function, "name", None),
            self.hlsl_function_declaration_name(function),
        }
        return any(
            (name, getattr(parameter, "name", None))
            in self.hlsl_omitted_resource_pointer_parameter_keys
            for name in function_names
            if name
        )

    def hlsl_resource_pointer_call_parameter_is_omitted(
        self, function_name, argument_index
    ):
        parameter, target_name = self.hlsl_resource_pointer_parameter_for_call(
            function_name, argument_index
        )
        return bool(
            parameter is not None
            and (
                target_name,
                getattr(parameter, "name", None),
            )
            in self.hlsl_omitted_resource_pointer_parameter_keys
        )
