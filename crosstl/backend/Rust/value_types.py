"""Type inference for Rust values, calls, generic substitutions and Option payloads."""

import re

from .import_tables import RUST_NUMERIC_LITERAL_RE
from .RustAst import (
    ArrayAccessNode,
    ArrayNode,
    BinaryOpNode,
    CastNode,
    FunctionCallNode,
    FunctionNode,
    IfNode,
    LabeledBlockNode,
    MatchNode,
    MemberAccessNode,
    ReturnNode,
    StructInitializationNode,
    StructNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)

# The element type of the structured buffer a storage-buffer slice parameter imports as.
STRUCTURED_BUFFER_ELEMENT_RE = re.compile(r"^(?:RW)?StructuredBuffer<(.+)>$")


class RustValueTypeMixin:
    """Value, call-result and member types of ``RustToCrossGLConverter``."""

    def option_payload_type_name(self, type_name):
        generic = self.parse_generic_type(type_name)
        if generic is None:
            return None

        base_name, args = generic
        if base_name.rsplit("::", 1)[-1] != "Option" or len(args) != 1:
            return None
        return args[0]

    def lookup_option_payload_return_type(self, function_name):
        if not isinstance(function_name, str):
            return None

        base_name, _ = self.split_function_type_arguments(function_name)
        if "::" in base_name:
            base_name = base_name.rsplit("::", 1)[-1]
        return_type = self.lookup_user_function_return_type(base_name)
        return self.option_payload_type_name(return_type)

    def option_unwrap_or_helper_name(self, function_name):
        base_name, _ = self.split_function_type_arguments(function_name)
        base_name = base_name.rsplit("::", 1)[-1]
        return self.crossgl_identifier(
            f"{base_name}_unwrap_or",
            forbidden=set(self.user_function_names),
        )

    def ensure_option_unwrap_or_helper(self, function_name):
        payload_type = self.lookup_option_payload_return_type(function_name)
        if payload_type is None:
            return None

        base_name, _ = self.split_function_type_arguments(function_name)
        base_name = base_name.rsplit("::", 1)[-1]
        source_func = self.function_definitions.get(base_name)
        if source_func is None:
            return None
        if getattr(source_func, "generics", None):
            return None

        helper_name = self.option_unwrap_or_helper_name(base_name)
        if helper_name not in self.option_unwrap_or_helpers:
            self.option_unwrap_or_helpers[helper_name] = {
                "function": source_func,
                "payload_type": payload_type,
                "fallback_name": self.crossgl_identifier("_rust_option_fallback"),
            }
        return helper_name

    def generate_option_unwrap_or_helpers(self, indent=1):
        code = ""
        generated = set()
        while True:
            pending = [
                (name, info)
                for name, info in self.option_unwrap_or_helpers.items()
                if name not in generated
            ]
            if not pending:
                break

            for helper_name, helper in pending:
                code += self.generate_option_unwrap_or_helper(
                    helper_name,
                    helper,
                    indent,
                )
                generated.add(helper_name)
        return code

    def generate_option_unwrap_or_helper(self, helper_name, helper, indent=1):
        source_func = helper["function"]
        payload_type = helper["payload_type"]
        fallback_name = helper["fallback_name"]
        fallback_param = VariableNode(payload_type, fallback_name)
        helper_func = FunctionNode(
            payload_type,
            helper_name,
            list(getattr(source_func, "params", []) or []) + [fallback_param],
            source_func.body,
            attributes=[],
            visibility=getattr(source_func, "visibility", None),
            generics=[],
            where_clauses=[],
            is_async=getattr(source_func, "is_async", False),
            is_unsafe=getattr(source_func, "is_unsafe", False),
            abi=getattr(source_func, "abi", None),
            is_const=getattr(source_func, "is_const", False),
        )

        previous_fallback_name = self.current_option_unwrap_fallback_name
        self.current_option_unwrap_fallback_name = fallback_name
        try:
            return self.generate_function(helper_func, indent=indent)
        finally:
            self.current_option_unwrap_fallback_name = previous_fallback_name

    def collect_function_type_signatures(self, ast):
        signatures = {}
        for func in getattr(ast, "functions", []):
            name = getattr(func, "name", None)
            return_type = getattr(func, "return_type", None)
            if not name or not return_type:
                continue

            params = []
            for param in getattr(func, "params", []):
                param_type = getattr(param, "vtype", None)
                if param_type:
                    params.append((getattr(param, "name", None), param_type))

            signatures[name] = {
                "return_type": return_type,
                "params": params,
                "generic_names": [
                    self.generic_parameter_name(generic)
                    for generic in getattr(func, "generics", []) or []
                ],
            }
        return signatures

    def collect_impl_method_signatures(self, ast):
        signatures_by_type = {}
        for impl_block in getattr(ast, "impl_blocks", []):
            struct_name = getattr(impl_block, "struct_name", None)
            if not struct_name:
                continue

            signature = signatures_by_type.setdefault(
                struct_name,
                {
                    "struct_name": struct_name,
                    "call_prefix": self.impl_function_prefix(struct_name),
                    "generic_names": [
                        self.generic_parameter_name(generic)
                        for generic in getattr(impl_block, "generics", []) or []
                    ],
                    "methods": {},
                },
            )

            for func in getattr(impl_block, "functions", []):
                method_name = getattr(func, "name", None)
                return_type = getattr(func, "return_type", None)
                params = []
                for param in getattr(func, "params", []):
                    param_type = getattr(param, "vtype", None)
                    if param_type:
                        params.append((getattr(param, "name", None), param_type))

                if method_name and return_type:
                    signature["methods"][method_name] = {
                        "return_type": return_type,
                        "params": params,
                        "generic_names": [
                            self.generic_parameter_name(generic)
                            for generic in getattr(func, "generics", []) or []
                        ],
                    }
        return signatures_by_type

    def collect_impl_method_name_sets(self):
        receiver_methods = set()
        associated_functions = set()

        for signature in self.impl_method_signatures.values():
            for method_name, method in signature["methods"].items():
                if self.method_has_self_parameter(method):
                    receiver_methods.add(method_name)
                else:
                    associated_functions.add(method_name)

        return receiver_methods, associated_functions

    def collect_struct_member_types(self, ast):
        member_types = {}
        for struct in getattr(ast, "structs", []):
            if not isinstance(struct, StructNode):
                continue

            members = {}
            for member in getattr(struct, "members", []):
                name = getattr(member, "name", None)
                type_name = getattr(member, "vtype", None)
                if name and type_name:
                    members[name] = type_name
            member_types[struct.name] = members

        return member_types

    def collect_struct_generics(self, ast):
        generics = {}
        for struct in getattr(ast, "structs", []):
            if not isinstance(struct, StructNode):
                continue
            generics[struct.name] = [
                self.generic_parameter_name(generic)
                for generic in getattr(struct, "generics", []) or []
            ]
        return generics

    def is_user_defined_function(self, func_name):
        return isinstance(func_name, str) and func_name in self.user_function_names

    def push_value_type_scope(self, entries=None):
        scope = {}
        if entries:
            for name, type_name in entries:
                if name and type_name:
                    scope[name] = type_name
        self.value_type_scopes.append(scope)

    def pop_value_type_scope(self):
        if self.value_type_scopes:
            self.value_type_scopes.pop()

    def add_value_type(self, name, type_name, struct_name=None):
        normalized_type = self.normalize_receiver_type(type_name, struct_name)
        if name and normalized_type and self.value_type_scopes:
            self.value_type_scopes[-1][name] = normalized_type

    def lookup_value_type(self, name):
        direct_type = self.lookup_direct_value_type(name)
        if direct_type:
            return direct_type

        if isinstance(name, str):
            return self.lookup_expression_value_type(name)

        return None

    def lookup_direct_value_type(self, name):
        for scope in reversed(self.value_type_scopes):
            type_name = scope.get(name)
            if type_name:
                return type_name
        return None

    def normalize_receiver_type(self, type_name, struct_name=None):
        if not type_name:
            return None

        type_name = self.strip_reference_type(type_name)
        if type_name == "Self" and struct_name:
            if struct_name == "()":
                return "Unit"
            return struct_name
        return type_name

    def normalize_parameter_type(self, type_name, struct_name=None, attributes=None):
        typed_buffer_type = self.map_typed_buffer_parameter_type(type_name)
        if typed_buffer_type is not None:
            return typed_buffer_type
        byte_addressable_type = self.map_byte_addressable_buffer_parameter_type(
            type_name
        )
        if byte_addressable_type is not None:
            return byte_addressable_type
        storage_buffer_type = self.map_rust_gpu_storage_buffer_parameter_type(
            type_name,
            attributes,
        )
        if storage_buffer_type is not None:
            return storage_buffer_type
        tuple_type = self.map_tuple_type(type_name)
        if tuple_type is not None:
            return tuple_type
        return self.normalize_receiver_type(type_name, struct_name)

    def infer_value_type(self, expression):
        if isinstance(expression, StructInitializationNode):
            return expression.struct_name
        if isinstance(expression, LabeledBlockNode):
            return self.infer_block_expression_value_type(expression.body)
        if isinstance(expression, ArrayNode):
            return self.infer_array_value_type(expression)
        if isinstance(expression, BinaryOpNode):
            return self.infer_binary_expression_value_type(expression)
        if isinstance(expression, UnaryOpNode):
            return self.infer_value_type(expression.operand)
        if isinstance(expression, CastNode):
            target_type = getattr(expression, "target_type", None)
            return target_type if isinstance(target_type, str) else None
        if isinstance(expression, FunctionCallNode):
            return self.infer_function_call_return_type(expression)
        if isinstance(expression, TernaryOpNode):
            return self.common_inferred_value_type(
                [
                    self.infer_value_type(expression.true_expr),
                    self.infer_value_type(expression.false_expr),
                ]
            )
        if isinstance(expression, IfNode):
            return self.common_inferred_value_type(
                [
                    self.infer_branch_body_value_type(expression.if_body),
                    self.infer_branch_body_value_type(expression.else_body),
                ]
            )
        if isinstance(expression, MatchNode):
            return self.common_inferred_value_type(
                [
                    self.infer_branch_body_value_type(arm.body)
                    for arm in getattr(expression, "arms", [])
                ]
            )
        if self.is_block_expression_node(expression):
            return self.infer_block_expression_value_type(expression)
        if isinstance(expression, str):
            if expression.strip() in {"true", "false"}:
                return "bool"
            literal_type = self.infer_numeric_literal_type(expression)
            if literal_type is not None:
                return literal_type
            path_type = self.infer_path_value_type(expression)
            if path_type is not None:
                return path_type
        if isinstance(
            expression, (str, VariableNode, MemberAccessNode, ArrayAccessNode)
        ):
            return self.lookup_value_type(self.generate_expression(expression))
        return None

    def infer_array_value_type(self, expression):
        array_size = self.infer_array_literal_size(expression)
        if array_size is None:
            return None

        element_types = [
            self.infer_value_type(element)
            for element in getattr(expression, "elements", []) or []
        ]
        element_type = self.common_array_element_type(element_types)
        if element_type is None:
            return None
        return self.outer_array_type(element_type, array_size)

    def outer_array_type(self, element_type, array_size):
        """The type of an array of ``array_size`` ``element_type`` values, the outer dimension first.

        ``[[f32; 3]; 2]`` is two arrays of three floats, which CrossGL, like C, writes ``float[2][3]``.
        """
        array_parts = self.split_array_type(element_type)
        if array_parts is not None:
            base_type, inner_suffix = array_parts
            return f"{base_type}[{array_size}]{inner_suffix}"
        return f"{element_type}[{array_size}]"

    def infer_array_literal_size(self, expression):
        if getattr(expression, "size", None) is not None:
            size = self.generate_expression(expression.size)
            if size:
                return str(size).strip()
            return None

        elements = getattr(expression, "elements", []) or []
        if elements:
            return str(len(elements))
        return None

    def infer_numeric_literal_type(self, expression):
        if not isinstance(expression, str):
            return None

        match = RUST_NUMERIC_LITERAL_RE.match(expression.strip())
        if match is None:
            return None

        body = match.group("body") or ""
        suffix = (match.group("suffix") or "").lstrip("_").lower()
        if suffix in {"f32"}:
            return "float"
        if suffix in {"f64"}:
            return "double"
        if suffix.startswith("u"):
            return "uint"
        if suffix.startswith("i"):
            return "int"
        if "." in body or "e" in body.lower():
            return "float"
        return "int"

    def common_array_element_type(self, element_types):
        if not element_types or any(not element_type for element_type in element_types):
            return None

        mapped_types = [self.map_type(element_type) for element_type in element_types]
        if any(element_type == "double" for element_type in mapped_types):
            return "double"
        if any(element_type == "float" for element_type in mapped_types):
            return "float"
        if all(element_type == "uint" for element_type in mapped_types):
            return "uint"
        if all(element_type in {"int", "uint"} for element_type in mapped_types):
            return "int"
        first_type = mapped_types[0]
        if all(element_type == first_type for element_type in mapped_types):
            return first_type
        return None

    def inferred_array_declarator(self, type_name, target_name):
        """A typed declarator for a ``let`` whose inferred type is an array or a non-generic struct.

        CrossGL's untyped ``let`` leaves the type to each generator, which cannot see through an initializer list,
        and the shared aggregate passes (equality, copies) need a struct operand's type.
        """
        if not type_name:
            return None

        if self.split_array_type(self.map_type(type_name)) is None:
            resolved = self.resolve_struct_type(self.normalize_receiver_type(type_name))
            if resolved is None or resolved[1]:
                return None
        return self.format_typed_declarator(type_name, target_name)

    def infer_binary_expression_value_type(self, expression):
        if expression.op in {"==", "!=", "<", ">", "<=", ">=", "&&", "||"}:
            return "bool"

        left_type = self.infer_value_type(expression.left)
        right_type = self.infer_value_type(expression.right)
        if left_type and right_type:
            return self.common_inferred_value_type([left_type, right_type])
        return left_type or right_type

    def infer_path_value_type(self, expression):
        if not isinstance(expression, str) or "::" not in expression:
            return None

        path = self.resolve_imported_module_path(expression)
        type_name = path.rsplit("::", 1)[0]
        mapped_type = self.map_type(type_name)
        if mapped_type == type_name:
            return None
        return mapped_type

    def infer_branch_body_value_type(self, body):
        if body is None:
            return None
        if isinstance(body, ReturnNode):
            return self.infer_value_type(body.value)
        if isinstance(body, list):
            if not body:
                return None
            return self.infer_branch_body_value_type(body[-1])
        return self.infer_value_type(body)

    def infer_block_expression_value_type(self, expression):
        block_node = self.get_block_expression_node(expression)
        if block_node is None:
            return None

        block_expression = self.get_block_expression(block_node)
        if block_expression is not None:
            return self.infer_value_type(block_expression)

        statements = getattr(block_node, "statements", []) or []
        if not statements:
            return None
        return self.infer_branch_body_value_type(statements[-1])

    def common_inferred_value_type(self, type_names):
        if not type_names or any(not type_name for type_name in type_names):
            return None

        first_type = self.normalize_receiver_type(type_names[0])
        if all(
            self.normalize_receiver_type(type_name) == first_type
            for type_name in type_names
        ):
            return first_type

        first_resource_type = self.map_resource_receiver_type(first_type)
        if not isinstance(
            first_resource_type, str
        ) or not first_resource_type.startswith(self.RESOURCE_TYPE_PREFIXES):
            return None

        for type_name in type_names[1:]:
            resource_type = self.map_resource_receiver_type(type_name)
            if resource_type != first_resource_type:
                return None
        return first_type

    def infer_function_call_return_type(self, expression):
        if isinstance(expression.name, MemberAccessNode):
            return self.infer_impl_method_call_return_type(expression)

        raw_function_name = self.function_call_name(expression.name)
        if raw_function_name is None:
            return None
        builtin_return_type = self.infer_builtin_function_return_type(
            raw_function_name,
            expression.args,
        )
        if builtin_return_type is not None:
            return builtin_return_type
        associated_return_type = self.infer_impl_associated_function_call_return_type(
            raw_function_name,
            expression.args,
        )
        if associated_return_type is not None:
            return associated_return_type
        default_value_type = self.default_value_type(raw_function_name, expression.args)
        if default_value_type is not None:
            return default_value_type

        function_name, explicit_type_args = self.split_function_type_arguments(
            raw_function_name
        )
        signature = self.lookup_user_function_signature(function_name)
        if signature is not None:
            return self.infer_user_function_call_return_type(
                signature,
                expression.args,
                explicit_type_args,
            )
        return self.lookup_user_function_return_type(function_name)

    def infer_impl_associated_function_call_return_type(self, function_name, args):
        match = self.lookup_impl_associated_function_match(function_name)
        if match is None:
            return None

        return self.infer_impl_associated_function_return_type(match, args)

    def infer_impl_method_call_return_type(self, expression):
        member_access = expression.name
        method_name = getattr(member_access, "member", None)
        builtin_return_type = self.infer_builtin_method_return_type(
            method_name,
            self.infer_value_type(member_access.object),
            expression.args,
        )
        if builtin_return_type is not None:
            return builtin_return_type

        obj = self.generate_expression(member_access.object)
        match = self.lookup_impl_method_receiver_match(obj, method_name)
        if match is None:
            return None

        return self.infer_impl_method_return_type(match, expression.args)

    def infer_builtin_function_return_type(self, function_name, args=None):
        if not isinstance(function_name, str):
            return None

        function_name = self.resolve_imported_module_path(function_name)
        primitive_bitcast_return_type = (
            self.infer_primitive_float_bitcast_associated_return_type(
                function_name,
                args,
            )
        )
        if primitive_bitcast_return_type is not None:
            return primitive_bitcast_return_type

        if "::" in function_name:
            type_name, method_name = function_name.rsplit("::", 1)
            if method_name in {"new", "splat", "from"}:
                mapped_type = self.map_type(type_name)
                if mapped_type != type_name:
                    return mapped_type

        return self.VECTOR_CONSTRUCTOR_RETURN_TYPES.get(
            function_name.rsplit("::", 1)[-1]
        )

    def infer_builtin_method_return_type(self, method_name, receiver_type, args):
        if method_name == "extend" and len(args) == 1:
            return self.extended_vector_constructor(receiver_type)
        if method_name == "truncate" and not args:
            return self.truncated_vector_constructor(receiver_type)
        if method_name in {"max_element", "min_element"} and not args:
            return self.vector_element_scalar_type(receiver_type)
        if method_name == "to_bits" and not args:
            bitcast_info = self.primitive_float_bitcast_info(receiver_type)
            if bitcast_info is not None:
                return bitcast_info["bits_type"]
        return None

    def infer_impl_method_return_type(self, match, arg_values):
        method = match["method"]
        return_type = match["method"].get("return_type")
        return_type = self.normalize_receiver_type(return_type, match["receiver_type"])
        substitutions = dict(match.get("substitutions") or {})
        generic_names = list(match.get("generic_names") or [])
        for generic_name in method.get("generic_names") or []:
            if generic_name not in generic_names:
                generic_names.append(generic_name)

        conflicts = set()
        for (_, param_type), arg in zip(
            self.method_value_params(method),
            arg_values,
        ):
            arg_type = self.infer_call_argument_type(arg)
            if arg_type is None:
                continue
            param_type = self.normalize_receiver_type(
                param_type,
                match["receiver_type"],
            )
            self.match_generic_type_parameters(
                param_type,
                arg_type,
                generic_names,
                substitutions,
                conflicts,
            )

        for generic_name in conflicts:
            substitutions.pop(generic_name, None)
        if substitutions:
            return_type = self.apply_type_substitutions(return_type, substitutions)
        return return_type

    def method_value_params(self, method):
        return [
            (name, param_type)
            for name, param_type in method.get("params", [])
            if not self.is_self_parameter(name, param_type)
        ]

    def method_has_self_parameter(self, method):
        return any(
            self.is_self_parameter(name, param_type)
            for name, param_type in method.get("params", [])
        )

    def is_self_parameter(self, name, param_type):
        return name == "self"

    def infer_impl_associated_function_return_type(self, match, arg_values):
        method = match["method"]
        return_type = method.get("return_type")
        return_type = self.normalize_receiver_type(return_type, match["receiver_type"])
        substitutions = dict(match.get("substitutions") or {})
        generic_names = list(match.get("generic_names") or [])
        method_generic_names = method.get("generic_names") or []
        for generic_name in method.get("generic_names") or []:
            if generic_name not in generic_names:
                generic_names.append(generic_name)

        conflicts = set()
        for generic_name, type_arg in zip(
            method_generic_names,
            match.get("method_type_args") or [],
        ):
            self.bind_generic_type_substitution(
                generic_name,
                type_arg,
                substitutions,
                conflicts,
            )

        for (_, param_type), arg in zip(
            self.method_value_params(method),
            arg_values,
        ):
            arg_type = self.infer_call_argument_type(arg)
            if arg_type is None:
                continue
            self.match_generic_type_parameters(
                param_type,
                arg_type,
                generic_names,
                substitutions,
                conflicts,
            )

        for generic_name in conflicts:
            substitutions.pop(generic_name, None)
        if substitutions:
            return_type = self.apply_type_substitutions(return_type, substitutions)
        return return_type

    def function_call_name(self, name):
        if isinstance(name, str):
            return name
        if isinstance(name, VariableNode):
            return name.name
        return None

    def split_function_type_arguments(self, function_name):
        if not isinstance(function_name, str):
            return function_name, []

        function_name = self.normalize_associated_type_path(function_name)
        generic = self.parse_generic_type(function_name)
        if generic is None:
            return function_name, []
        return generic

    def format_function_type_arguments(self, function_name, type_args):
        if not type_args:
            return function_name

        normalized_args = [
            self.normalize_unmapped_function_generic_argument(arg) for arg in type_args
        ]
        normalized_args = [arg for arg in normalized_args if arg]
        if not normalized_args:
            return function_name
        return f"{function_name}<{', '.join(normalized_args)}>"

    def normalize_unmapped_function_generic_argument(self, arg):
        normalized = self.normalize_lifetime_generic_argument(arg)
        function_pointer_type = self.map_function_pointer_type(normalized)
        if function_pointer_type is not None:
            return function_pointer_type

        raw_pointer_type = self.strip_raw_pointer_type(normalized)
        if raw_pointer_type != normalized:
            return f"ptr<{self.map_type(raw_pointer_type)}>"
        return normalized

    def lookup_user_function_signature(self, function_name):
        if not isinstance(function_name, str):
            return None

        function_name = self.resolve_imported_module_path(function_name)
        signature = self.function_type_signatures.get(function_name)
        if signature is not None:
            return signature

        current_module_function = self.current_module_user_function_name(function_name)
        if current_module_function is None:
            return None
        return self.function_type_signatures.get(current_module_function)

    def infer_user_function_call_return_type(
        self,
        signature,
        args,
        explicit_type_args=None,
    ):
        return_type = signature.get("return_type")
        if not return_type:
            return None

        generic_names = signature.get("generic_names") or []
        if not generic_names:
            return return_type

        substitutions = {}
        conflicts = set()
        for generic_name, type_arg in zip(generic_names, explicit_type_args or []):
            self.bind_generic_type_substitution(
                generic_name,
                type_arg,
                substitutions,
                conflicts,
            )

        for (_, param_type), arg in zip(signature.get("params", []), args):
            arg_type = self.infer_call_argument_type(arg)
            if arg_type is None:
                continue
            self.match_generic_type_parameters(
                param_type,
                arg_type,
                generic_names,
                substitutions,
                conflicts,
            )

        for generic_name in conflicts:
            substitutions.pop(generic_name, None)
        if substitutions:
            return self.apply_type_substitutions(return_type, substitutions)
        return return_type

    def infer_call_argument_type(self, arg):
        if isinstance(arg, str):
            return self.lookup_value_type(arg)
        inferred_type = self.infer_value_type(arg)
        if inferred_type is not None:
            return inferred_type
        return self.lookup_value_type(self.generate_expression(arg))

    def match_generic_type_parameters(
        self,
        pattern_type,
        actual_type,
        generic_names,
        substitutions,
        conflicts,
    ):
        pattern_type = self.normalize_receiver_type(pattern_type)
        actual_type = self.normalize_receiver_type(actual_type)
        if not pattern_type or not actual_type:
            return

        if pattern_type in generic_names:
            self.bind_generic_type_substitution(
                pattern_type,
                actual_type,
                substitutions,
                conflicts,
            )
            return

        pattern_array = self.split_array_type(pattern_type)
        actual_array = self.split_array_type(actual_type)
        if pattern_array and actual_array:
            self.match_generic_type_parameters(
                pattern_array[0],
                actual_array[0],
                generic_names,
                substitutions,
                conflicts,
            )
            return

        for pattern_candidate in self.generic_type_match_candidates(pattern_type):
            pattern_generic = self.parse_generic_type(pattern_candidate)
            if pattern_generic is None:
                continue
            pattern_base, pattern_args = pattern_generic

            for actual_candidate in self.generic_type_match_candidates(actual_type):
                actual_generic = self.parse_generic_type(actual_candidate)
                if actual_generic is None:
                    continue
                actual_base, actual_args = actual_generic
                if not self.generic_type_bases_match(pattern_base, actual_base):
                    continue
                if len(pattern_args) != len(actual_args):
                    continue

                for pattern_arg, actual_arg in zip(pattern_args, actual_args):
                    self.match_generic_type_parameters(
                        pattern_arg,
                        actual_arg,
                        generic_names,
                        substitutions,
                        conflicts,
                    )
                return

    def bind_generic_type_substitution(
        self,
        generic_name,
        replacement,
        substitutions,
        conflicts,
    ):
        replacement = self.normalize_receiver_type(replacement)
        if not generic_name or not replacement or generic_name in conflicts:
            return

        existing = substitutions.get(generic_name)
        if existing is None or self.inferred_types_equivalent(existing, replacement):
            substitutions[generic_name] = replacement
            return

        conflicts.add(generic_name)
        substitutions.pop(generic_name, None)

    def generic_type_match_candidates(self, type_name):
        candidates = []

        def add_candidate(candidate):
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        add_candidate(type_name)
        resolved = self.resolve_imported_module_path(type_name)
        add_candidate(resolved)
        alias_target = self.resolve_type_alias_target(type_name)
        add_candidate(alias_target)
        if alias_target is not None:
            add_candidate(self.resolve_imported_module_path(alias_target))
        return candidates

    def generic_type_bases_match(self, left, right):
        return any(
            left_name == right_name
            for left_name in self.type_lookup_names(left)
            for right_name in self.type_lookup_names(right)
        )

    def inferred_types_equivalent(self, left, right):
        if left == right:
            return True
        left_resource = self.map_resource_receiver_type(left)
        right_resource = self.map_resource_receiver_type(right)
        if left_resource == right_resource:
            return True
        return self.map_type(left) == self.map_type(right)

    def lookup_user_function_return_type(self, function_name):
        if not isinstance(function_name, str):
            return None

        function_name = self.resolve_imported_module_path(function_name)
        return_type = self.function_return_types.get(function_name)
        if return_type is not None:
            return return_type

        current_module_function = self.current_module_user_function_name(function_name)
        if current_module_function is None:
            return None
        return self.function_return_types.get(current_module_function)

    def lookup_expression_value_type(self, expression):
        parts = self.split_member_expression(expression)
        if not parts:
            return None

        current_type = self.lookup_indexed_expression_part_type(parts[0])
        for member_part in parts[1:]:
            if current_type is None:
                return None

            member_name, index_count = self.split_indexed_expression_part(member_part)
            if not member_name:
                return None

            current_type = self.lookup_struct_member_type(current_type, member_name)
            for _ in range(index_count):
                current_type = self.array_element_type(current_type)
                if current_type is None:
                    return None

        return current_type

    def lookup_indexed_expression_part_type(self, expression):
        root, index_count = self.split_indexed_expression_part(expression)
        if not root:
            return None

        if len(self.split_member_expression(root)) > 1:
            current_type = self.lookup_expression_value_type(root)
        else:
            current_type = self.lookup_direct_value_type(root)
            if current_type is None:
                current_type = self.infer_generated_call_return_type(root)

        for _ in range(index_count):
            current_type = self.array_element_type(current_type)
            if current_type is None:
                return None

        return current_type

    def infer_generated_call_return_type(self, expression):
        parsed = self.parse_generated_call_expression(expression)
        if parsed is None:
            return None

        function_name, args = parsed
        return_type = self.infer_generated_impl_method_return_type(
            function_name,
            args,
        )
        if return_type is not None:
            return return_type

        return_type = self.infer_generated_impl_associated_function_return_type(
            function_name,
            args,
        )
        if return_type is not None:
            return return_type

        function_name, explicit_type_args = self.split_function_type_arguments(
            function_name
        )
        signature = self.lookup_user_function_signature(function_name)
        if signature is not None:
            return self.infer_user_function_call_return_type(
                signature,
                args,
                explicit_type_args,
            )
        return self.lookup_user_function_return_type(function_name)

    def parse_generated_call_expression(self, expression):
        if not isinstance(expression, str) or not expression.endswith(")"):
            return None

        open_index = self.find_matching_call_open(expression)
        if open_index is None or open_index == 0:
            return None

        function_name = expression[:open_index].strip()
        if not function_name:
            return None

        args_text = expression[open_index + 1 : -1]
        args = self.split_generic_arguments(args_text)
        return function_name, args

    def find_matching_call_open(self, expression):
        depth = 0
        for index in range(len(expression) - 1, -1, -1):
            char = expression[index]
            if char == ")":
                depth += 1
            elif char == "(":
                depth -= 1
                if depth == 0:
                    return index
        return None

    def infer_generated_impl_method_return_type(self, function_name, args):
        if not args:
            return None

        for signature in self.impl_method_signatures.values():
            for method_name, method in signature["methods"].items():
                if function_name != f"{signature['call_prefix']}_{method_name}":
                    continue

                match = self.lookup_impl_method_receiver_match(args[0], method_name)
                if match is None:
                    return None
                return self.infer_impl_method_return_type(match, args[1:])

        return None

    def infer_generated_impl_associated_function_return_type(self, function_name, args):
        for signature in self.impl_method_signatures.values():
            for method_name, method in signature["methods"].items():
                if function_name != f"{signature['call_prefix']}_{method_name}":
                    continue
                if self.method_has_self_parameter(method):
                    continue

                match = {
                    "receiver_type": signature["struct_name"],
                    "call_prefix": signature["call_prefix"],
                    "method_name": method_name,
                    "generic_names": signature["generic_names"],
                    "method": method,
                    "substitutions": {},
                }
                return self.infer_impl_associated_function_return_type(match, args)

        return None

    def split_member_expression(self, expression):
        if not isinstance(expression, str):
            return []

        parts = []
        current = []
        depth = 0
        for char in expression:
            if char == "." and depth == 0:
                part = "".join(current).strip()
                if not part:
                    return []
                parts.append(part)
                current = []
                continue

            if char in "[({":
                depth += 1
            elif char in "])}":
                depth = max(0, depth - 1)
            current.append(char)

        part = "".join(current).strip()
        if part:
            parts.append(part)
        return parts

    def split_indexed_expression_part(self, expression):
        if not isinstance(expression, str):
            return "", 0

        root = expression.strip()
        index_count = 0
        while root.endswith("]"):
            open_index = self.find_matching_index_open(root)
            if open_index is None or open_index == 0:
                break
            root = root[:open_index].strip()
            index_count += 1

        return root, index_count

    def find_matching_index_open(self, expression):
        depth = 0
        for index in range(len(expression) - 1, -1, -1):
            char = expression[index]
            if char == "]":
                depth += 1
            elif char == "[":
                depth -= 1
                if depth == 0:
                    return index
        return None

    def array_element_type(self, type_name):
        if not type_name:
            return None

        normalized = self.normalize_receiver_type(type_name)
        array_parts = self.split_array_type(normalized)
        if array_parts:
            return array_parts[0]

        mapped = self.map_type(normalized)
        array_parts = self.split_array_type(mapped)
        if array_parts:
            return array_parts[0]

        # A storage-buffer slice parameter, which the importer types as a structured buffer.
        for candidate in (normalized, mapped):
            if isinstance(candidate, str):
                match = STRUCTURED_BUFFER_ELEMENT_RE.match(candidate.strip())
                if match is not None:
                    return match.group(1).strip()
        return None

    def lookup_struct_member_type(self, struct_type, member_name):
        if not struct_type or not member_name:
            return None

        struct_type = self.normalize_receiver_type(struct_type)
        array_parts = self.split_array_type(struct_type)
        if array_parts:
            struct_type = array_parts[0]

        resolved = self.resolve_struct_type(struct_type)
        if resolved is None:
            return None

        struct_name, substitutions = resolved
        members = self.struct_member_types.get(struct_name, {})
        member_type = members.get(member_name)
        if member_type is None and self.is_tuple_field_member(member_name):
            member_type = members.get(self.tuple_field_member_name(member_name))
        if member_type is None:
            return None

        if substitutions:
            member_type = self.apply_type_substitutions(member_type, substitutions)
        return self.normalize_receiver_type(member_type)

    def resolve_member_access_name(self, obj, member_name):
        member_base, type_args = self.split_function_type_arguments(member_name)
        if type_args:
            member_name = self.format_function_type_arguments(member_base, type_args)

        if not self.is_tuple_field_member(member_name):
            return member_name

        receiver_type = self.lookup_value_type(obj)
        if receiver_type is None:
            return member_name

        field_name = self.tuple_field_member_name(member_name)
        tuple_elements = self.split_tuple_type(receiver_type)
        if tuple_elements is not None and int(member_name) < len(tuple_elements):
            return field_name

        generic = self.parse_generic_type(receiver_type)
        if (
            generic is not None
            and generic[0] == "Tuple"
            and int(member_name) < len(generic[1])
        ):
            return field_name

        if self.lookup_struct_member_type(receiver_type, field_name) is None:
            return member_name
        return field_name

    def is_tuple_field_member(self, member_name):
        return isinstance(member_name, str) and member_name.isdigit()

    def tuple_field_member_name(self, member_name):
        return f"field{member_name}"

    def resolve_struct_type(self, type_name, seen=None):
        seen = seen or set()
        generic = self.parse_generic_type(type_name)
        if generic is not None:
            base_name, args = generic
            for candidate in self.type_lookup_names(base_name):
                if candidate in self.struct_member_types:
                    generics = self.struct_generics.get(candidate, [])
                    if len(generics) != len(args):
                        continue
                    substitutions = dict(zip(generics, args))
                    return candidate, substitutions

                if candidate in seen:
                    continue

                alias = self.type_aliases.get(candidate)
                if alias is None:
                    continue

                alias_type = self.apply_alias_type_arguments(alias, args)
                if alias_type is None:
                    continue

                seen.add(candidate)
                resolved = self.resolve_struct_type(alias_type, seen)
                if resolved is not None:
                    return resolved
            return None

        for candidate in self.type_lookup_names(type_name):
            if candidate in self.struct_member_types:
                return candidate, {}

            if candidate in seen:
                continue

            alias = self.type_aliases.get(candidate)
            if alias is not None and not getattr(alias, "generics", None):
                seen.add(candidate)
                resolved = self.resolve_struct_type(alias.alias_type, seen)
                if resolved is not None:
                    return resolved

        return None

    def apply_alias_type_arguments(self, alias, args):
        generics = getattr(alias, "generics", None) or []
        if not generics:
            return alias.alias_type if not args else None
        if len(generics) != len(args):
            return None

        substitutions = {
            self.generic_parameter_name(generic): arg
            for generic, arg in zip(generics, args)
        }
        return self.apply_type_substitutions(alias.alias_type, substitutions)

    def apply_type_substitutions(self, type_name, substitutions):
        result = type_name
        for name, replacement in substitutions.items():
            result = self.replace_type_identifier(result, name, replacement)
        return result
