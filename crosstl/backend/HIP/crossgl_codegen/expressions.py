"""Constructors, lambdas, atomics, allocation, type aliases, member access, device queries, synchronization and inline assembly in the HIP to CrossGL converter."""

import re

from ..HipAst import (
    HipDevicePropertyNode,
    InitializerListNode,
    MemberAccessNode,
    UnaryOpNode,
    VariableNode,
)


class HipExpressionMixin:
    """Expression members of ``HipToCrossGLConverter``."""

    def format_vector_constructor(self, vector_name, args, element_type="f32"):
        if vector_name == "vec1":
            return args[0]
        return f"{vector_name}<{element_type}>({', '.join(args)})"

    def format_hip_dim3_constructor_call(self, function_name, args):
        if isinstance(function_name, str) and function_name.startswith("::"):
            function_name = function_name[2:]
        if function_name != "dim3" or len(args) > 3:
            return None

        padded_args = list(args) + ["1"] * (3 - len(args))
        return f"vec3<u32>({', '.join(padded_args)})"

    def format_hip_dim3_default_initializer(self, node):
        if not self.is_dim3_variable_declaration(node):
            return None
        if getattr(node, "value", None) is not None:
            return None
        return self.format_hip_dim3_constructor_call("dim3", [])

    def format_hip_dim3_brace_initializer(self, node):
        if not self.is_dim3_variable_declaration(node):
            return None

        value = getattr(node, "value", None)
        if not isinstance(value, InitializerListNode) or len(value.elements) > 3:
            return None

        args = [self.visit(element) for element in value.elements]
        return self.format_hip_dim3_constructor_call("dim3", args)

    def is_dim3_variable_declaration(self, node):
        return self.strip_type_qualifiers(getattr(node, "vtype", "")) == "dim3"

    def format_lambda_call(self, args):
        if not args:
            return "lambda()"

        rendered_args = [self.format_lambda_parameter(arg) for arg in args[:-1]] + [
            self.format_lambda_body(args[-1])
        ]
        return f"lambda({', '.join(rendered_args)})"

    def format_lambda_parameter(self, arg):
        if isinstance(arg, VariableNode):
            if arg.vtype:
                param_type = self.convert_hip_type_to_crossgl(arg.vtype)
                return f"{param_type} {arg.name}"
            return arg.name
        return self.format_lambda_body(arg)

    def format_lambda_body(self, arg):
        if isinstance(arg, str):
            return arg
        return self.visit(arg)

    def format_atomic_argument(self, arg, index):
        if index == 0 and isinstance(arg, UnaryOpNode) and arg.op == "&":
            return self.visit(arg.operand)
        return self.visit(arg)

    def visit_AtomicOperationNode(self, node):
        args = [self.format_atomic_argument(arg, i) for i, arg in enumerate(node.args)]
        args_str = ", ".join(args)
        operation = node.operation
        scope = None
        for suffix, scope_name in (("_block", "block"), ("_system", "system")):
            if operation.endswith(suffix):
                operation = operation[: -len(suffix)]
                scope = scope_name
                break

        crossgl_func = self.convert_hip_builtin_function(operation)
        expression = f"{crossgl_func}({args_str})"
        if scope == "system":
            return self.refuse_import(
                f"{self.IMPORT_REFUSAL_TYPE.label} {node.operation}: CrossGL atomics are device-scoped, and a system-scoped atomic is also "
                "atomic for the host and peer devices",
                expression,
            )
        # A block-scoped atomic is atomic for its block; the device-scoped CrossGL atomic is atomic for every
        # thread on the device, which includes the block.
        return expression

    def is_get_method_call(self, node):
        return (
            isinstance(getattr(node, "name", None), MemberAccessNode)
            and node.name.member == "get"
            and not getattr(node, "args", [])
            and self.is_unique_ptr_expression(node.name.object)
        )

    def format_make_unique_call(self, function_name, args):
        base_name, template_args = self.parse_cpp_template(function_name)
        if not self.is_std_make_unique_base_name(base_name) or not template_args:
            return None

        target_type, is_array = self.unwrap_array_template_type(template_args[0])
        target_type = self.convert_hip_type_to_crossgl(target_type)
        args_str = ", ".join(args)
        if is_array:
            return f"new_array<{target_type}>({args_str})"
        return f"new<{target_type}>({args_str})"

    def format_unique_ptr_constructor_call(self, function_name, args):
        base_name, _ = self.parse_cpp_template(function_name)
        if len(args) != 1:
            return None
        if not self.is_std_unique_ptr_base_name(
            base_name
        ) and not self.is_unique_ptr_type_name(function_name):
            return None

        return args[0]

    def visit_DeleteNode(self, node):
        target = self.visit(node.expression)
        form = "delete[]" if node.is_array else "delete"
        self.refuse_import(
            f"{self.IMPORT_REFUSAL_TYPE.label} {form} {target}: CrossGL has no device heap"
        )

    def visit_TypeAliasNode(self, node):
        self.register_type_alias(node.name, node.alias_type)
        if self.is_decltype_type_name(node.alias_type):
            alias_type = node.alias_type
        else:
            alias_type = self.convert_hip_type_to_crossgl(node.alias_type)
        if self.is_concrete_crossgl_alias_type(alias_type):
            self.register_concrete_type_alias(node.name, alias_type)
            if self.ERASE_CONCRETE_TYPE_ALIASES:
                return
        if self.indent_level == 0:
            alias_type = self.normalize_top_level_type_alias(alias_type)
        # An alias named ``let`` is declared as ``let_``, the name its uses get.
        self.emit(f"typedef {alias_type} {self.sanitize_identifier_name(node.name)};")

    def normalize_top_level_type_alias(self, alias_type):
        """Keep C++ trait aliases reparsable by CrossGL's top-level typedef parser."""
        member_alias = self.convert_type_member_alias(alias_type)
        if member_alias is not None:
            return member_alias

        template_alias = self.convert_reparsable_std_template_alias(alias_type)
        if template_alias is not None:
            return template_alias

        scoped_alias = self.convert_reparsable_scoped_alias(alias_type)
        if scoped_alias is not None:
            return scoped_alias

        return alias_type

    def convert_reparsable_scoped_alias(self, alias_type):
        if not isinstance(alias_type, str):
            return None

        alias_type = alias_type.strip()
        if "<" in alias_type or ">" in alias_type or "::" not in alias_type:
            return None
        if not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+",
            alias_type,
        ):
            return None

        return "ptr<void>"

    def convert_type_member_alias(self, alias_type):
        parsed = self.parse_cpp_template_member_type(alias_type)
        if parsed is None:
            return None

        base_name, template_args = parsed
        if not base_name.startswith("std::"):
            return None

        alias_base = f"{base_name.rsplit('::', 1)[-1]}_t"
        converted_args = [
            self.convert_top_level_alias_template_arg(arg) for arg in template_args
        ]
        return f"{alias_base}<{', '.join(converted_args)}>"

    def convert_reparsable_std_template_alias(self, alias_type):
        base_name, template_args = self.parse_cpp_template(alias_type)
        if not template_args or not base_name.startswith("std::"):
            return None

        alias_base = base_name.rsplit("::", 1)[-1]
        converted_args = [
            self.convert_top_level_alias_template_arg(arg) for arg in template_args
        ]
        return f"{alias_base}<{', '.join(converted_args)}>"

    def convert_top_level_alias_template_arg(self, arg):
        member_alias = self.convert_type_member_alias(arg)
        if member_alias is not None:
            return member_alias

        template_alias = self.convert_reparsable_std_template_alias(arg)
        if template_alias is not None:
            return template_alias

        return self.convert_hip_type_to_crossgl(arg)

    def parse_cpp_template_member_type(self, text):
        if not isinstance(text, str):
            return None

        text = text.strip()
        start = text.find("<")
        if start == -1:
            return None

        depth = 0
        end = None
        for index in range(start, len(text)):
            char = text[index]
            if char == "<":
                depth += 1
            elif char == ">":
                depth -= 1
                if depth == 0:
                    end = index
                    break

        if end is None or text[end + 1 :].strip() != "::type":
            return None

        base_name = text[:start].strip()
        template_args = self.split_cpp_template_args(text[start + 1 : end])
        return base_name, template_args

    def visit_MemberAccessNode(self, node):
        self.validate_record_layout_use(
            self.infer_record_expression_definition(node.object),
            f"member access '{node.member}'",
        )
        if getattr(node, "is_pointer", False):
            direct_record = self.infer_direct_record_expression_definition(node.object)
            if direct_record is not None:
                issue = self.record_operator_issue(direct_record, "->")
                self.raise_record_semantic(issue, "pointer-member expression")
        static_member = self.format_record_static_member_access(node)
        if static_member is not None:
            return static_member
        self.validate_union_member_access(node)
        if self.suppress_device_property_member_access == 0:
            property_expression = self.format_hip_device_property_member_read(node)
            if property_expression is not None:
                return property_expression
        if self.suppress_device_query_value_access == 0:
            member_query_expression = self.format_hip_member_query_read(node)
            if member_query_expression is not None:
                return member_query_expression

        obj = self.visit(node.object)
        operator = "->" if getattr(node, "is_pointer", False) else "."
        if node.member == "x" and operator == "." and self.is_vector1_name(node.object):
            return obj
        return f"{obj}{operator}{node.member}"

    def format_hip_device_property_member_read(self, node):
        if not isinstance(getattr(node, "object", None), str):
            return None

        object_name = node.object
        object_type = self.lookup_variable_type(object_name)
        if not self.is_hip_device_property_object_type(
            object_type, getattr(node, "is_pointer", False)
        ):
            return None

        return self.visit_HipDevicePropertyNode(
            HipDevicePropertyNode(
                node.member, self.lookup_device_property_source(object_name)
            )
        )

    def is_hip_device_property_object_type(self, object_type, is_pointer_access):
        if is_pointer_access:
            return object_type == "ptr<hipDeviceProp_t>"
        return object_type == "hipDeviceProp_t"

    def format_hip_member_query_read(self, node):
        path = self.get_member_query_path(node)
        if path is None:
            return None

        object_name, member_path = path
        source = self.lookup_member_query_source(object_name)
        if not source:
            return None

        query_name = source.get(member_path)
        if query_name is None:
            return None

        return f"(/* HIP device query: {query_name} */ 0)"

    def format_member_query_index(self, index):
        if isinstance(index, int):
            return str(index)
        if isinstance(index, str) and index.isdigit():
            return index
        return None

    def visit_ArrayAccessNode(self, node):
        self.validate_record_subscript(node)
        if self.suppress_device_query_value_access == 0:
            member_query_expression = self.format_hip_member_query_read(node)
            if member_query_expression is not None:
                return member_query_expression

        array = self.visit(node.array)
        index = self.visit(node.index)
        return f"{array}[{index}]"

    def visit_InitializerListNode(self, node):
        elements = ", ".join(self.visit(element) for element in node.elements)
        return f"{{{elements}}}"

    def visit_DesignatedInitializerNode(self, node):
        designators = []
        for kind, target in node.designators:
            if kind == "index":
                designators.append(f"[{self.visit(target)}]")
            else:
                designators.append(f".{target}")

        value = self.visit(node.value)
        return f"{''.join(designators)} = {value}"

    def visit_HipAsmNode(self, node):
        volatility = " volatile" if node.is_volatile else ""
        self.refuse_import(
            f"{self.IMPORT_REFUSAL_TYPE.label} inline asm{volatility}({node.template}): CrossGL has no inline assembly"
        )

    def format_hip_asm_operands(self, operands):
        formatted = []
        for operand in operands:
            prefix = (
                f"[{operand.symbolic_name}] "
                if operand.symbolic_name is not None
                else ""
            )
            expression = (
                self.visit_source_comment_expression(operand.expression)
                if operand.expression is not None
                else None
            )
            if expression is None:
                formatted.append(f"{prefix}{operand.constraint}")
            else:
                formatted.append(f"{prefix}{operand.constraint}({expression})")
        return ", ".join(formatted)

    def visit_HipBuiltinNode(self, node):
        # HIP's warpSize is the device's wavefront width, 32 or 64 lanes (HIP programming guide, "Warp
        # cross-lane functions"), not a constant: it is the subgroup size of the target, as an int.
        builtin_map = {
            "threadIdx": "gl_LocalInvocationID",
            "blockIdx": "gl_WorkGroupID",
            "gridDim": "gl_NumWorkGroups",
            "blockDim": "gl_WorkGroupSize",
            "warpSize": "int(gl_SubgroupSize)",
        }

        # The parser builds this node only for the five built-ins builtin_map covers, flat ones included.
        base_name = builtin_map[node.builtin_name]
        if hasattr(node, "component") and node.component:
            return f"{base_name}.{node.component}"
        else:
            return base_name
